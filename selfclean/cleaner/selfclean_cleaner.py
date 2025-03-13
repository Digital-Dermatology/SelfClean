import math
import tempfile
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
import scienceplots  # noqa: F401
import sklearn  # noqa: F401
from loguru import logger
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from ..cleaner.auto_cleaning_mixin import AutoCleaningMixin
from ..cleaner.base_cleaner import BaseCleaner
from ..cleaner.issue_manager import IssueManager, IssueTypes
from ..cleaner.label_errors.intra_extra_distance_mixin import (
    IntraExtraDistanceLabelErrorMixin,
)
from ..cleaner.near_duplicates.embedding_distance_mixin import EmbeddingDistanceMixin
from ..cleaner.off_topic_samples.lad_mixin import LADOffTopicMixin
from ..distances import *  # noqa: F401, F403
from ..distances.projective_distance import *  # noqa: F401, F403
from ..core.src.utils.logging import set_log_level
from ..core.src.utils.utils import fix_random_seeds
from ..utils.plotting import plot_inspection_result
from ..utils.utils import triu_indices_memmap


class SelfCleanCleaner(
    BaseCleaner,
    LADOffTopicMixin,
    EmbeddingDistanceMixin,
    IntraExtraDistanceLabelErrorMixin,
    AutoCleaningMixin,
):
    def __init__(
        self,
        # distance calculation
        distance_function_path: str = "sklearn.metrics.pairwise.",
        distance_function_name: str = "cosine_similarity",
        chunk_size: int = 10_000,
        precision_type_distance: type = np.float32,
        # memory management
        memmap: bool = True,
        memmap_path: Union[Path, str, None] = None,
        approximate_nn: bool = False,
        # plotting
        plot_distribution: bool = False,
        plot_top_N: Optional[int] = None,
        output_path: Optional[str] = None,
        figsize: tuple = (10, 8),
        # utils
        random_seed: int = 42,
        # logging
        log_level: str = "INFO",
        **kwargs,
    ):
        self.log_level = log_level
        set_log_level(min_log_level=self.log_level)
        fix_random_seeds(seed=random_seed)

        self.memmap = memmap
        self.approximate_nn = approximate_nn
        self.chunk_size = chunk_size
        self.precision_type_distance = precision_type_distance

        self.output_path = output_path
        if self.output_path is not None:
            self.output_path = Path(self.output_path)

        if memmap_path is None:
            self.memmap_path = Path(tempfile.mkdtemp())
        else:
            self.memmap_path = Path(memmap_path)
            self.memmap_path.parent.mkdir(parents=True, exist_ok=True)

        self.distance_function_name = distance_function_name
        self.distance_function: Callable = eval(
            f"{distance_function_path}{self.distance_function_name}"
        )

        self.plot_distribution = plot_distribution
        self.plot_top_N = plot_top_N
        self.figsize = figsize
        self.is_fitted = False
        super().__init__(**kwargs)

    def fit(
        self,
        emb_space: np.ndarray,
        labels: Optional[np.ndarray] = None,
        paths: Optional[np.ndarray] = None,
        dataset: Optional[Dataset] = None,
        class_labels: Optional[list] = None,
    ):
        self.emb_space = emb_space
        self.labels = labels
        self.dataset = dataset
        self.paths = paths
        self.class_labels = class_labels
        self.N, self.D = emb_space.shape
        self.condensed_size = int(self.N * ((self.N - 1) / 2))
        logger.info(f"Fitting cleaner on representation space: {emb_space.shape}")

        if self.memmap:
            dist_file = self.memmap_path / "dist_matrix.dat"
            if dist_file.exists():
                dist_file.unlink()
            self.distance_matrix = np.memmap(
                str(dist_file),
                dtype=self.precision_type_distance,
                mode="w+",
                shape=(self.N, self.N),
            )
        else:
            self.distance_matrix = np.zeros(
                shape=(self.N, self.N),
                dtype=self.precision_type_distance,
            )

        # create the distance matrix in chunks
        n_chunks = math.ceil(self.N / self.chunk_size)
        iterator = range(n_chunks)
        for i in (
            tqdm(
                iterator,
                desc="Creating distance matrix",
                total=n_chunks,
                position=0,
                leave=True,
            )
            if self.log_level == "DEBUG"
            else iterator
        ):
            chunk_slice = slice(i * self.chunk_size, (i + 1) * self.chunk_size, 1)
            X_emb = emb_space[chunk_slice]
            distance_row = self.distance_function(
                X=X_emb,
                Y=emb_space,
            )
            distance_row = np.squeeze(distance_row)
            if self.distance_function_name == "cosine_similarity":
                # normalize and invert the cosine similarity to obtain distance
                distance_row = 1 - ((distance_row + 1) / 2)
            self.distance_matrix[chunk_slice, :] = distance_row
            del distance_row
        # clip the values to range [0, 1]
        # could be outside because of floating point inaccuracy
        np.clip(self.distance_matrix, 0.0, 1.0, out=self.distance_matrix)
        # create the condensed matrix
        if self.memmap:
            p_dist_file = self.memmap_path / "p_distances.dat"
            if p_dist_file.exists():
                p_dist_file.unlink()
            self.p_distances = np.memmap(
                str(p_dist_file),
                dtype=self.precision_type_distance,
                mode="w+",
                shape=(self.condensed_size,),
            )
            triu_indices = triu_indices_memmap(
                str(self.memmap_path / "triu_indices"),
                N=self.N,
                k=1,
            )
        else:
            self.p_distances = np.zeros(
                shape=(self.condensed_size,),
                dtype=self.precision_type_distance,
            )
            triu_indices = np.triu_indices(self.N, k=1)
        # create the upper triangular matrix of the distance matrix
        for start_idx in range(0, len(triu_indices[0]), self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, len(triu_indices[0]))
            self.p_distances[start_idx:end_idx] = self.distance_matrix[
                triu_indices[0][start_idx:end_idx], triu_indices[1][start_idx:end_idx]
            ]
        self.is_fitted = True
        del triu_indices
        return self

    def predict(
        self,
        issues_to_detect: List[IssueTypes] = [
            IssueTypes.NEAR_DUPLICATES,
            IssueTypes.OFF_TOPIC_SAMPLES,
            IssueTypes.LABEL_ERRORS,
        ],
    ) -> IssueManager:
        return_dict = {}
        if IssueTypes.NEAR_DUPLICATES in issues_to_detect:
            if not self.approximate_nn:
                pred_nd_scores, pred_nd_indices = self.get_near_duplicate_ranking()
                return_dict["near_duplicates"] = {
                    "indices": pred_nd_indices,
                    "scores": pred_nd_scores,
                }
            else:
                approx_result_df = self.get_approx_near_duplicate_ranking()
                return_dict["approx_near_duplicates"] = approx_result_df
        if IssueTypes.OFF_TOPIC_SAMPLES in issues_to_detect:
            pred_ot_scores, pred_ot_indices = self.get_off_topic_ranking()
            return_dict["off_topic_samples"] = {
                "indices": pred_ot_indices,
                "scores": pred_ot_scores,
            }
        if IssueTypes.LABEL_ERRORS in issues_to_detect:
            pred_lbl_errs_scores, pred_lbl_errs_indices = self.get_label_error_ranking()
            if pred_lbl_errs_scores is not None and pred_lbl_errs_indices is not None:
                return_dict["label_errors"] = {
                    "indices": pred_lbl_errs_indices,
                    "scores": pred_lbl_errs_scores,
                }

        if self.labels is not None:
            # transform labels using class names if given
            labels = [
                self.class_labels[x] if self.class_labels is not None else x
                for x in self.labels
            ]
        else:
            labels = self.labels
        # create the manger for the issues to pass to plotting and return
        issue_manager = IssueManager(
            issue_dict=return_dict,
            meta_data_dict={
                "path": self.paths,
                "label": labels,
            },
        )

        if self.plot_top_N is not None and self.dataset is not None:
            plot_inspection_result(
                issue_manger=issue_manager,
                dataset=self.dataset,
                labels=labels,
                plot_top_N=self.plot_top_N,
                output_path=self.output_path,
                figsize=self.figsize,
            )
        return_dict = self.perform_auto_cleaning(
            issue_manger=issue_manager,
            return_dict=return_dict,
            output_path=self.output_path,
        )
        return issue_manager
