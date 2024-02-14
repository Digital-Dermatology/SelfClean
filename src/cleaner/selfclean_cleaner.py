import math
import tempfile
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import scienceplots  # noqa: F401
import sklearn  # noqa: F401
from tqdm import tqdm

import src.distances  # noqa: F401
import src.distances.projective_distance  # noqa: F401
from src.cleaner.auto_cleaning_mixin import AutoCleaningMixin
from src.cleaner.base_cleaner import BaseCleaner
from src.cleaner.irrelevants.lad_mixin import LADIrrelevantMixin
from src.cleaner.label_errors.intra_extra_distance_mixin import (
    IntraExtraDistanceLabelErrorMixin,
)
from src.cleaner.near_duplicates.embedding_distance_mixin import EmbeddingDistanceMixin
from src.utils.plotting import plot_inspection_result


class SelfCleanCleaner(
    BaseCleaner,
    LADIrrelevantMixin,
    EmbeddingDistanceMixin,
    IntraExtraDistanceLabelErrorMixin,
    AutoCleaningMixin,
):
    def __init__(
        self,
        # distance calculation
        distance_function_path: str = "sklearn.metrics.pairwise.",
        distance_function_name: str = "cosine_similarity",
        chunk_size: int = 100,
        precision_type_distance: type = np.float32,
        # memory management
        memmap: bool = True,
        memmap_path: Union[Path, str, None] = None,
        # plotting
        plot_distribution: bool = False,
        plot_top_N: Optional[int] = None,
        output_path: Optional[str] = None,
        figsize: tuple = (10, 8),
        **kwargs,
    ):
        self.memmap = memmap
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
        super().__init__(**kwargs)

    def fit(
        self,
        emb_space: np.ndarray,
        labels: Optional[np.ndarray] = None,
        images: Optional[np.ndarray] = None,
        paths: Optional[np.ndarray] = None,
        class_labels: Optional[list] = None,
    ):
        self.labels = labels
        self.images = images
        self.paths = paths
        self.class_labels = class_labels
        self.N, self.D = emb_space.shape
        self.condensed_size = int(self.N * ((self.N - 1) / 2))

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
        for i in tqdm(
            range(n_chunks),
            desc="Creating distance matrix",
            total=n_chunks,
            position=0,
            leave=True,
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
        else:
            self.p_distances = np.zeros(
                shape=(self.condensed_size,),
                dtype=self.precision_type_distance,
            )
        self.p_distances[:] = self.distance_matrix[
            ~np.tril(np.ones((self.N, self.N), dtype=bool))
        ]
        return self

    def predict(self) -> dict:
        pred_nd = self.get_near_duplicate_ranking()
        pred_nd_scores = np.asarray([x[0] for x in pred_nd])
        pred_nd_indices = np.asarray([x[1] for x in pred_nd])
        del pred_nd

        pred_oods = self.get_irrelevant_ranking()
        pred_oods_scores = np.asarray([x[0] for x in pred_oods])
        pred_oods_indices = np.asarray([x[1] for x in pred_oods])
        del pred_oods

        pred_lbl_errs = self.get_label_error_ranking()
        if pred_lbl_errs is not None:
            pred_lbl_errs_scores = np.asarray([x[0] for x in pred_lbl_errs])
            pred_lbl_errs_indices = np.asarray([x[1] for x in pred_lbl_errs])
        else:
            pred_lbl_errs_scores = None
            pred_lbl_errs_indices = None
        del pred_lbl_errs

        if self.plot_top_N is not None and self.images is not None:
            plot_inspection_result(
                pred_dups_indices=pred_nd_indices,
                pred_oods_indices=pred_oods_indices,
                pred_lbl_errs_indices=pred_lbl_errs_indices,
                images=self.images,
                labels=self.labels,
                class_labels=self.class_labels,
                plot_top_N=self.plot_top_N,
                output_path=self.output_path,
                figsize=self.figsize,
            )

        return_dict = {
            "irrelevants": {
                "indices": pred_oods_indices,
                "scores": pred_oods_scores,
            },
            "near_duplicates": {
                "indices": pred_nd_indices,
                "scores": pred_nd_scores,
            },
            "label_errors": {
                "indices": pred_lbl_errs_indices,
                "scores": pred_lbl_errs_scores,
            },
        }

        return_dict = self.perform_auto_cleaning(
            return_dict=return_dict,
            pred_near_duplicate_scores=pred_nd_scores,
            pred_irrelevant_scores=pred_oods_scores,
            pred_label_error_scores=pred_lbl_errs_scores,
            output_path=self.output_path,
        )
        return return_dict
