import math
import tempfile
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import scienceplots  # noqa: F401
import scipy
import scipy.stats
import sklearn  # noqa: F401
from tqdm import tqdm

import src.distances  # noqa: F401
import src.distances.projective_distance  # noqa: F401
from src.cleaner.base_cleaner import BaseCleaner
from src.cleaner.irrelevants.lad_mixin import LADIrrelevantMixin
from src.cleaner.label_errors.intra_extra_distance_mixin import (
    IntraExtraDistanceLabelErrorMixin,
)
from src.cleaner.near_duplicates.embedding_distance_mixin import EmbeddingDistanceMixin
from src.utils.utils_proba import get_scale_loc
from utils.plotting import (
    plot_frac_cut,
    plot_inspection_result,
    plot_sensitivity,
    subplot_frac_cut,
    subplot_sensitivity,
)


class SelfCleanCleaner(
    BaseCleaner,
    IntraExtraDistanceLabelErrorMixin,
    EmbeddingDistanceMixin,
    LADIrrelevantMixin,
):
    def __init__(
        self,
        # cleaning
        auto_cleaning: bool = False,
        near_duplicate_cut_off: float = 0.01,
        irrelevant_cut_off: float = 0.01,
        label_error_cut_off: float = 0.01,
        cleaner_kwargs: dict = {},
        # distance calculation
        distance_function_path: str = "sklearn.metrics.pairwise.",
        distance_function_name: str = "cosine_similarity",
        chunk_size: int = 100,
        precision_type_distance: type = np.float32,
        # memory management
        memmap: bool = True,
        memmap_path: Union[Path, str, None] = None,
        # plotting
        plot_top_N: Optional[int] = None,
        output_path: Optional[str] = None,
        figsize: tuple = (10, 8),
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

    def predict(
        self,
        plot_top_N: Optional[int] = None,
        figsize: tuple = (10, 8),
        auto_cleaning: bool = False,
        cleaner_kwargs: dict = {},
        near_duplicate_cut_off: float = 0.01,
        irrelevant_cut_off: float = 0.01,
        label_error_cut_off: float = 0.01,
    ):
        pred_nd = self.get_near_duplicate_ranking()
        pred_nd_scores = np.asarray([x[0] for x in pred_nd])
        pred_nd_indices = np.asarray([x[1] for x in pred_nd])

        pred_oods = self.get_irrelevant_ranking()
        pred_oods_scores = np.asarray([x[0] for x in pred_oods])
        pred_oods_indices = np.asarray([x[1] for x in pred_oods])

        # TODO: make this nicer
        pred_lbl_errs = self.get_label_error_ranking()
        # check if label errors should be skipped
        skip_lbl_errs = False
        if pred_lbl_errs is not None:
            pred_lbl_errs_scores = np.asarray([x[0] for x in pred_lbl_errs])
            pred_lbl_errs_indices = np.asarray([x[1] for x in pred_lbl_errs])
        else:
            skip_lbl_errs = True
        del pred_oods, pred_lbl_errs

        # TODO: create a mixin with the autocleaning functionality
        if auto_cleaning:
            # Near Duplicates
            if self.output_path is not None:
                cleaner_kwargs[
                    "path"
                ] = f"{self.output_path.stem}_auto_dups{self.output_path.suffix}"
            cleaner_kwargs["alpha"] = near_duplicate_cut_off
            issues_dup = self.fraction_cut(
                scores=pred_nd_scores,
                **cleaner_kwargs,
            )

            # Irrelevant Samples
            if self.output_path is not None:
                cleaner_kwargs[
                    "path"
                ] = f"{self.output_path.stem}_auto_oods{self.output_path.suffix}"
            cleaner_kwargs["alpha"] = irrelevant_cut_off
            issues_ood = self.fraction_cut(
                scores=pred_oods_scores,
                **cleaner_kwargs,
            )

            # Label Errors
            if not skip_lbl_errs:
                if self.output_path is not None:
                    cleaner_kwargs[
                        "path"
                    ] = f"{self.output_path.stem}_auto_lbls{self.output_path.suffix}"
                cleaner_kwargs["alpha"] = label_error_cut_off
                issues_lbl = self.fraction_cut(
                    scores=pred_lbl_errs_scores,
                    **cleaner_kwargs,
                )

        if plot_top_N is not None and self.images is not None:
            plot_inspection_result(
                pred_dups_indices=pred_nd_indices,
                pred_oods_indices=pred_oods_indices,
                pred_lbl_errs_indices=pred_lbl_errs_indices,
                images=self.images,
                labels=self.labels,
                class_labels=self.class_labels,
                skip_lbl_errs=skip_lbl_errs,
                plot_top_N=plot_top_N,
                output_path=self.output_path,
                figsize=figsize,
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
        }
        if not skip_lbl_errs:
            return_dict["label_errors"] = {
                "indices": pred_lbl_errs_indices,
                "scores": pred_lbl_errs_scores,
            }

        if auto_cleaning:
            return_dict["near_duplicates"]["auto_issues"] = issues_dup
            return_dict["irrelevants"]["auto_issues"] = issues_ood
            if not skip_lbl_errs:
                return_dict["label_errors"]["auto_issues"] = issues_lbl
        return return_dict

    def fraction_cut(
        self,
        scores: np.ndarray,
        alpha: float = 0.01,
        q: float = 0.05,
        dist=scipy.stats.logistic,
        plot_result: bool = False,
        ax=None,
        bins="sqrt",
        debug: bool = False,
        path: Optional[str] = None,
    ):
        M = len(scores)
        if M == self.condensed_size:
            # scale alpha for duplicates
            alpha = alpha**2
        # only consider the point in range [0,1]
        _scores = scores[(scores > 0) & (scores < 1)]
        # logit transform
        logit_scores = np.log(_scores / (1 - _scores))

        # calculate the quantiles
        p = alpha
        prob = q * p * self.N / M
        q1 = np.quantile(logit_scores, p)
        q2 = np.quantile(logit_scores, (0.5 * p) ** 0.5)

        # calculate the cut-off
        scale, loc = get_scale_loc(dist, logit_scores, p, (0.5 * p) ** 0.5)
        cutoff = dist.ppf(prob) * scale + loc

        # Exclude the scores below probability threshold
        exclude = logit_scores < cutoff
        n = exclude.sum()
        if debug:
            print(f"{n} outliers ({n/self.N:.1%})")

        if plot_result:
            if ax is not None:
                subplot_frac_cut(
                    ax,
                    logit_scores,
                    bins,
                    q1,
                    q2,
                    cutoff,
                    dist,
                    loc,
                    scale,
                )
            else:
                plot_frac_cut(
                    dist,
                    logit_scores,
                    bins,
                    q1,
                    q2,
                    cutoff,
                    loc,
                    scale,
                    path,
                )

        return np.where(exclude)[0]

    def threshold_sensitivity(self, scores: np.ndarray, ax=None):
        thresholds = 2 ** np.linspace(-10, -2, 17)
        result = np.array(
            [
                (
                    q,
                    self.fraction_cut(
                        scores=scores,
                        alpha=0.1,
                        q=q,
                        plot_result=False,
                        debug=False,
                    ).shape[0],
                )
                for q in thresholds
            ]
        )
        result[:, 1] = result[:, 1] / self.N
        if ax is not None:
            subplot_sensitivity(
                ax,
                result,
                ylabel="Fraction of detected outliers",
                xlabel=r"Significance level $q$",
            )
        else:
            plot_sensitivity(
                result,
                ylabel="Fraction of detected outliers",
                xlabel=r"Significance level $q$",
            )
        return result

    def alpha_sensitivity(self, scores: np.ndarray, ax=None):
        alphas = 2 ** np.linspace(-10, -2, 17)
        result = np.array(
            [
                (
                    a,
                    self.fraction_cut(
                        scores=scores,
                        alpha=a,
                        plot_result=False,
                        debug=False,
                    ).shape[0],
                )
                for a in alphas
            ]
        )
        result[:, 1] = result[:, 1] / self.N
        if ax is not None:
            subplot_sensitivity(
                ax,
                result,
                ylabel="Fraction of detected outliers",
                xlabel=r"Contamination rate guess $\alpha$",
            )
        else:
            plot_sensitivity(
                result,
                ylabel="Fraction of detected outliers",
                xlabel=r"Contamination rate guess $\alpha$",
            )
        return result
