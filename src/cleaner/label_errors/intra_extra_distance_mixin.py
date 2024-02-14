import warnings
from typing import List, Optional, Tuple

import numpy as np

from src.cleaner.label_errors.base_label_error_mixin import BaseLabelErrorMixin
from src.utils.utils import has_same_label
from ssl_library.src.utils.logging import plot_dist


class IntraExtraDistanceLabelErrorMixin(BaseLabelErrorMixin):
    def labels_calc_scores(self) -> np.ndarray:
        """
        Calculate intra-/ extra distance ratio.

        The samples which have a smaller distance to a sample with an other label
        than to their own have a larger score.
        """
        assert self.labels is not None
        # create a int matrix of same label relation
        L = has_same_label(self.labels)
        o_hot_same = L.astype(float)
        o_hot_diff = (~L).astype(float)
        # set non-equal values to inf for multiplication
        o_hot_same[o_hot_same == 0.0] = np.inf
        o_hot_diff[o_hot_diff == 0.0] = np.inf
        # ensure one can not choose it's own distance
        np.fill_diagonal(o_hot_same, np.inf)
        # calc. the matrices for same and other lbl dists.
        min_same = np.nanmin((o_hot_same * self.distance_matrix), axis=-1)
        min_diff = np.nanmin((o_hot_diff * self.distance_matrix), axis=-1)
        # check if there are samples without any same labels
        missing_same_indices = np.where(np.sum(o_hot_same == 1, axis=-1) == 0)[0]
        if len(missing_same_indices) > 0:
            # if yes set them to the maximum of the other distances
            val_missing = (o_hot_diff * self.distance_matrix)[missing_same_indices]
            min_same[missing_same_indices] = (
                np.ma.masked_invalid(val_missing).max(axis=-1).data
            )
        # calc. score matrix
        lbl_scores = (min_diff**2) / (min_same**2 + min_diff**2)
        return lbl_scores

    def get_label_error_ranking(self) -> Optional[List[Tuple[float, int]]]:
        if self.labels is None:
            warnings.warn("Can't find label errors without having access to labels.")
            return None
        if len(np.unique(self.labels)) == 1:
            warnings.warn("Can't detect label errors with only one label.")
            return None

        label_error_scores = self.labels_calc_scores()
        label_error_scores = [(label_error_scores[i], i) for i in list(range(self.N))]
        label_error_scores = sorted(
            label_error_scores,
            key=lambda tup: tup[0],
            reverse=False,
        )

        if self.plot_distribution:
            plot_dist(
                scores=label_error_scores,
                title="Distribution of possible label errors",
            )
        return label_error_scores
