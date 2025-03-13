import warnings
from typing import Optional, Tuple

import numpy as np

from ...cleaner.label_errors.base_label_error_mixin import BaseLabelErrorMixin
from ...core.src.utils.plotting import plot_dist
from ...utils.utils import has_same_label


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
        o_hot_same = L.astype(np.float16)
        o_hot_diff = (~L).astype(np.float16)
        # set non-equal values to inf for multiplication
        o_hot_same[o_hot_same == 0.0] = np.inf
        o_hot_diff[o_hot_diff == 0.0] = np.inf
        # ensure one can not choose it's own distance
        np.fill_diagonal(o_hot_same, np.inf)
        # calc. the matrices for same and other lbl dists.
        with np.errstate(all="ignore"):
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

    def get_label_error_ranking(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.labels is None:
            warnings.warn("Can't find label errors without having access to labels.")
            return None, None
        if len(np.unique(self.labels)) == 1:
            warnings.warn("Can't detect label errors with only one label.")
            return None, None

        label_errors = self.labels_calc_scores()
        label_errors = [(label_errors[i], i) for i in list(range(self.N))]
        label_errors = sorted(
            label_errors,
            key=lambda tup: tup[0],
            reverse=False,
        )

        if self.plot_distribution:
            plot_dist(
                scores=np.asarray([x[0] for x in label_errors]),
                title="Distribution of possible label errors",
            )
        label_error_scores = np.asarray([x[0] for x in label_errors])
        label_error_indices = np.asarray([x[1] for x in label_errors])
        return label_error_scores, label_error_indices
