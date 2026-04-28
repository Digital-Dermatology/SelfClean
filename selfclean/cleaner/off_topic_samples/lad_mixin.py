from typing import Tuple

import numpy as np
from scipy.cluster.hierarchy import single

from ...cleaner.off_topic_samples.base_off_topic_mixin import BaseOffTopicMixin
from ...scoring.lad_scoring import LAD
from ...core.src.utils.plotting import plot_dist


class LADOffTopicMixin(BaseOffTopicMixin):
    def __init__(self, global_leaves: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.global_leaves = global_leaves

    def get_off_topic_ranking(self) -> Tuple[np.ndarray, np.ndarray]:
        # linkage_matrix: [idx1, idx2, dist, sample_count]
        # Exact path: single-linkage dendrogram from the full condensed
        # distance vector. Approximate path: same dendrogram structure but
        # built from the MST of the cached KNN graph (built in
        # `SelfCleanCleaner._build_knn_linkage`). LAD scoring is unchanged
        # in both modes; only the source of the linkage matrix differs.
        if getattr(self, "approximate_nn", False):
            if not hasattr(self, "knn_linkage_matrix"):
                raise RuntimeError(
                    "Approximate off-topic ranking requires `_build_knn_index` "
                    "to have been called by `fit`. Pass `approximate_nn=True` "
                    "to the cleaner constructor and re-fit."
                )
            linkage_matrix = self.knn_linkage_matrix
        else:
            linkage_matrix = single(self.p_distances)
        lad = LAD()
        off_topic_samples = lad.calc_scores(
            linkage_matrix=linkage_matrix,
            global_leaves=self.global_leaves,
        )
        # free up allocated memory
        del lad, linkage_matrix

        if self.plot_distribution and off_topic_samples is not None:
            plot_dist(
                scores=np.asarray([x[0] for x in off_topic_samples]),
                title="Distribution of off-topic samples",
            )
        off_topic_scores = np.asarray([x[0] for x in off_topic_samples])
        off_topic_indices = np.asarray([x[1] for x in off_topic_samples])
        return off_topic_scores, off_topic_indices
