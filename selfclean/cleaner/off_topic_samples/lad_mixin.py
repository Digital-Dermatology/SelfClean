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
