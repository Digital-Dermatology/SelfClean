from typing import Tuple

import numpy as np
from scipy.cluster.hierarchy import single

from ...cleaner.irrelevants.base_irrelevant_mixin import BaseIrrelevantMixin
from ...scoring.lad_scoring import LAD
from ...ssl_library.src.utils.logging import plot_dist


class LADIrrelevantMixin(BaseIrrelevantMixin):
    def __init__(self, global_leaves: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.global_leaves = global_leaves

    def get_irrelevant_ranking(self) -> Tuple[np.ndarray, np.ndarray]:
        # linkage_matrix: [idx1, idx2, dist, sample_count]
        linkage_matrix = single(self.p_distances)
        lad = LAD()
        irrelevants = lad.calc_scores(
            linkage_matrix=linkage_matrix,
            global_leaves=self.global_leaves,
        )
        # free up allocated memory
        del lad, linkage_matrix

        if self.plot_distribution and irrelevants is not None:
            plot_dist(
                scores=np.asarray([x[0] for x in irrelevants]),
                title="Distribution of irrelevant samples",
            )
        irrelevant_scores = np.asarray([x[0] for x in irrelevants])
        irrelevant_indices = np.asarray([x[1] for x in irrelevants])
        return irrelevant_scores, irrelevant_indices
