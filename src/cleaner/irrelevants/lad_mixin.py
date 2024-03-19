from typing import List, Tuple

import numpy as np
from scipy.cluster.hierarchy import single

from src.cleaner.irrelevants.base_irrelevant_mixin import BaseIrrelevantMixin
from src.scoring.lad_scoring import LAD
from ssl_library.src.utils.logging import plot_dist


class LADIrrelevantMixin(BaseIrrelevantMixin):
    def __init__(self, global_leaves: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.global_leaves = global_leaves

    def get_irrelevant_ranking(self) -> List[Tuple[float, int]]:
        # linkage_matrix: [idx1, idx2, dist, sample_count]
        linkage_matrix = single(self.p_distances)
        lad = LAD()
        irrelevant_score = lad.calc_scores(
            linkage_matrix=linkage_matrix,
            global_leaves=self.global_leaves,
        )
        # free up allocated memory
        del lad, linkage_matrix

        if self.plot_distribution and irrelevant_score is not None:
            plot_dist(
                scores=np.asarray([x[0] for x in irrelevant_score]),
                title="Distribution of irrelevant samples",
            )
        return irrelevant_score
