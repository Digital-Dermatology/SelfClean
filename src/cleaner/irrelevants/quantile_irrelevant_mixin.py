from typing import Tuple

import numpy as np

from ...cleaner.irrelevants.base_irrelevant_mixin import BaseIrrelevantMixin
from ...ssl_library.src.utils.logging import plot_dist


class QuantileIrrelevantMixin(BaseIrrelevantMixin):
    def __init__(self, quantile: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.quantile = quantile

    def get_irrelevant_ranking(self) -> Tuple[np.ndarray, np.ndarray]:
        irrelevants = np.quantile(self.distance_matrix, self.quantile, axis=0)
        irrelevants = [(irrelevants[i], i) for i in list(range(self.N))]
        irrelevants = sorted(
            irrelevants,
            key=lambda tup: tup[0],
            reverse=True,
        )

        if self.plot_distribution and irrelevants is not None:
            plot_dist(
                scores=np.asarray([x[0] for x in irrelevants]),
                title="Distribution of irrelevant samples",
            )
        irrelevant_scores = np.asarray([x[0] for x in irrelevants])
        irrelevant_indices = np.asarray([x[1] for x in irrelevants])
        return irrelevant_scores, irrelevant_indices
