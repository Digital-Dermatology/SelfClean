from typing import List, Tuple

import numpy as np

from ...cleaner.irrelevants.base_irrelevant_mixin import BaseIrrelevantMixin
from ...ssl_library.src.utils.logging import plot_dist
from ...utils.plotting import plot_irrelevant_samples


class QuantileIrrelevantMixin(BaseIrrelevantMixin):
    def __init__(self, quantile: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.quantile = quantile

    def get_irrelevant_ranking(self) -> List[Tuple[float, int]]:
        irrelevant_score = np.quantile(self.distance_matrix, self.quantile, axis=0)
        irrelevant_score = [(irrelevant_score[i], i) for i in list(range(self.N))]
        irrelevant_score = sorted(
            irrelevant_score,
            key=lambda tup: tup[0],
            reverse=True,
        )

        if self.plot_top_N is not None and self.images is not None:
            plot_irrelevant_samples(
                irrelevant_score=irrelevant_score,
                images=self.images,
                plot_top_N=self.plot_top_N,
                plot_title=self.plot_title,
                return_fig=self.return_fig,
            )

        if self.plot_distribution and irrelevant_score is not None:
            plot_dist(
                scores=np.asarray([x[0] for x in irrelevant_score]),
                title="Distribution of irrelevant samples",
            )
        return irrelevant_score
