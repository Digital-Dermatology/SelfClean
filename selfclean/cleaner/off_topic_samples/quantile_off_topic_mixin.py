from typing import Tuple

import numpy as np

from ...cleaner.off_topic_samples.base_off_topic_mixin import BaseOffTopicMixin
from ...core.src.utils.plotting import plot_dist


class QuantileOffTopicMixin(BaseOffTopicMixin):
    def __init__(self, quantile: float = 0.01, **kwargs):
        super().__init__(**kwargs)
        self.quantile = quantile

    def get_off_topic_ranking(self) -> Tuple[np.ndarray, np.ndarray]:
        off_topic_samples = np.quantile(self.distance_matrix, self.quantile, axis=0)
        off_topic_samples = [(off_topic_samples[i], i) for i in list(range(self.N))]
        off_topic_samples = sorted(
            off_topic_samples,
            key=lambda tup: tup[0],
            reverse=True,
        )

        if self.plot_distribution and off_topic_samples is not None:
            plot_dist(
                scores=np.asarray([x[0] for x in off_topic_samples]),
                title="Distribution of off-topic samples",
            )
        off_topic_scores = np.asarray([x[0] for x in off_topic_samples])
        off_topic_indices = np.asarray([x[1] for x in off_topic_samples])
        return off_topic_scores, off_topic_indices
