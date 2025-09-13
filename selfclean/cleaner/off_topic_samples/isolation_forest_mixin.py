from typing import Tuple

import numpy as np
from pyod.models.iforest import IForest

from ...cleaner.off_topic_samples.base_off_topic_mixin import BaseOffTopicMixin
from ...core.src.utils.plotting import plot_dist


class IsolationForestOffTopicMixin(BaseOffTopicMixin):
    """
    Off-topic sample detection using Isolation Forest from PyOD.

    Based on the baseline implementation from external_code/mse-cleaning-audio.
    Uses IsolationForest to detect outliers/irrelevant samples in embedding space.
    """

    def __init__(self, random_state: int = 42, **kwargs):
        """
        Initialize IsolationForest mixin.

        Args:
            random_state (int): Random state for reproducible results.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self.random_state = random_state

    def get_off_topic_ranking(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect off-topic samples using Isolation Forest.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (scores, indices) where higher scores
            indicate more likely off-topic samples.
        """
        # Initialize Isolation Forest model
        model = IForest(random_state=self.random_state)

        # Fit the model on embeddings
        model.fit(self.emb_space)

        # Get outlier scores (higher scores = more outlier-like)
        outlier_scores = model.decision_scores_

        # Create (score, index) pairs and sort by score (descending)
        off_topic_samples = [(outlier_scores[i], i) for i in range(len(outlier_scores))]
        off_topic_samples = sorted(
            off_topic_samples,
            key=lambda tup: tup[0],
            reverse=True,
        )

        if self.plot_distribution and off_topic_samples is not None:
            plot_dist(
                scores=np.asarray([x[0] for x in off_topic_samples]),
                title="Distribution of off-topic samples (Isolation Forest)",
            )

        off_topic_scores = np.asarray([x[0] for x in off_topic_samples])
        off_topic_indices = np.asarray([x[1] for x in off_topic_samples])

        return off_topic_scores, off_topic_indices
