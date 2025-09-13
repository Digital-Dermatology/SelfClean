from typing import Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from ...cleaner.off_topic_samples.base_off_topic_mixin import BaseOffTopicMixin
from ...core.src.utils.plotting import plot_dist


class CleanLabOffTopicMixin(BaseOffTopicMixin):
    """
    Supervised off-topic ranking using cross-validated probabilities.
    Returns a continuous score for every sample (no thresholding).
    """

    def __init__(
        self,
        C: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-2,
        random_state: int = 42,
        cv_folds: int = 5,
        **kwargs,
    ):
        """
        Initialize CleanLab mixin.

        Args:
            C (float): Regularization parameter for LogisticRegression.
            max_iter (int): Maximum iterations for LogisticRegression.
            tol (float): Tolerance for LogisticRegression.
            random_state (int): Random state for reproducible results.
            cv_folds (int): Number of cross-validation folds.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cv_folds = cv_folds

    def get_off_topic_ranking(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rank off-topic samples based on 1 - max predicted probability.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (scores, indices) where higher scores
            indicate more likely off-topic samples.
        """
        # Convert labels to numeric if they're not already
        if hasattr(self, "labels") and self.labels is not None:
            unique_labels = np.unique(self.labels)
            if isinstance(unique_labels[0], str):
                # Create label mapping
                label_to_num = {label: i for i, label in enumerate(unique_labels)}
                numeric_labels = np.array(
                    [label_to_num[label] for label in self.labels]
                )
            else:
                numeric_labels = self.labels
        else:
            # If no labels available, we can't use CleanLab's supervised approach
            # Fall back to unsupervised outlier detection based on embedding distances
            return self._unsupervised_outlier_detection()

        # Train LogisticRegression model with cross-validation
        model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )

        # Get out-of-sample predictions using cross-validation
        cv = min(self.cv_folds, len(numeric_labels))
        if cv < 2:
            # Fallback: fit once and use in-sample probabilities
            model.fit(self.emb_space, numeric_labels)
            pred_probs = model.predict_proba(self.emb_space)
        else:
            pred_probs = cross_val_predict(
                estimator=model,
                X=self.emb_space,
                y=numeric_labels,
                cv=cv,
                method="predict_proba",
            )

        # Continuous score for every sample: lower max prob => more off-topic
        max_probs = np.max(pred_probs, axis=1)
        off_topic_scores = 1.0 - max_probs

        # Return full ranking (descending by score)
        sort_idx = np.argsort(off_topic_scores)[::-1]
        sorted_scores = off_topic_scores[sort_idx]
        sorted_indices = np.arange(len(off_topic_scores))[sort_idx]

        if self.plot_distribution and len(sorted_scores) > 0:
            plot_dist(
                scores=sorted_scores,
                title="Distribution of off-topic samples (supervised)",
            )

        return sorted_scores, sorted_indices

    def _unsupervised_outlier_detection(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fallback unsupervised outlier detection when labels are not available.
        Uses distance-based approach similar to quantile method.
        """
        # Calculate mean distance to all other samples for each sample
        distances = np.mean(self.distance_matrix, axis=1)

        # Create (score, index) pairs and sort by distance (descending)
        off_topic_samples = [(distances[i], i) for i in range(len(distances))]
        off_topic_samples = sorted(
            off_topic_samples,
            key=lambda tup: tup[0],
            reverse=True,
        )

        off_topic_scores = np.asarray([x[0] for x in off_topic_samples])
        off_topic_indices = np.asarray([x[1] for x in off_topic_samples])

        return off_topic_scores, off_topic_indices
