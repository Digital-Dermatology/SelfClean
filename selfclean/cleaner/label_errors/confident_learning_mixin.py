from typing import Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from ...cleaner.label_errors.base_label_error_mixin import BaseLabelErrorMixin
from ...core.src.utils.plotting import plot_dist


class ConfidentLearningLabelErrorMixin(BaseLabelErrorMixin):
    """
    Label error detection using Confident Learning approach.

    This implementation uses cross-validation to get out-of-sample predictions
    and identifies label errors using confident learning principles without
    the broken CleanLab dependencies.
    """

    def __init__(
        self,
        C: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-2,
        random_state: int = 42,
        cv_folds: int = 5,
        confidence_threshold: float = 0.8,
        **kwargs,
    ):
        """
        Initialize Confident Learning label error mixin.

        Args:
            C (float): Regularization parameter for LogisticRegression.
            max_iter (int): Maximum iterations for LogisticRegression.
            tol (float): Tolerance for LogisticRegression.
            random_state (int): Random state for reproducible results.
            cv_folds (int): Number of cross-validation folds.
            confidence_threshold (float): Threshold for confident predictions.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.confidence_threshold = confidence_threshold

    def get_label_error_ranking(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect label errors using confident learning principles and return a full ranking.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (scores, indices) for all samples, sorted
            by descending error likelihood (higher score = more likely error).
        """
        # Check if we have labels available
        if not hasattr(self, "labels") or self.labels is None:
            print("Warning: No labels available for label error detection")
            return np.array([]), np.array([])

        # Convert labels to numeric if they're strings
        original_labels = np.array(self.labels)
        unique_labels = np.unique(original_labels)

        if len(unique_labels) < 2:
            print("Warning: Need at least 2 classes for label error detection")
            return np.array([]), np.array([])

        if isinstance(unique_labels[0], str):
            # Create label mapping
            label_to_num = {label: i for i, label in enumerate(unique_labels)}
            numeric_labels = np.array(
                [label_to_num[label] for label in original_labels]
            )
        else:
            numeric_labels = np.array(original_labels)

        # Train LogisticRegression model with cross-validation
        model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )

        # Get out-of-sample predictions using cross-validation
        try:
            cv = min(self.cv_folds, len(numeric_labels))
            if cv < 2:
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
        except Exception as e:
            print(f"Warning: Cross-validation failed: {e}")
            # Fallback: try simple fit
            try:
                model.fit(self.emb_space, numeric_labels)
                pred_probs = model.predict_proba(self.emb_space)
            except Exception:
                return np.array([]), np.array([])

        # Compute a continuous error score for every sample
        # Use 1 - P(model assigns to the given/observed label)
        given_conf = []
        for i, tl in enumerate(numeric_labels):
            tl_int = int(tl) if int(tl) < pred_probs.shape[1] else None
            given_conf.append(pred_probs[i, tl_int] if tl_int is not None else 0.0)
        given_conf = np.asarray(given_conf)
        error_scores = 1.0 - given_conf

        # Sort by error score descending and return full ranking
        sort_idx = np.argsort(error_scores)[::-1]
        sorted_scores = error_scores[sort_idx]
        sorted_indices = np.asarray(range(len(error_scores)))[sort_idx]

        if self.plot_distribution and len(sorted_scores) > 0:
            plot_dist(
                scores=sorted_scores,
                title="Distribution of label errors (Confident Learning)",
            )

        return sorted_scores, sorted_indices

    def _compute_class_thresholds(
        self, pred_probs: np.ndarray, true_labels: np.ndarray
    ) -> dict:
        """
        Compute confidence thresholds for each class.

        This is a more sophisticated approach that computes class-specific thresholds
        based on the distribution of predicted probabilities.
        """
        thresholds = {}
        unique_classes = np.unique(true_labels)

        for class_idx in unique_classes:
            # Get predictions for this class
            class_mask = true_labels == class_idx
            class_probs = pred_probs[class_mask, int(class_idx)]

            if len(class_probs) > 0:
                # Use median as threshold for this class
                thresholds[class_idx] = np.median(class_probs)
            else:
                thresholds[class_idx] = self.confidence_threshold

        return thresholds
