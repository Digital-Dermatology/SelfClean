import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from cleanlab import Datalab
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from ...cleaner.label_errors.base_label_error_mixin import BaseLabelErrorMixin
from ...core.src.utils.plotting import plot_dist


class CleanLabLabelErrorMixin(BaseLabelErrorMixin):
    """
    Label error detection using CleanLab on pretrained representations.

    Based on the baseline implementation from external_code/mse-cleaning-audio.
    Uses CleanLab's supervised label error detection with cross-validation.
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
        Initialize CleanLab label error mixin.

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

    def get_label_error_ranking(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect label errors using CleanLab's supervised approach.

        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: (scores, indices) where
            lower scores indicate more likely label errors. Returns (None, None) if
            labels are not available or insufficient for detection.
        """
        # Check if labels are available
        if self.labels is None:
            warnings.warn("Can't find label errors without having access to labels.")
            return None, None

        # Check if we have more than one unique label
        unique_labels = np.unique(self.labels)
        if len(unique_labels) == 1:
            warnings.warn("Can't detect label errors with only one label.")
            return None, None

        # Check if we have enough samples for cross-validation
        if len(self.labels) < self.cv_folds:
            warnings.warn(
                f"Not enough samples ({len(self.labels)}) for {self.cv_folds}-fold cross-validation."
            )
            return None, None

        try:
            # Convert labels to numeric if they're strings
            if isinstance(unique_labels[0], str):
                # Create label mapping
                label_to_num = {label: i for i, label in enumerate(unique_labels)}
                numeric_labels = np.array(
                    [label_to_num[label] for label in self.labels]
                )
            else:
                numeric_labels = self.labels.astype(int)

            # Train LogisticRegression model with cross-validation
            model = LogisticRegression(
                C=self.C,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
            )

            # Get out-of-sample predictions using cross-validation
            pred_probs = cross_val_predict(
                estimator=model,
                X=self.emb_space,
                y=numeric_labels,
                cv=self.cv_folds,
                method="predict_proba",
            )

            # Create data structure for CleanLab
            data_dict = {
                "class": self.labels,  # Use original labels for CleanLab
                "id": list(range(len(self.labels))),
            }
            datalab_data = pd.DataFrame(data_dict)

            # Create Datalab object and find issues
            lab = Datalab(datalab_data, label_name="class")
            lab.find_issues(pred_probs=pred_probs, features=self.emb_space)

            # Get label error issues
            label_issues = lab.get_issues("label")

            if label_issues is not None and len(label_issues) > 0:
                # Process CleanLab results into standardized format
                results = self._prepare_cleanlab_results(label_issues, datalab_data)

                if len(results) > 0:
                    # Extract scores and indices
                    scores = results["scores"].values
                    indices = results["indices"].values

                    if self.plot_distribution:
                        plot_dist(
                            scores=scores,
                            title="Distribution of possible label errors (CleanLab)",
                        )

                    return scores, indices

            # If no label errors found, return empty arrays
            return np.array([]), np.array([])

        except Exception as e:
            warnings.warn(f"CleanLab label error detection failed: {e}")
            return None, None

    def _prepare_cleanlab_results(self, label_issues, datalab_data):
        """
        Process CleanLab results into standardized format.

        Args:
            label_issues: DataFrame from lab.get_issues("label")
            datalab_data: Original data DataFrame with class and id

        Returns:
            DataFrame with columns ['indices', 'class', 'id', 'scores']
        """
        # Concatenate CleanLab results with original data
        results = pd.concat(
            [label_issues.reset_index(drop=True), datalab_data.reset_index(drop=True)],
            axis=1,
        )

        # Sort by label_score (ascending - lower scores = more likely errors)
        results = results.sort_values("label_score").reset_index(drop=False)

        # Clean up columns
        if "is_label_issue" in results.columns:
            results = results.drop(["is_label_issue"], axis=1)

        # Rename columns for consistency
        results = results.rename(columns={"index": "indices", "label_score": "scores"})

        # Return final format
        return results[["indices", "class", "id", "scores"]]
