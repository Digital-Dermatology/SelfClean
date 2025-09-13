from typing import Tuple

import numpy as np

from ...cleaner.near_duplicates.base_near_duplicate_mixin import BaseNearDuplicateMixin
from ...core.src.utils.plotting import plot_dist


class CleanLabNearDuplicateMixin(BaseNearDuplicateMixin):
    """
    Near duplicate ranking based on pairwise embedding distances.
    Returns a continuous distance score for all pairs (no thresholding).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_near_duplicate_ranking(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return full pairwise ranking based on distances in embedding space.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (scores, indices) where lower scores
            indicate more similar pairs. Indices are pairs [i, j].
        """
        # Prefer precomputed condensed distances if available
        if hasattr(self, "p_distances") and self.p_distances is not None:
            sorted_idx = np.argsort(self.p_distances)
            sorted_scores = self.p_distances[sorted_idx]

            from ...utils.utils import condensed_to_square

            top_indices = np.array([condensed_to_square(i, self.N) for i in sorted_idx])

            if self.plot_distribution:
                plot_dist(scores=sorted_scores, title="Distribution of near-duplicates")

            return sorted_scores, top_indices
        else:
            # Compute from embeddings
            from sklearn.metrics.pairwise import cosine_similarity

            sim = cosine_similarity(self.emb_space)
            dist = 1 - sim
            triu = np.triu_indices(self.N, k=1)
            scores = dist[triu]
            order = np.argsort(scores)
            sorted_scores = scores[order]
            sorted_indices = np.column_stack([triu[0][order], triu[1][order]])

            if self.plot_distribution:
                plot_dist(scores=sorted_scores, title="Distribution of near-duplicates")

            return sorted_scores, sorted_indices

    def _fallback_distance_based(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fallback to distance-based near duplicate detection when labels are not available.
        Similar to the existing EmbeddingDistanceMixin but simplified.
        """
        # Get pairwise distances and find the most similar pairs
        # Use condensed distance matrix if available
        if hasattr(self, "p_distances") and self.p_distances is not None:
            # Sort distances to find most similar pairs
            sorted_indices = np.argsort(self.p_distances)
            sorted_scores = self.p_distances[sorted_indices]

            # Convert condensed indices to square matrix indices
            from ...utils.utils import condensed_to_square

            vec_index_mapping = np.vectorize(condensed_to_square)

            # Take top 1000 most similar pairs to avoid memory issues
            top_n = min(1000, len(sorted_indices))
            top_sorted_indices = sorted_indices[:top_n]
            top_scores = sorted_scores[:top_n]

            # Convert to square matrix indices
            square_indices = np.array(
                [condensed_to_square(idx, self.N) for idx in top_sorted_indices]
            )

            return top_scores, square_indices
        else:
            # Calculate distances from scratch if needed
            from sklearn.metrics.pairwise import cosine_similarity

            similarity_matrix = cosine_similarity(self.emb_space)
            # Convert to distance (1 - similarity)
            distance_matrix = 1 - similarity_matrix

            # Get upper triangular indices (avoid duplicates)
            triu_indices = np.triu_indices(self.N, k=1)
            distances = distance_matrix[triu_indices]

            # Sort and get top pairs
            sorted_idx = np.argsort(distances)
            top_n = min(1000, len(sorted_idx))

            top_scores = distances[sorted_idx[:top_n]]
            top_indices = np.column_stack(
                [
                    triu_indices[0][sorted_idx[:top_n]],
                    triu_indices[1][sorted_idx[:top_n]],
                ]
            )

            return top_scores, top_indices
