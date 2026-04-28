import math
from typing import Tuple

import numpy as np
from tqdm.auto import tqdm

from ...cleaner.near_duplicates.base_near_duplicate_mixin import BaseNearDuplicateMixin
from ...core.src.utils.plotting import plot_dist
from ...utils.utils import condensed_to_square


class EmbeddingDistanceMixin(BaseNearDuplicateMixin):
    def __init__(
        self,
        approx_no_neighbors: int = 100,
        tree_size: int = 100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.approx_no_neighbors = approx_no_neighbors
        self.tree_size = tree_size

    def get_near_duplicate_ranking(self) -> Tuple[np.ndarray, np.ndarray]:
        """Dispatches to the exact or approximate ranking based on
        `self.approximate_nn`. Both branches return the same `(scores, indices)`
        shape: 1D scores ascending + (M, 2) int32 pair indices."""
        if getattr(self, "approximate_nn", False):
            return self._get_approx_near_duplicate_ranking()
        return self._get_exact_near_duplicate_ranking()

    def _get_exact_near_duplicate_ranking(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.memmap:
            score_file = self.memmap_path / "near_duplicate_scores.dat"
            # make sure the files do not exist already
            if score_file.exists():
                score_file.unlink()
            scores_near_dup = np.memmap(
                str(score_file),
                dtype=self.precision_type_distance,
                mode="w+",
                shape=(self.condensed_size,),
            )
        else:
            scores_near_dup = np.zeros(
                shape=(self.condensed_size,),
                dtype=self.precision_type_distance,
            )

        # sort the values in the condensed matrix
        sorting = self.p_distances.argsort()
        scores_near_dup[:] = np.take(self.p_distances, indices=sorting, axis=0)
        if self.memmap:
            scores_near_dup.flush()
        # vectorize the mapping function
        vec_index_mapping = np.vectorize(condensed_to_square)
        # here the chunk size is x**2 since we have quadratically more
        chunk_size = self.chunk_size**2
        # chunk the sorted values for memory efficiency
        n_chunks = math.ceil(self.condensed_size / chunk_size)
        if self.memmap:
            indices_file = self.memmap_path / "near_duplicate_indices.dat"
            # make sure the files do not exist already
            if indices_file.exists():
                indices_file.unlink()
            indices_near_dup = np.memmap(
                str(indices_file),
                dtype=np.int32,
                mode="w+",
                shape=(self.condensed_size, 2),
            )
        else:
            indices_near_dup = np.zeros(
                shape=(self.condensed_size, 2),
                dtype=np.int32,
            )
        # this creates the corresponding indices of the sorted array
        for i in tqdm(
            range(n_chunks),
            desc="Processing possible near duplicates",
            total=n_chunks,
            position=0,
            leave=True,
        ):
            chunk_slice = slice(i * chunk_size, (i + 1) * chunk_size, 1)
            chunk_sorting = sorting[chunk_slice]
            # map the indices from the condensed to the redundant distance matrix
            mapping_row = np.asarray(vec_index_mapping(chunk_sorting, self.N)).T
            indices_near_dup[chunk_slice, :] = mapping_row
            if self.memmap:
                indices_near_dup.flush()
            del mapping_row

        if self.plot_distribution:
            plot_dist(
                scores=scores_near_dup,
                title="Distribution of near-duplicates",
            )
        return scores_near_dup, indices_near_dup

    def _get_approx_near_duplicate_ranking(self) -> Tuple[np.ndarray, np.ndarray]:
        """KNN-based near-duplicate ranking that consumes the cached graph
        populated by `SelfCleanCleaner._build_knn_index`.

        Returns the same `(scores, indices)` shape the exact path returns —
        only the count differs. Where the exact path produces every pair
        (`condensed_size`), the approximate path produces at most `N * K`
        unique pairs derived from the K-nearest-neighbour graph, with `K =
        self.approx_no_neighbors`. For each (i, j) emitted, `i < j`. Scores
        are in [0, 1] and sorted ascending (most-duplicate first).
        """
        if not hasattr(self, "knn_indices"):
            raise RuntimeError(
                "Approximate near-duplicate ranking requires `_build_knn_index` "
                "to have been called by `fit`. Pass `approximate_nn=True` to "
                "the cleaner constructor and re-fit."
            )

        K = self.knn_indices.shape[1]
        rows = np.repeat(np.arange(self.N, dtype=np.int64), K)
        cols = self.knn_indices.reshape(-1).astype(np.int64)
        dists = self.knn_distances.reshape(-1).astype(self.precision_type_distance)

        # Order each pair canonically as (a < b)
        a = np.minimum(rows, cols)
        b = np.maximum(rows, cols)
        valid = a != b
        a, b, dists = a[valid], b[valid], dists[valid]

        # Pack (a, b) -> single key for de-duplication
        key = a * self.N + b
        # Sort by key first, then by distance ascending within each key,
        # so the first occurrence per key keeps the smallest distance
        # (mutual neighbours can yield slightly different angular distances).
        order = np.lexsort((dists, key))
        key_sorted = key[order]
        a_sorted = a[order]
        b_sorted = b[order]
        dists_sorted = dists[order]
        keep = np.concatenate([[True], key_sorted[1:] != key_sorted[:-1]])
        a_u = a_sorted[keep]
        b_u = b_sorted[keep]
        d_u = dists_sorted[keep]

        # Final sort by distance ascending
        final_order = np.argsort(d_u, kind="stable")
        indices_near_dup = np.column_stack(
            [a_u[final_order], b_u[final_order]]
        ).astype(np.int32)
        scores_near_dup = d_u[final_order].astype(self.precision_type_distance)

        if self.plot_distribution:
            plot_dist(
                scores=scores_near_dup,
                title="Distribution of near-duplicates (approximate)",
            )
        return scores_near_dup, indices_near_dup

    # Backwards-compatible alias for downstream callers that imported the
    # public method by name. Returns the same `(scores, indices)` tuple as
    # `get_near_duplicate_ranking()` when `approximate_nn=True`.
    def get_approx_near_duplicate_ranking(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._get_approx_near_duplicate_ranking()
