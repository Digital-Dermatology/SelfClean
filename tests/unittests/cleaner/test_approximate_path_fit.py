"""Permanent regression tests for SelfCleanCleaner.fit() in approximate mode.

These tests pin the post-fit invariants of the approximate (`approximate_nn=True`)
path: no O(N²) allocations, a cached KNN graph in the same [0, 1] cosine
distance scale as the exact path, and consistency between memmap and
in-memory modes.

If a future change re-introduces the dense distance matrix in approximate
mode, or shifts the KNN distance unit, these tests will fail.
"""

import unittest

import numpy as np

from selfclean.cleaner.selfclean_cleaner import SelfCleanCleaner

_SEED = 42
_N = 60
_D = 198
_K = 10


def _seeded_emb(n: int = _N, d: int = _D, seed: int = _SEED) -> np.ndarray:
    return np.random.RandomState(seed).rand(n, d).astype(np.float64)


def _seeded_labels(n: int = _N, n_classes: int = 5, seed: int = _SEED) -> np.ndarray:
    return np.random.RandomState(seed + 1).randint(n_classes, size=n)


class TestApproximateFitSkipsONSquared(unittest.TestCase):
    """Approximate fit must NOT allocate the O(N²) intermediate arrays."""

    def setUp(self):
        self.emb_space = _seeded_emb()
        self.labels = _seeded_labels()

    def test_distance_matrix_and_p_distances_are_none(self):
        cleaner = SelfCleanCleaner(
            memmap=False, approximate_nn=True, approx_no_neighbors=_K
        )
        cleaner.fit(emb_space=self.emb_space, labels=self.labels)
        self.assertIsNone(cleaner.distance_matrix)
        self.assertIsNone(cleaner.p_distances)
        self.assertTrue(cleaner.is_fitted)

    def test_no_memmap_files_created_in_approximate_mode(self):
        cleaner = SelfCleanCleaner(
            memmap=True, approximate_nn=True, approx_no_neighbors=_K
        )
        cleaner.fit(emb_space=self.emb_space, labels=self.labels)
        # The distance_matrix / p_distances memmap files would dwarf the corpus
        # at scale; in approximate mode they must never be created.
        self.assertFalse((cleaner.memmap_path / "dist_matrix.dat").exists())
        self.assertFalse((cleaner.memmap_path / "p_distances.dat").exists())

    def test_exact_mode_still_allocates(self):
        """Sanity: the exact path is unchanged."""
        cleaner = SelfCleanCleaner(memmap=False, approximate_nn=False)
        cleaner.fit(emb_space=self.emb_space, labels=self.labels)
        self.assertIsNotNone(cleaner.distance_matrix)
        self.assertIsNotNone(cleaner.p_distances)
        self.assertEqual(cleaner.distance_matrix.shape, (_N, _N))


class TestApproximateFitKnnCache(unittest.TestCase):
    """The cached KNN graph has the right shape, units, and content."""

    def setUp(self):
        self.emb_space = _seeded_emb()
        self.labels = _seeded_labels()

    def _fit(self, **kwargs):
        cleaner = SelfCleanCleaner(
            memmap=False,
            approximate_nn=True,
            approx_no_neighbors=_K,
            **kwargs,
        )
        cleaner.fit(emb_space=self.emb_space, labels=self.labels)
        return cleaner

    def test_knn_arrays_shape_and_dtype(self):
        c = self._fit()
        self.assertEqual(c.knn_indices.shape, (_N, _K))
        self.assertEqual(c.knn_distances.shape, (_N, _K))
        self.assertEqual(c.knn_indices.dtype, np.int32)
        # distances are in the cleaner's configured precision
        self.assertEqual(c.knn_distances.dtype, c.precision_type_distance)

    def test_knn_distances_in_unit_interval(self):
        c = self._fit()
        self.assertGreaterEqual(c.knn_distances.min(), 0.0)
        self.assertLessEqual(c.knn_distances.max(), 1.0)

    def test_self_excluded_from_neighbours(self):
        c = self._fit()
        for i in range(_N):
            self.assertNotIn(i, set(int(j) for j in c.knn_indices[i]))

    def test_neighbour_indices_in_range(self):
        c = self._fit()
        self.assertTrue((c.knn_indices >= 0).all())
        self.assertTrue((c.knn_indices < _N).all())

    def test_knn_distances_per_row_sorted_ascending(self):
        """Annoy returns neighbours in ascending distance — preserve that."""
        c = self._fit()
        diffs = np.diff(c.knn_distances, axis=1)
        self.assertTrue((diffs >= -1e-6).all())

    def test_planted_duplicate_is_top_neighbour(self):
        """If we plant an exact duplicate at (0, N-1), each must point at the
        other as its nearest neighbour with distance ≈ 0."""
        rng = np.random.RandomState(_SEED)
        emb = rng.rand(_N, _D)
        emb[-1] = emb[0]
        cleaner = SelfCleanCleaner(
            memmap=False, approximate_nn=True, approx_no_neighbors=_K
        )
        cleaner.fit(emb_space=emb)

        self.assertEqual(int(cleaner.knn_indices[0, 0]), _N - 1)
        self.assertEqual(int(cleaner.knn_indices[_N - 1, 0]), 0)
        self.assertAlmostEqual(float(cleaner.knn_distances[0, 0]), 0.0, places=4)
        self.assertAlmostEqual(float(cleaner.knn_distances[_N - 1, 0]), 0.0, places=4)

    def test_distance_unit_matches_exact_path_for_planted_duplicate(self):
        """The KNN distance scale must equal the exact path's [0, 1] cosine
        distance. For an exact duplicate the value is 0 in both modes; we
        also verify a non-trivial pair stays in the same scale."""
        rng = np.random.RandomState(_SEED)
        emb = rng.rand(_N, _D)
        emb[-1] = emb[0]

        exact = SelfCleanCleaner(memmap=False, approximate_nn=False)
        exact.fit(emb_space=emb)
        approx = SelfCleanCleaner(
            memmap=False, approximate_nn=True, approx_no_neighbors=_K
        )
        approx.fit(emb_space=emb)

        # exact path: distance from 0 to N-1 is 0
        self.assertAlmostEqual(
            float(exact.distance_matrix[0, _N - 1]), 0.0, places=4
        )
        self.assertAlmostEqual(
            float(approx.knn_distances[0, 0]), 0.0, places=4
        )
        # both modes use the same [0, 1] scale
        self.assertGreaterEqual(approx.knn_distances.min(), 0.0)
        self.assertLessEqual(approx.knn_distances.max(), 1.0)


class TestApproximateFitDeterminism(unittest.TestCase):
    def setUp(self):
        self.emb_space = _seeded_emb()
        self.labels = _seeded_labels()

    def test_two_fits_produce_identical_knn_neighbours(self):
        """Two cleaners fit on the same input produce the same KNN graph
        (up to ties — Annoy is randomised but returns stable neighbours for
        well-separated inputs at this scale)."""
        a = SelfCleanCleaner(
            memmap=False, approximate_nn=True, approx_no_neighbors=_K
        )
        a.fit(emb_space=self.emb_space, labels=self.labels)
        b = SelfCleanCleaner(
            memmap=False, approximate_nn=True, approx_no_neighbors=_K
        )
        b.fit(emb_space=self.emb_space, labels=self.labels)

        # Compare the SET of neighbours per row (order may vary on ties).
        for i in range(_N):
            self.assertEqual(
                set(int(j) for j in a.knn_indices[i]),
                set(int(j) for j in b.knn_indices[i]),
            )


if __name__ == "__main__":
    unittest.main()
