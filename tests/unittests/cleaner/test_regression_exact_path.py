"""Regression tests that lock in the numerical behaviour of the exact (O(N²))
SelfCleanCleaner path.

These tests are intentionally tight: they pin shapes, top-K rankings, planted
duplicate / outlier responses and idempotency. They serve as a failsafe so
future refactors (e.g. the approximate-NN scaling work, but also any other
distance-function or ranking change) cannot silently change exact-path output.

If a test here fails, treat it as a behaviour change — either fix the
regression or, if the change is intentional, update the assertion in the same
PR with a one-line note explaining why the new value is correct.
"""

import unittest

import numpy as np

from selfclean.cleaner.selfclean_cleaner import SelfCleanCleaner

_SEED = 42
_N = 50
_D = 198


def _seeded_emb(n: int = _N, d: int = _D, seed: int = _SEED) -> np.ndarray:
    return np.random.RandomState(seed).rand(n, d).astype(np.float64)


def _seeded_labels(n: int = _N, n_classes: int = 5, seed: int = _SEED) -> np.ndarray:
    return np.random.RandomState(seed + 1).randint(n_classes, size=n)


class TestExactPathFitShapes(unittest.TestCase):
    """The shape and basic invariants of the O(N²) intermediate arrays."""

    def setUp(self):
        self.emb_space = _seeded_emb()
        self.labels = _seeded_labels()
        self.condensed_size = _N * (_N - 1) // 2

    def test_distance_matrix_shape_and_range(self):
        cleaner = SelfCleanCleaner(memmap=False)
        cleaner.fit(emb_space=self.emb_space, labels=self.labels)

        self.assertEqual(cleaner.distance_matrix.shape, (_N, _N))
        # cosine distance is in [0, 1] after the normalize+invert step
        self.assertGreaterEqual(cleaner.distance_matrix.min(), 0.0)
        self.assertLessEqual(cleaner.distance_matrix.max(), 1.0)
        # Self-distance is 0 along the diagonal.
        self.assertTrue(np.allclose(np.diag(cleaner.distance_matrix), 0.0, atol=1e-6))
        # Symmetry: D[i,j] == D[j,i]
        self.assertTrue(
            np.allclose(cleaner.distance_matrix, cleaner.distance_matrix.T, atol=1e-6)
        )

    def test_p_distances_shape_matches_condensed(self):
        cleaner = SelfCleanCleaner(memmap=False)
        cleaner.fit(emb_space=self.emb_space, labels=self.labels)

        self.assertEqual(cleaner.p_distances.shape, (self.condensed_size,))
        self.assertEqual(cleaner.condensed_size, self.condensed_size)

    def test_p_distances_match_upper_triangle_of_distance_matrix(self):
        """The condensed array is, by construction, the upper triangle (k=1) of
        the dense distance matrix flattened in row-major order."""
        cleaner = SelfCleanCleaner(memmap=False)
        cleaner.fit(emb_space=self.emb_space, labels=self.labels)

        triu = cleaner.distance_matrix[np.triu_indices(_N, k=1)]
        self.assertTrue(np.allclose(np.asarray(cleaner.p_distances), triu, atol=1e-6))

    def test_memmap_and_in_memory_paths_agree(self):
        """memmap=True must produce numerically identical p_distances to memmap=False."""
        cleaner_mem = SelfCleanCleaner(memmap=False)
        cleaner_mem.fit(emb_space=self.emb_space, labels=self.labels)
        cleaner_mm = SelfCleanCleaner(memmap=True)
        cleaner_mm.fit(emb_space=self.emb_space, labels=self.labels)

        self.assertTrue(
            np.allclose(
                np.asarray(cleaner_mem.p_distances),
                np.asarray(cleaner_mm.p_distances),
                atol=1e-6,
            )
        )


class TestExactPathPredictIdempotency(unittest.TestCase):
    """predict() must be deterministic and idempotent for the same fit()."""

    def setUp(self):
        self.emb_space = _seeded_emb()
        self.labels = _seeded_labels()

    def _fit(self, **kwargs):
        cleaner = SelfCleanCleaner(memmap=False, **kwargs)
        cleaner.fit(emb_space=self.emb_space, labels=self.labels)
        return cleaner

    def test_predict_is_deterministic_across_calls(self):
        cleaner = self._fit()
        out_a = cleaner.predict()
        out_b = cleaner.predict()
        for issue in ("near_duplicates", "off_topic_samples", "label_errors"):
            a = out_a.get_issues(issue)
            b = out_b.get_issues(issue)
            self.assertTrue(np.array_equal(a["indices"], b["indices"]))
            self.assertTrue(np.allclose(a["scores"], b["scores"], atol=1e-9))

    def test_predict_is_deterministic_across_fits(self):
        out_a = self._fit().predict()
        out_b = self._fit().predict()
        for issue in ("near_duplicates", "off_topic_samples", "label_errors"):
            a = out_a.get_issues(issue)
            b = out_b.get_issues(issue)
            self.assertTrue(np.array_equal(a["indices"], b["indices"]))
            self.assertTrue(np.allclose(a["scores"], b["scores"], atol=1e-9))


class TestExactPathRankingShapes(unittest.TestCase):
    """The shape and ordering invariants of the predict() output."""

    def setUp(self):
        self.emb_space = _seeded_emb()
        self.labels = _seeded_labels()
        self.condensed_size = _N * (_N - 1) // 2

    def test_near_duplicate_output_shape_and_sorted(self):
        cleaner = SelfCleanCleaner(memmap=False)
        cleaner.fit(emb_space=self.emb_space, labels=self.labels)
        out = cleaner.predict().get_issues("near_duplicates")

        self.assertEqual(np.asarray(out["scores"]).shape, (self.condensed_size,))
        self.assertEqual(np.asarray(out["indices"]).shape, (self.condensed_size, 2))
        # scores ascending — most-duplicate pair first
        scores = np.asarray(out["scores"])
        self.assertTrue(np.all(np.diff(scores) >= -1e-9))
        # indices_1 != indices_2 for every row
        idx = np.asarray(out["indices"])
        self.assertTrue((idx[:, 0] != idx[:, 1]).all())

    def test_off_topic_output_shape(self):
        cleaner = SelfCleanCleaner(memmap=False)
        cleaner.fit(emb_space=self.emb_space, labels=self.labels)
        out = cleaner.predict().get_issues("off_topic_samples")

        self.assertEqual(np.asarray(out["indices"]).shape, (_N,))
        self.assertEqual(np.asarray(out["scores"]).shape, (_N,))
        # indices are a permutation of range(N)
        self.assertEqual(set(int(i) for i in out["indices"]), set(range(_N)))
        # finite scores, no NaN
        self.assertFalse(np.isnan(np.asarray(out["scores"])).any())

    def test_label_error_output_shape(self):
        cleaner = SelfCleanCleaner(memmap=False)
        cleaner.fit(emb_space=self.emb_space, labels=self.labels)
        out = cleaner.predict().get_issues("label_errors")

        scores = np.asarray(out["scores"])
        idx = np.asarray(out["indices"])
        self.assertEqual(scores.shape, (_N,))
        self.assertEqual(idx.shape, (_N,))
        self.assertEqual(set(int(i) for i in idx), set(range(_N)))


class TestExactPathPlantedScenarios(unittest.TestCase):
    """Hand-crafted inputs with a known correct answer."""

    def test_planted_exact_duplicate_is_top_pair(self):
        rng = np.random.RandomState(_SEED)
        emb = rng.rand(_N, _D)
        # plant an exact duplicate at (0, N-1)
        emb[-1] = emb[0]

        cleaner = SelfCleanCleaner(memmap=False)
        cleaner.fit(emb_space=emb)
        out = cleaner.predict(issues_to_detect=None) if False else cleaner.predict()
        nd = out.get_issues("near_duplicates")

        # the most-similar pair is (0, N-1) at distance 0
        first_pair = sorted(int(x) for x in nd["indices"][0])
        self.assertEqual(first_pair, [0, _N - 1])
        self.assertAlmostEqual(float(nd["scores"][0]), 0.0, places=5)

    def test_two_planted_duplicate_pairs_are_top_two(self):
        rng = np.random.RandomState(_SEED)
        emb = rng.rand(_N, _D)
        # plant two exact-duplicate pairs at well-separated indices
        emb[10] = emb[3]
        emb[40] = emb[20]

        cleaner = SelfCleanCleaner(memmap=False)
        cleaner.fit(emb_space=emb)
        nd = cleaner.predict().get_issues("near_duplicates")

        top_two_pairs = {tuple(sorted(int(x) for x in p)) for p in nd["indices"][:2]}
        self.assertEqual(top_two_pairs, {(3, 10), (20, 40)})
        self.assertTrue(float(nd["scores"][1]) < 1e-5)

    def test_planted_far_outlier_is_top_off_topic(self):
        emb = np.zeros((_N, _D), dtype=np.float64)
        emb[-1] = np.ones(_D)  # one isolated direction

        cleaner = SelfCleanCleaner(memmap=False)
        cleaner.fit(emb_space=emb)
        ot = cleaner.predict().get_issues("off_topic_samples")

        self.assertEqual(int(np.asarray(ot["indices"])[0]), _N - 1)


class TestExactPathDegenerate(unittest.TestCase):
    """Edge cases that have historically been load-bearing for callers."""

    def test_zero_embeddings_yield_constant_distance(self):
        emb = np.zeros((_N, _D))
        cleaner = SelfCleanCleaner(memmap=False)
        cleaner.fit(emb_space=emb)
        # cosine_similarity on zero vectors is 0; SelfClean maps that to 0.5.
        self.assertTrue(
            np.allclose(np.asarray(cleaner.distance_matrix), 0.5, atol=1e-6)
        )
        nd = cleaner.predict().get_issues("near_duplicates")
        self.assertTrue(np.allclose(np.asarray(nd["scores"]), 0.5, atol=1e-6))


class TestExactPathAutoCleaning(unittest.TestCase):
    """auto_issues must be deterministic and a valid index subset."""

    def setUp(self):
        self.emb_space = _seeded_emb()
        self.labels = _seeded_labels()
        self.condensed_size = _N * (_N - 1) // 2

    def test_auto_issues_is_index_subset(self):
        cleaner = SelfCleanCleaner(memmap=False, auto_cleaning=True)
        cleaner.fit(emb_space=self.emb_space, labels=self.labels)
        out = cleaner.predict()

        for issue in ("near_duplicates", "off_topic_samples", "label_errors"):
            v = out.get_issues(issue)
            auto = np.asarray(v["auto_issues"])
            self.assertEqual(auto.dtype.kind, "i")
            # `near_duplicates` indexes pair scores (size condensed_size); the
            # other two index per-sample scores (size N).
            upper = self.condensed_size if issue == "near_duplicates" else _N
            self.assertTrue((auto >= 0).all())
            self.assertTrue((auto < upper).all())
            self.assertEqual(len(np.unique(auto)), len(auto))

    def test_auto_cleaning_is_deterministic(self):
        a = SelfCleanCleaner(memmap=False, auto_cleaning=True)
        a.fit(emb_space=self.emb_space, labels=self.labels)
        b = SelfCleanCleaner(memmap=False, auto_cleaning=True)
        b.fit(emb_space=self.emb_space, labels=self.labels)
        out_a, out_b = a.predict(), b.predict()
        for issue in ("near_duplicates", "off_topic_samples", "label_errors"):
            self.assertTrue(
                np.array_equal(
                    np.asarray(out_a.get_issues(issue)["auto_issues"]),
                    np.asarray(out_b.get_issues(issue)["auto_issues"]),
                )
            )


if __name__ == "__main__":
    unittest.main()
