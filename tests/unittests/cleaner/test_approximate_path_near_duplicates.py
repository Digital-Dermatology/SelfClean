"""Permanent regression tests for the approximate near-duplicate ranking.

Pins the contract that approximate near-dup output is shape-compatible with
the exact path: `(scores, indices)` under the standard `near_duplicates` key,
scores 1D ascending, indices `(M, 2)` int32 with `i < j`. Also pins that
planted duplicates and known top pairs are recovered.
"""

import unittest

import numpy as np

from selfclean.cleaner.issue_manager import IssueTypes
from selfclean.cleaner.selfclean_cleaner import SelfCleanCleaner

_SEED = 42
_N = 60
_D = 198
_K = 20


def _seeded_emb(n: int = _N, d: int = _D, seed: int = _SEED) -> np.ndarray:
    return np.random.RandomState(seed).rand(n, d).astype(np.float64)


def _fit_approx(emb, K=_K, **kwargs):
    cleaner = SelfCleanCleaner(
        memmap=False, approximate_nn=True, approx_no_neighbors=K, **kwargs
    )
    cleaner.fit(emb_space=emb)
    return cleaner


class TestApproxNearDuplicateSchema(unittest.TestCase):
    """The approximate path must emit the standard `near_duplicates` key with
    the same `(scores, indices)` shape contract as the exact path."""

    def setUp(self):
        self.cleaner = _fit_approx(_seeded_emb())
        self.out = self.cleaner.predict(issues_to_detect=[IssueTypes.NEAR_DUPLICATES])

    def test_emits_near_duplicates_key_not_approx_key(self):
        nd = self.out.get_issues("near_duplicates")
        self.assertIsNotNone(nd)
        self.assertIn("scores", nd)
        self.assertIn("indices", nd)
        # The previous divergent key is gone.
        self.assertIsNone(self.out.get_issues("approx_near_duplicates"))

    def test_shapes_and_dtypes(self):
        nd = self.out.get_issues("near_duplicates")
        scores = np.asarray(nd["scores"])
        indices = np.asarray(nd["indices"])
        self.assertEqual(scores.ndim, 1)
        self.assertEqual(indices.ndim, 2)
        self.assertEqual(indices.shape[1], 2)
        self.assertEqual(scores.shape[0], indices.shape[0])
        self.assertEqual(indices.dtype, np.int32)

    def test_scores_sorted_ascending_and_in_unit_interval(self):
        nd = self.out.get_issues("near_duplicates")
        scores = np.asarray(nd["scores"])
        self.assertTrue(np.all(np.diff(scores) >= -1e-6))
        self.assertGreaterEqual(scores.min(), 0.0)
        self.assertLessEqual(scores.max(), 1.0)

    def test_indices_canonical_and_distinct(self):
        nd = self.out.get_issues("near_duplicates")
        idx = np.asarray(nd["indices"])
        # i < j on every row
        self.assertTrue((idx[:, 0] < idx[:, 1]).all())
        # no duplicate (i, j) pairs
        unique_pairs = np.unique(idx, axis=0)
        self.assertEqual(unique_pairs.shape[0], idx.shape[0])
        # all in range
        self.assertTrue((idx >= 0).all())
        self.assertTrue((idx < _N).all())

    def test_pair_count_bounded_by_n_times_k(self):
        nd = self.out.get_issues("near_duplicates")
        idx = np.asarray(nd["indices"])
        self.assertLessEqual(idx.shape[0], _N * _K)

    def test_dataframe_view_uses_indices_1_indices_2(self):
        """IssueManager.get_issues(return_as_df=True) splits an (M, 2) indices
        array into `indices_1` and `indices_2` columns. Verify the consumer
        contract sc4b's CSV exporter relies on."""
        df = self.out.get_issues("near_duplicates", return_as_df=True)
        self.assertIn("indices_1", df.columns)
        self.assertIn("indices_2", df.columns)
        self.assertIn("scores", df.columns)


class TestApproxNearDuplicatePlanted(unittest.TestCase):
    def test_planted_exact_duplicate_is_top_pair(self):
        rng = np.random.RandomState(_SEED)
        emb = rng.rand(_N, _D)
        emb[-1] = emb[0]

        cleaner = _fit_approx(emb)
        nd = cleaner.predict(
            issues_to_detect=[IssueTypes.NEAR_DUPLICATES]
        ).get_issues("near_duplicates")

        first_pair = sorted(int(x) for x in np.asarray(nd["indices"])[0])
        self.assertEqual(first_pair, [0, _N - 1])
        self.assertAlmostEqual(float(np.asarray(nd["scores"])[0]), 0.0, places=4)

    def test_two_planted_duplicate_pairs_are_top_two(self):
        rng = np.random.RandomState(_SEED)
        emb = rng.rand(_N, _D)
        emb[10] = emb[3]
        emb[40] = emb[20]

        cleaner = _fit_approx(emb)
        nd = cleaner.predict(
            issues_to_detect=[IssueTypes.NEAR_DUPLICATES]
        ).get_issues("near_duplicates")

        idx = np.asarray(nd["indices"])
        scores = np.asarray(nd["scores"])
        top_two_pairs = {tuple(sorted(int(x) for x in p)) for p in idx[:2]}
        self.assertEqual(top_two_pairs, {(3, 10), (20, 40)})
        self.assertLess(float(scores[1]), 1e-4)


class TestApproxVsExactNearDuplicateAgreement(unittest.TestCase):
    """For tractable N where Annoy can return all neighbours (`approx_no_neighbors
    >= N - 1`), the approximate path's top-K pairs must match the exact path's
    top-K. Stronger than a smoke test: a real parity check on a clustered
    toy dataset."""

    def test_top_k_pairs_match_exact_when_k_is_full(self):
        rng = np.random.RandomState(_SEED)
        emb = rng.rand(_N, _D)
        # plant two near-duplicate pairs to make the top of the ranking unambiguous
        emb[-1] = emb[0] + 1e-8 * rng.rand(_D)
        emb[-2] = emb[1] + 1e-8 * rng.rand(_D)

        exact = SelfCleanCleaner(memmap=False, approximate_nn=False)
        exact.fit(emb_space=emb)
        approx = _fit_approx(emb, K=_N - 1)

        nd_e = exact.predict(
            issues_to_detect=[IssueTypes.NEAR_DUPLICATES]
        ).get_issues("near_duplicates")
        nd_a = approx.predict(
            issues_to_detect=[IssueTypes.NEAR_DUPLICATES]
        ).get_issues("near_duplicates")

        top_k = 5
        exact_pairs = {
            tuple(sorted(int(x) for x in p))
            for p in np.asarray(nd_e["indices"])[:top_k]
        }
        approx_pairs = {
            tuple(sorted(int(x) for x in p))
            for p in np.asarray(nd_a["indices"])[:top_k]
        }
        self.assertEqual(exact_pairs, approx_pairs)


class TestApproxRequiresFit(unittest.TestCase):
    def test_calling_approx_method_without_fit_raises(self):
        cleaner = SelfCleanCleaner(
            memmap=False, approximate_nn=True, approx_no_neighbors=_K
        )
        # not fit yet
        with self.assertRaises(RuntimeError):
            cleaner._get_approx_near_duplicate_ranking()


if __name__ == "__main__":
    unittest.main()
