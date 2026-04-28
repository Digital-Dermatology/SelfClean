"""Permanent regression tests for the approximate-mode off-topic path.

Approximate mode reuses LAD's scoring algorithm — only the source of the
single-linkage dendrogram changes. In the exact path the dendrogram comes
from `scipy.cluster.hierarchy.single(self.p_distances)` over all N(N-1)/2
pairs; in the approximate path it comes from the MST of the cached KNN
graph (Kruskal/union-find = single linkage). For densely-clustered points
the MST edges are all KNN edges, so the dendrograms — and therefore LAD's
output — are identical.

These tests pin: (a) the schema and score direction, (b) planted-outlier
recovery in approximate mode, (c) parity between approximate and exact LAD
on a small-N planted-outlier dataset, (d) auto-cleaning compatibility.
"""

import unittest

import numpy as np

from selfclean.cleaner.issue_manager import IssueTypes
from selfclean.cleaner.selfclean_cleaner import SelfCleanCleaner

_SEED = 42
_N = 60
_D = 198
_K = 30


def _seeded_emb(n: int = _N, d: int = _D, seed: int = _SEED) -> np.ndarray:
    return np.random.RandomState(seed).rand(n, d).astype(np.float64)


def _fit_approx(emb, K=_K, **kwargs):
    cleaner = SelfCleanCleaner(
        memmap=False, approximate_nn=True, approx_no_neighbors=K, **kwargs
    )
    cleaner.fit(emb_space=emb)
    return cleaner


class TestApproxLinkageBuilt(unittest.TestCase):
    """`fit(approximate_nn=True)` must populate the cached linkage matrix
    that LAD will consume."""

    def test_knn_linkage_matrix_shape(self):
        c = _fit_approx(_seeded_emb())
        # Single-linkage dendrogram has N-1 merges, 4 columns each.
        self.assertEqual(c.knn_linkage_matrix.shape, (_N - 1, 4))
        self.assertEqual(c.knn_linkage_matrix.dtype, np.float64)

    def test_knn_linkage_distances_in_unit_interval_and_sorted(self):
        c = _fit_approx(_seeded_emb())
        d = c.knn_linkage_matrix[:, 2]
        self.assertGreaterEqual(d.min(), 0.0)
        self.assertLessEqual(d.max(), 1.0)
        # Single-linkage merges are non-decreasing.
        self.assertTrue(np.all(np.diff(d) >= -1e-9))

    def test_knn_linkage_sample_count_total_is_N(self):
        c = _fit_approx(_seeded_emb())
        # The last merge fuses everything — its `merged_size` column equals N.
        self.assertEqual(int(c.knn_linkage_matrix[-1, 3]), _N)


class TestApproxLADSchema(unittest.TestCase):
    def setUp(self):
        self.cleaner = _fit_approx(_seeded_emb())
        self.out = self.cleaner.predict(
            issues_to_detect=[IssueTypes.OFF_TOPIC_SAMPLES]
        )

    def test_emits_off_topic_samples_key(self):
        v = self.out.get_issues("off_topic_samples")
        self.assertIsNotNone(v)
        self.assertIn("scores", v)
        self.assertIn("indices", v)

    def test_shapes_match_exact_path_contract(self):
        v = self.out.get_issues("off_topic_samples")
        scores = np.asarray(v["scores"])
        indices = np.asarray(v["indices"])
        self.assertEqual(scores.shape, (_N,))
        self.assertEqual(indices.shape, (_N,))
        self.assertEqual(set(int(i) for i in indices), set(range(_N)))
        self.assertFalse(np.isnan(scores).any())


class TestApproxLADPlanted(unittest.TestCase):
    """LAD-on-MST recovers a real outlier when it is genuinely separable in
    cosine space.

    Note: the exact-path `test_far_outlier` uses an all-zero fixture where
    every cosine distance is identical (0.5), so the exact LAD's tie-breaking
    happens to put the only non-zero vector at position 0. That ordering is
    a property of `scipy.cluster.hierarchy.single` on equal-weight edges, not
    of the LAD algorithm itself, and does not transfer across MST builders
    (scipy's `minimum_spanning_tree` resolves ties differently). Use a
    non-degenerate fixture here so the test pins the algorithmic behaviour
    rather than a tie-breaking artefact."""

    def test_planted_outlier_with_real_distance_gap(self):
        rng = np.random.RandomState(_SEED)
        # tight in-distribution cluster
        cluster = rng.randn(_N - 1, _D) * 0.05 + 0.5
        # clear outlier in a different direction
        outlier = np.full((1, _D), -2.0)
        emb = np.concatenate([cluster, outlier], axis=0)

        cleaner = _fit_approx(emb)
        v = cleaner.predict(
            issues_to_detect=[IssueTypes.OFF_TOPIC_SAMPLES]
        ).get_issues("off_topic_samples")
        self.assertEqual(int(np.asarray(v["indices"])[0]), _N - 1)


class TestApproxVsExactLADAgreement(unittest.TestCase):
    """Algorithmic parity: same algorithm (LAD), same single-linkage
    dendrogram structure → same off-topic ranking on tractable N.

    For a clustered dataset with one planted outlier, both modes must agree
    that the outlier is at position 0. With K large enough that the KNN
    graph spans the relevant cluster topology, the top-K rankings agree."""

    def _planted_outlier_dataset(self, seed=_SEED):
        rng = np.random.RandomState(seed)
        cluster = rng.randn(_N - 1, _D) * 0.05 + 0.5
        outlier = np.full((1, _D), -2.0)
        return np.concatenate([cluster, outlier], axis=0)

    def test_top_outlier_matches_exact(self):
        emb = self._planted_outlier_dataset()
        exact = SelfCleanCleaner(memmap=False, approximate_nn=False)
        exact.fit(emb_space=emb)
        approx = _fit_approx(emb, K=_N - 1)

        v_e = exact.predict(
            issues_to_detect=[IssueTypes.OFF_TOPIC_SAMPLES]
        ).get_issues("off_topic_samples")
        v_a = approx.predict(
            issues_to_detect=[IssueTypes.OFF_TOPIC_SAMPLES]
        ).get_issues("off_topic_samples")
        self.assertEqual(
            int(np.asarray(v_e["indices"])[0]),
            int(np.asarray(v_a["indices"])[0]),
        )

    def test_linkage_matrices_identical_when_K_full(self):
        """With K = N - 1 the KNN graph is the complete graph, so its MST is
        the true MST, and the dendrogram from union-find equals
        `scipy.cluster.hierarchy.single` up to merge ordering of equal-weight
        edges. We compare the merge-distance sequence, which is invariant to
        labelling."""
        from scipy.cluster.hierarchy import single

        emb = self._planted_outlier_dataset()
        approx = _fit_approx(emb, K=_N - 1)
        approx_dists = np.sort(approx.knn_linkage_matrix[:, 2])

        # Exact single-linkage dendrogram from the full pairwise distances
        # SelfClean would build (1 - cos_sim) / 2 in the exact path.
        from sklearn.metrics.pairwise import cosine_similarity

        cs = cosine_similarity(emb)
        full_dist = (1.0 - cs) / 2.0
        # condense (upper triangle) to feed scipy
        triu = full_dist[np.triu_indices(_N, k=1)]
        np.clip(triu, 0.0, 1.0, out=triu)
        exact_link = single(triu)
        exact_dists = np.sort(exact_link[:, 2])

        self.assertTrue(np.allclose(approx_dists, exact_dists, atol=1e-4))


class TestApproxLADAutoCleaning(unittest.TestCase):
    def test_auto_cleaning_runs_in_approximate_mode(self):
        emb = _seeded_emb()
        cleaner = SelfCleanCleaner(
            memmap=False,
            approximate_nn=True,
            approx_no_neighbors=_K,
            auto_cleaning=True,
        )
        cleaner.fit(emb_space=emb)
        v = cleaner.predict(
            issues_to_detect=[IssueTypes.OFF_TOPIC_SAMPLES]
        ).get_issues("off_topic_samples")
        self.assertIn("auto_issues", v)
        auto = np.asarray(v["auto_issues"])
        self.assertEqual(auto.dtype.kind, "i")
        self.assertTrue((auto >= 0).all())
        self.assertTrue((auto < _N).all())
        self.assertEqual(len(np.unique(auto)), len(auto))


class TestApproxLADRequiresFit(unittest.TestCase):
    def test_calling_off_topic_without_fit_raises(self):
        cleaner = SelfCleanCleaner(
            memmap=False, approximate_nn=True, approx_no_neighbors=_K
        )
        # not fit yet -> no knn_linkage_matrix attribute
        with self.assertRaises(RuntimeError):
            cleaner.get_off_topic_ranking()


class TestExactLADUntouched(unittest.TestCase):
    """The exact LAD path is unchanged when `approximate_nn=False`."""

    def test_lad_path_still_recovers_planted_outlier(self):
        emb = np.zeros((_N, _D), dtype=np.float64)
        emb[-1] = np.ones(_D)

        cleaner = SelfCleanCleaner(memmap=False, approximate_nn=False)
        cleaner.fit(emb_space=emb)
        v = cleaner.predict(
            issues_to_detect=[IssueTypes.OFF_TOPIC_SAMPLES]
        ).get_issues("off_topic_samples")
        self.assertEqual(int(np.asarray(v["indices"])[0]), _N - 1)


if __name__ == "__main__":
    unittest.main()
