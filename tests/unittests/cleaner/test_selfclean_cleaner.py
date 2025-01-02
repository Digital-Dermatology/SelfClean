import unittest

import numpy as np
from memory_profiler import profile

from src.cleaner.base_cleaner import BaseCleaner
from src.cleaner.issue_manager import IssueTypes
from src.cleaner.selfclean_cleaner import SelfCleanCleaner


class TestSelfCleanCleaner(unittest.TestCase):
    def setUp(self):
        self.emb_space = np.random.rand(50, 198)
        self.labels = np.random.randint(5, size=50)
        self.class_labels = [f"test_{x}" for x in np.unique(self.labels)]
        self.memory_profiling = False

    def test_fit(self):
        cleaner = SelfCleanCleaner(memmap=False)
        self.assertEqual(cleaner.is_fitted, False)
        if self.memory_profiling:
            cleaner.fit = profile(cleaner.fit, precision=4)
        cleaner.fit(emb_space=self.emb_space, labels=self.labels)
        self.assertEqual(cleaner.is_fitted, True)
        self.assertIsInstance(cleaner, BaseCleaner)
        self.assertIsNotNone(cleaner.distance_matrix)
        self.assertIsNotNone(cleaner.p_distances)

    def test_fit_with_memmaps(self):
        cleaner = SelfCleanCleaner(memmap=True)
        if self.memory_profiling:
            cleaner.fit = profile(cleaner.fit, precision=4)
        cleaner.fit(emb_space=self.emb_space, labels=self.labels)
        self.assertIsNotNone(cleaner.distance_matrix)
        self.assertIsNotNone(cleaner.p_distances)

    def test_predict(self):
        cleaner = SelfCleanCleaner(memmap=False)
        if self.memory_profiling:
            cleaner.fit = profile(cleaner.fit, precision=4)
        cleaner.fit(emb_space=self.emb_space, labels=self.labels)
        if self.memory_profiling:
            cleaner.predict = profile(cleaner.predict, precision=4)
        out_dict = cleaner.predict()
        for issue_type in ["off_topic_samples", "near_duplicates", "label_errors"]:
            v = out_dict.get_issues(issue_type)
            self.assertIsNotNone(v)
            self.assertTrue("indices" in v)
            self.assertTrue("scores" in v)
            self.assertIsNotNone(v["indices"])
            self.assertIsNotNone(v["scores"])
            self.assertEqual(len(v["indices"]), len(v["scores"]))

    def test_2_predict(self):
        cleaner = SelfCleanCleaner(memmap=False)
        cleaner.fit(emb_space=self.emb_space, labels=self.labels)
        out_dict = cleaner.predict()
        out_dict = cleaner.predict()
        for issue_type in ["off_topic_samples", "near_duplicates", "label_errors"]:
            v = out_dict.get_issues(issue_type)
            self.assertIsNotNone(v)
            self.assertTrue("indices" in v)
            self.assertTrue("scores" in v)
            self.assertIsNotNone(v["indices"])
            self.assertIsNotNone(v["scores"])
            self.assertEqual(len(v["indices"]), len(v["scores"]))

    def test_predict_with_class_labels(self):
        cleaner = SelfCleanCleaner(memmap=False)
        cleaner.fit(
            emb_space=self.emb_space,
            labels=self.labels,
            class_labels=self.class_labels,
        )
        out_dict = cleaner.predict()
        for issue_type in ["off_topic_samples", "near_duplicates", "label_errors"]:
            v = out_dict.get_issues(issue_type)
            self.assertIsNotNone(v)
            self.assertTrue("indices" in v)
            self.assertTrue("scores" in v)
            self.assertIsNotNone(v["indices"])
            self.assertIsNotNone(v["scores"])
            self.assertEqual(len(v["indices"]), len(v["scores"]))

    def test_predict_with_plotting(self):
        cleaner = SelfCleanCleaner(
            memmap=False,
            plot_distribution=True,
            plot_top_N=10,
        )
        cleaner.fit(emb_space=self.emb_space, labels=self.labels)
        out_dict = cleaner.predict()
        for issue_type in ["off_topic_samples", "near_duplicates", "label_errors"]:
            v = out_dict.get_issues(issue_type)
            self.assertIsNotNone(v)
            self.assertTrue("indices" in v)
            self.assertTrue("scores" in v)
            self.assertIsNotNone(v["indices"])
            self.assertIsNotNone(v["scores"])

    def test_predict_without_labels(self):
        cleaner = SelfCleanCleaner(memmap=False)
        cleaner.fit(emb_space=self.emb_space)
        out_dict = cleaner.predict()
        for issue_type in ["off_topic_samples", "near_duplicates"]:
            v = out_dict.get_issues(issue_type)
            self.assertIsNotNone(v)
            self.assertTrue("indices" in v)
            self.assertTrue("scores" in v)
        self.assertIsNone(out_dict.get_issues("label_errors"))

    def test_predict_distance_function(self):
        cleaner = SelfCleanCleaner(
            memmap=False,
            distance_function_path="",
            distance_function_name="pairwise_projective_distance",
        )
        cleaner.fit(emb_space=self.emb_space, labels=self.labels)
        out_dict = cleaner.predict()
        for issue_type in ["off_topic_samples", "near_duplicates", "label_errors"]:
            v = out_dict.get_issues(issue_type)
            self.assertIsNotNone(v)
            self.assertTrue("indices" in v)
            self.assertTrue("scores" in v)
            self.assertIsNotNone(v["indices"])
            self.assertIsNotNone(v["scores"])

    def test_predict_single_issues(self):
        cleaner = SelfCleanCleaner(memmap=False)
        cleaner.fit(emb_space=self.emb_space, labels=self.labels)
        out_dict = cleaner.predict(issues_to_detect=[IssueTypes.OFF_TOPIC_SAMPLES])
        for issue_type in ["off_topic_samples"]:
            v = out_dict.get_issues(issue_type)
            self.assertIsNotNone(v)
            self.assertTrue("indices" in v)
            self.assertTrue("scores" in v)
            self.assertIsNotNone(v["indices"])
            self.assertIsNotNone(v["scores"])
        for issue_type in ["near_duplicates", "label_errors"]:
            v = out_dict.get_issues(issue_type)
            self.assertIsNone(v)

    def test_predict_multi_issues(self):
        cleaner = SelfCleanCleaner(memmap=False)
        cleaner.fit(emb_space=self.emb_space, labels=self.labels)
        out_dict = cleaner.predict(
            issues_to_detect=[
                IssueTypes.LABEL_ERRORS,
                IssueTypes.NEAR_DUPLICATES,
            ],
        )
        for issue_type in ["near_duplicates", "label_errors"]:
            v = out_dict.get_issues(issue_type)
            self.assertIsNotNone(v)
            self.assertTrue("indices" in v)
            self.assertTrue("scores" in v)
            self.assertIsNotNone(v["indices"])
            self.assertIsNotNone(v["scores"])
        for issue_type in ["off_topic_samples"]:
            v = out_dict.get_issues(issue_type)
            self.assertIsNone(v)

    def test_approx_nearest_duplicates(self):
        cleaner = SelfCleanCleaner(
            memmap=False,
            approximate_nn=True,
            approx_no_neighbors=10,
        )
        cleaner.fit(emb_space=self.emb_space, labels=self.labels)
        out_dict = cleaner.predict(issues_to_detect=[IssueTypes.NEAR_DUPLICATES])
        for issue_type in ["approx_near_duplicates"]:
            v = out_dict.get_issues(issue_type)
            self.assertIsNotNone(v)
            self.assertEqual(len([x for x in v.columns if "nn_idx_" in x]), 10 - 1)
            self.assertEqual(len([x for x in v.columns if "nn_dist_" in x]), 10 - 1)
        for issue_type in ["near_duplicates", "off_topic_samples", "label_errors"]:
            v = out_dict.get_issues(issue_type)
            self.assertIsNone(v)

    def test_approx_nearest_duplicates_w_exact(self):
        cleaner = SelfCleanCleaner(
            memmap=False,
            approximate_nn=True,
            approx_no_neighbors=len(self.emb_space),
        )
        cleaner.fit(emb_space=self.emb_space, labels=self.labels)
        out_dict = cleaner.predict(issues_to_detect=[IssueTypes.NEAR_DUPLICATES])
        df_approx_nn = out_dict.get_issues("approx_near_duplicates")

        # fit without approximation
        cleaner.approximate_nn = False
        out_dict = cleaner.predict(issues_to_detect=[IssueTypes.NEAR_DUPLICATES])
        df_nn = out_dict.get_issues("near_duplicates", return_as_df=True)

        # check if they align
        for index in range(len(self.emb_space)):
            nn = df_nn[
                (df_nn["indices_1"] == index) | (df_nn["indices_2"] == index)
            ].iloc[0]
            nn_approx = df_approx_nn[df_approx_nn["seed_idx"] == index].iloc[0]
            idx = nn["indices_1"] if nn["indices_1"] != index else nn["indices_2"]
            idx_approx = nn_approx["nn_idx_1"]
            self.assertEqual(idx, idx_approx)


if __name__ == "__main__":
    unittest.main()
