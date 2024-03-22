import unittest

import numpy as np

from src.cleaner.base_cleaner import BaseCleaner
from src.cleaner.selfclean_cleaner import SelfCleanCleaner


class TestSelfCleanCleaner(unittest.TestCase):
    def setUp(self):
        self.emb_space = np.random.rand(50, 198)
        self.labels = np.random.randint(5, size=50)
        self.class_labels = [f"test_{x}" for x in np.unique(self.labels)]

    def test_fit(self):
        cleaner = SelfCleanCleaner(memmap=False)
        cleaner.fit(emb_space=self.emb_space, labels=self.labels)
        self.assertIsInstance(cleaner, BaseCleaner)
        self.assertIsNotNone(cleaner.distance_matrix)
        self.assertIsNotNone(cleaner.p_distances)

    def test_fit_with_memmaps(self):
        cleaner = SelfCleanCleaner(memmap=True)
        cleaner.fit(emb_space=self.emb_space, labels=self.labels)
        self.assertIsNotNone(cleaner.distance_matrix)
        self.assertIsNotNone(cleaner.p_distances)

    def test_predict(self):
        cleaner = SelfCleanCleaner(memmap=False)
        cleaner.fit(emb_space=self.emb_space, labels=self.labels)
        out_dict = cleaner.predict()
        for issue_type in ["irrelevants", "near_duplicates", "label_errors"]:
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
        for issue_type in ["irrelevants", "near_duplicates", "label_errors"]:
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
        for issue_type in ["irrelevants", "near_duplicates", "label_errors"]:
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
        for issue_type in ["irrelevants", "near_duplicates", "label_errors"]:
            v = out_dict.get_issues(issue_type)
            self.assertIsNotNone(v)
            self.assertTrue("indices" in v)
            self.assertTrue("scores" in v)
        self.assertIsNone(out_dict.get_issues("label_errors")["indices"])
        self.assertIsNone(out_dict.get_issues("label_errors")["scores"])

    def test_predict_distance_function(self):
        cleaner = SelfCleanCleaner(
            memmap=False,
            distance_function_path="",
            distance_function_name="pairwise_projective_distance",
        )
        cleaner.fit(emb_space=self.emb_space, labels=self.labels)
        out_dict = cleaner.predict()
        for issue_type in ["irrelevants", "near_duplicates", "label_errors"]:
            v = out_dict.get_issues(issue_type)
            self.assertIsNotNone(v)
            self.assertTrue("indices" in v)
            self.assertTrue("scores" in v)
            self.assertIsNotNone(v["indices"])
            self.assertIsNotNone(v["scores"])


if __name__ == "__main__":
    unittest.main()
