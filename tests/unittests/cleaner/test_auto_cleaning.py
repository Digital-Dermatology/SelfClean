import unittest

import numpy as np

from src.cleaner.selfclean_cleaner import SelfCleanCleaner


class TestAutoCleaning(unittest.TestCase):
    def setUp(self):
        self.emb_space = np.random.rand(50, 198)
        self.labels = np.random.randint(5, size=50)

    def test_predict_without_auto_cleaning(self):
        cleaner = SelfCleanCleaner(memmap=False, auto_cleaning=False)
        cleaner.fit(emb_space=self.emb_space, labels=self.labels)
        out_dict = cleaner.predict()
        for issue_type in ["off_topic_samples", "near_duplicates", "label_errors"]:
            v = out_dict.get_issues(issue_type)
            self.assertIsNotNone(v)
            self.assertTrue("indices" in v)
            self.assertTrue("scores" in v)
            self.assertTrue("auto_issues" not in v)
            self.assertIsNotNone(v["indices"])
            self.assertIsNotNone(v["scores"])

    def test_predict_auto_cleaning(self):
        cleaner = SelfCleanCleaner(memmap=False, auto_cleaning=True)
        cleaner.fit(emb_space=self.emb_space, labels=self.labels)
        out_dict = cleaner.predict()
        for issue_type in ["off_topic_samples", "near_duplicates", "label_errors"]:
            v = out_dict.get_issues(issue_type)
            self.assertIsNotNone(v)
            self.assertTrue("indices" in v)
            self.assertTrue("scores" in v)
            self.assertTrue("auto_issues" in v)
            self.assertIsNotNone(v["indices"])
            self.assertIsNotNone(v["scores"])
            self.assertIsNotNone(v["auto_issues"])

    def test_predict_auto_cleaning_without_labels(self):
        cleaner = SelfCleanCleaner(memmap=False, auto_cleaning=True)
        cleaner.fit(emb_space=self.emb_space)
        out_dict = cleaner.predict()
        for issue_type in ["off_topic_samples", "near_duplicates"]:
            v = out_dict.get_issues(issue_type)
            self.assertIsNotNone(v)
            self.assertTrue("indices" in v)
            self.assertTrue("scores" in v)
        self.assertIsNone(out_dict.get_issues("label_errors"))

    def test_predict_auto_cleaning_with_plotting(self):
        cleaner = SelfCleanCleaner(
            memmap=False,
            auto_cleaning=True,
            cleaner_kwargs={"plot_result": True},
        )
        cleaner.fit(emb_space=self.emb_space, labels=self.labels)
        out_dict = cleaner.predict()
        for issue_type in ["off_topic_samples", "near_duplicates", "label_errors"]:
            v = out_dict.get_issues(issue_type)
            self.assertIsNotNone(v)
            self.assertTrue("indices" in v)
            self.assertTrue("scores" in v)
            self.assertTrue("auto_issues" in v)
            self.assertIsNotNone(v["indices"])
            self.assertIsNotNone(v["scores"])
            self.assertIsNotNone(v["auto_issues"])

    def test_predict_auto_cleaning_diff_cut_off(self):
        cleaner = SelfCleanCleaner(
            memmap=False,
            auto_cleaning=True,
            off_topic_cut_off=0.01,
            near_duplicate_cut_off=0.01,
            label_error_cut_off=0.01,
            significance_level=0.01,
        )
        cleaner.fit(emb_space=self.emb_space, labels=self.labels)
        out_dict = cleaner.predict()

        cleaner.off_topic_cut_off = 0.5
        cleaner.near_duplicate_cut_off = 0.5
        cleaner.label_error_cut_off = 0.5
        out_dict2 = cleaner.predict()

        for issue_type in ["off_topic_samples", "near_duplicates", "label_errors"]:
            v1 = out_dict.get_issues(issue_type)
            v2 = out_dict2.get_issues(issue_type)
            self.assertTrue((v1["indices"] == v2["indices"]).all())
            self.assertTrue((v1["scores"] == v2["scores"]).all())

    def test_threshold_sensitivity(self):
        cleaner = SelfCleanCleaner(
            memmap=False,
            auto_cleaning=False,
        )
        cleaner.fit(emb_space=self.emb_space, labels=self.labels)
        out_dict = cleaner.predict()
        scores = out_dict.get_issues("near_duplicates")["scores"]
        cleaner.threshold_sensitivity(scores=scores)

    def test_alpha_sensitivity(self):
        cleaner = SelfCleanCleaner(
            memmap=False,
            auto_cleaning=False,
        )
        cleaner.fit(emb_space=self.emb_space, labels=self.labels)
        out_dict = cleaner.predict()
        scores = out_dict.get_issues("near_duplicates")["scores"]
        cleaner.alpha_sensitivity(scores=scores)


if __name__ == "__main__":
    unittest.main()
