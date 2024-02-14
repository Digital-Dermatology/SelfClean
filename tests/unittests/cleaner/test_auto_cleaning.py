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
        self.assertTrue("irrelevants" in out_dict)
        self.assertTrue("near_duplicates" in out_dict)
        self.assertTrue("label_errors" in out_dict)
        for v in out_dict.values():
            self.assertTrue("indices" in v)
            self.assertTrue("scores" in v)
            self.assertTrue("auto_issues" not in v)
            self.assertIsNotNone(v["indices"])
            self.assertIsNotNone(v["scores"])

    def test_predict_auto_cleaning(self):
        cleaner = SelfCleanCleaner(memmap=False, auto_cleaning=True)
        cleaner.fit(emb_space=self.emb_space, labels=self.labels)
        out_dict = cleaner.predict()
        self.assertTrue("irrelevants" in out_dict)
        self.assertTrue("near_duplicates" in out_dict)
        self.assertTrue("label_errors" in out_dict)
        for v in out_dict.values():
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
        self.assertTrue("irrelevants" in out_dict)
        self.assertTrue("near_duplicates" in out_dict)
        self.assertTrue("label_errors" in out_dict)
        for v in out_dict.values():
            self.assertTrue("indices" in v)
            self.assertTrue("scores" in v)
        self.assertIsNone(out_dict["label_errors"]["indices"])
        self.assertIsNone(out_dict["label_errors"]["scores"])

    def test_predict_auto_cleaning_with_plotting(self):
        cleaner = SelfCleanCleaner(
            memmap=False,
            auto_cleaning=True,
            cleaner_kwargs={"debug": True, "plot_result": True},
        )
        cleaner.fit(emb_space=self.emb_space, labels=self.labels)
        out_dict = cleaner.predict()
        self.assertTrue("irrelevants" in out_dict)
        self.assertTrue("near_duplicates" in out_dict)
        self.assertTrue("label_errors" in out_dict)
        for v in out_dict.values():
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
            irrelevant_cut_off=0.01,
            near_duplicate_cut_off=0.01,
            label_error_cut_off=0.01,
        )
        cleaner.fit(emb_space=self.emb_space, labels=self.labels)
        out_dict = cleaner.predict()

        cleaner.irrelevant_cut_off = 0.5
        cleaner.near_duplicate_cut_off = 0.5
        cleaner.label_error_cut_off = 0.5
        out_dict2 = cleaner.predict()

        for v1, v2 in zip(out_dict.values(), out_dict2.values()):
            self.assertTrue((v1["indices"] == v2["indices"]).all())
            self.assertTrue((v1["scores"] == v2["scores"]).all())

    def test_threshold_sensitivity(self):
        cleaner = SelfCleanCleaner(
            memmap=False,
            auto_cleaning=False,
        )
        cleaner.fit(emb_space=self.emb_space, labels=self.labels)
        out_dict = cleaner.predict()
        scores = out_dict["near_duplicates"]["scores"]
        cleaner.threshold_sensitivity(scores=scores)

    def test_alpha_sensitivity(self):
        cleaner = SelfCleanCleaner(
            memmap=False,
            auto_cleaning=False,
        )
        cleaner.fit(emb_space=self.emb_space, labels=self.labels)
        out_dict = cleaner.predict()
        scores = out_dict["near_duplicates"]["scores"]
        cleaner.alpha_sensitivity(scores=scores)


if __name__ == "__main__":
    unittest.main()
