import unittest

import numpy as np

from src.cleaner.base_cleaner import BaseCleaner
from src.cleaner.selfclean_cleaner import SelfCleanCleaner


class TestSelfCleanCleaner(unittest.TestCase):
    def setUp(self):
        self.emb_space = np.random.rand(50, 198)
        self.labels = np.random.randint(5, size=50)

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
        self.assertTrue("irrelevants" in out_dict)
        self.assertTrue("near_duplicates" in out_dict)
        self.assertTrue("label_errors" in out_dict)
        for v in out_dict.values():
            self.assertTrue("indices" in v)
            self.assertTrue("scores" in v)
            self.assertIsNotNone(v["indices"])
            self.assertIsNotNone(v["scores"])

    def test_predict_with_plotting(self):
        cleaner = SelfCleanCleaner(
            memmap=False,
            plot_distribution=True,
            plot_top_N=10,
        )
        cleaner.fit(emb_space=self.emb_space, labels=self.labels)
        out_dict = cleaner.predict()
        self.assertTrue("irrelevants" in out_dict)
        self.assertTrue("near_duplicates" in out_dict)
        self.assertTrue("label_errors" in out_dict)
        for v in out_dict.values():
            self.assertTrue("indices" in v)
            self.assertTrue("scores" in v)
            self.assertIsNotNone(v["indices"])
            self.assertIsNotNone(v["scores"])

    def test_predict_without_labels(self):
        cleaner = SelfCleanCleaner(memmap=False)
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

    def test_predict_distance_function(self):
        cleaner = SelfCleanCleaner(
            memmap=False,
            distance_function_path="src.distances.projective_distance.",
            distance_function_name="pairwise_projective_distance",
        )
        cleaner.fit(emb_space=self.emb_space, labels=self.labels)
        out_dict = cleaner.predict()
        self.assertTrue("irrelevants" in out_dict)
        self.assertTrue("near_duplicates" in out_dict)
        self.assertTrue("label_errors" in out_dict)
        for v in out_dict.values():
            self.assertTrue("indices" in v)
            self.assertTrue("scores" in v)
            self.assertIsNotNone(v["indices"])
            self.assertIsNotNone(v["scores"])


if __name__ == "__main__":
    unittest.main()
