import unittest

import numpy as np

from src.cleaner.issue_manager import IssueManager
from src.cleaner.selfclean_cleaner import SelfCleanCleaner


class TestIssueManager(unittest.TestCase):
    def setUp(self):
        self.emb_space = np.random.rand(50, 198)
        self.labels = np.random.randint(5, size=50)
        self.cleaner = SelfCleanCleaner(memmap=False, auto_cleaning=True)
        self.cleaner.fit(emb_space=self.emb_space, labels=self.labels)
        self.issues = self.cleaner.predict()

    def test_get(self):
        self.assertIsInstance(self.issues, IssueManager)
        for issue_type in ["irrelevants", "near_duplicates", "label_errors"]:
            v = self.issues[issue_type]
            self.assertIsNotNone(v)
            self.assertTrue("indices" in v)
            self.assertTrue("scores" in v)
            self.assertIsNotNone(v["indices"])
            self.assertIsNotNone(v["scores"])
            self.assertEqual(len(v["indices"]), len(v["scores"]))

    def test_get_issues(self):
        self.assertIsInstance(self.issues, IssueManager)
        for issue_type in ["irrelevants", "near_duplicates", "label_errors"]:
            v = self.issues.get_issues(issue_type)
            self.assertIsNotNone(v)
            self.assertTrue("indices" in v)
            self.assertTrue("scores" in v)
            self.assertIsNotNone(v["indices"])
            self.assertIsNotNone(v["scores"])
            self.assertEqual(len(v["indices"]), len(v["scores"]))

    def test_get_issues_as_dataframe(self):
        self.assertIsInstance(self.issues, IssueManager)
        for issue_type in ["irrelevants", "near_duplicates", "label_errors"]:
            df = self.issues.get_issues(issue_type, return_as_df=True)
            self.assertIsNotNone(df)
            if issue_type == "near_duplicates":
                self.assertTrue("indices_1" in df.columns)
                self.assertTrue("indices_2" in df.columns)
                self.assertTrue("label_indices_1" in df.columns)
                self.assertTrue("label_indices_2" in df.columns)
            else:
                self.assertTrue("indices" in df.columns)
                self.assertTrue("label" in df.columns)
            self.assertTrue("scores" in df.columns)
            self.assertTrue("auto_issues" in df.columns)

    def test_get_wrong(self):
        self.assertIsInstance(self.issues, IssueManager)
        with self.assertRaises(ValueError):
            _ = self.issues['irr']


if __name__ == "__main__":
    unittest.main()
