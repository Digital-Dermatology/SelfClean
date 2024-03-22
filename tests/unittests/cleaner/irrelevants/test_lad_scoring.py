import unittest

import numpy as np

from src.cleaner.selfclean_cleaner import SelfCleanCleaner


class TestIrrelevantsLADScoring(unittest.TestCase):
    def test_far_outlier(self):
        emb_space = np.zeros(shape=(50, 198))
        emb_space[-1] = np.ones_like(emb_space[0])

        cleaner = SelfCleanCleaner(memmap=False)
        cleaner.fit(emb_space=emb_space)
        out_dict = cleaner.predict()

        self.assertEqual(
            out_dict.get_issues("irrelevants")["indices"][0], len(emb_space) - 1
        )


if __name__ == "__main__":
    unittest.main()
