import unittest

import numpy as np

from selfclean.cleaner.selfclean_cleaner import SelfCleanCleaner


class TestLabelErrorIntraExtraDistance(unittest.TestCase):
    def test_raises_wo_label(self):
        emb_space = np.zeros(shape=(50, 198))

        cleaner = SelfCleanCleaner(memmap=False)
        cleaner.fit(emb_space=emb_space)

        with self.assertWarns(Warning):
            cleaner.predict()

    def test_raises_single_label(self):
        emb_space = np.zeros(shape=(50, 198))
        labels = np.zeros(shape=(50))

        cleaner = SelfCleanCleaner(memmap=False)
        cleaner.fit(emb_space=emb_space, labels=labels)

        with self.assertWarns(Warning):
            cleaner.predict()

    def test_exact_duplicate_with_diff_label(self):
        emb_space = np.random.rand(50, 198)
        labels = np.random.randint(5, size=50)

        # create an exact duplicate with a different label
        emb_space[-1] = emb_space[0]
        diff_available_labels = list(range(5))
        del diff_available_labels[labels[0]]
        labels[-1] = np.random.choice(diff_available_labels)

        cleaner = SelfCleanCleaner(memmap=False)
        cleaner.fit(emb_space=emb_space, labels=labels)
        out_dict = cleaner.predict()

        np.testing.assert_array_equal(
            np.sort(out_dict.get_issues("label_errors")["indices"][:2]),
            np.sort(np.asarray([0, len(emb_space) - 1])),
        )
        self.assertAlmostEqual(out_dict.get_issues("label_errors")["scores"][0], 0)


if __name__ == "__main__":
    unittest.main()
