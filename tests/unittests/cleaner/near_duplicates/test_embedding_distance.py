import unittest

import numpy as np

from src.cleaner.selfclean_cleaner import SelfCleanCleaner


class TestNearDuplicateDistance(unittest.TestCase):
    def test_distances_with_zeros(self):
        emb_space = np.zeros(shape=(50, 198))
        labels = np.zeros(shape=(50))

        cleaner = SelfCleanCleaner(memmap=False)
        cleaner.fit(emb_space=emb_space, labels=labels)
        out_dict = cleaner.predict()

        self.assertTrue(
            (cleaner.distance_matrix == np.ones(shape=(50, 50)) * 0.5).all()
        )
        self.assertTrue((out_dict["near_duplicates"]["scores"] == 0.5).all())

    def test_exact_duplicate_distances(self):
        emb_space = np.random.rand(50, 198)
        labels = np.random.randint(5, size=50)

        # create an exact duplicate
        emb_space[-1] = emb_space[0]
        labels[-1] = labels[0]

        cleaner = SelfCleanCleaner(memmap=False)
        cleaner.fit(emb_space=emb_space, labels=labels)
        out_dict = cleaner.predict()

        self.assertTrue(
            (out_dict["near_duplicates"]["indices"][0] == [0, len(emb_space) - 1]).all()
        )
        self.assertAlmostEqual(out_dict["near_duplicates"]["scores"][0], 0)


if __name__ == "__main__":
    unittest.main()
