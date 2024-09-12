import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.utils.utils import triu_indices_memmap


class TestUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp_path = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_path)

    def test_triu_indices_memmap(self):
        l_N = np.arange(1, 100, 10)
        for N in l_N:
            l_k = np.arange(0, N)
            for k in l_k:
                triu_indices = np.triu_indices(N, k=k)
                triu_indices_mem = triu_indices_memmap(
                    str(TestUtils.temp_path / "triu_indices"),
                    N=N,
                    k=k,
                )
                self.assertTrue(np.array_equal(triu_indices_mem, triu_indices))


if __name__ == "__main__":
    unittest.main()
