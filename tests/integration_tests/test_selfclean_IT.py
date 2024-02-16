import os
import re
import shutil
import unittest

from torchvision.datasets import FakeData

from src.cleaner.selfclean import PretrainingType, SelfClean
from tests.testutils.paths import testfiles_path


class TestSelfCleanIT(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        pattern = r".*DINO-(test_files|FakeData).*"
        searchDir = "./"
        innerDirs = os.listdir(searchDir)
        for dir_path in innerDirs:
            if re.search(pattern, dir_path):
                shutil.rmtree(searchDir + dir_path)

    def test_run_with_files_dino(self):
        selfclean = SelfClean()
        out_dict = selfclean.run_on_image_folder(
            input_path=testfiles_path,
            pretraining_type=PretrainingType.DINO,
        )
        self.assertTrue("irrelevants" in out_dict)
        self.assertTrue("near_duplicates" in out_dict)
        self.assertTrue("label_errors" in out_dict)
        for v in out_dict.values():
            self.assertTrue("indices" in v)
            self.assertTrue("scores" in v)
            self.assertIsNotNone(v["indices"])
            self.assertIsNotNone(v["scores"])

    def test_run_with_files_imagenet(self):
        selfclean = SelfClean()
        out_dict = selfclean.run_on_image_folder(
            input_path=testfiles_path,
            pretraining_type=PretrainingType.IMAGENET,
        )
        self.assertTrue("irrelevants" in out_dict)
        self.assertTrue("near_duplicates" in out_dict)
        self.assertTrue("label_errors" in out_dict)
        for v in out_dict.values():
            self.assertTrue("indices" in v)
            self.assertTrue("scores" in v)
            self.assertIsNotNone(v["indices"])
            self.assertIsNotNone(v["scores"])

    def test_run_with_files_imagenet_vit(self):
        selfclean = SelfClean()
        out_dict = selfclean.run_on_image_folder(
            input_path=testfiles_path,
            pretraining_type=PretrainingType.IMAGENET_VIT,
        )
        self.assertTrue("irrelevants" in out_dict)
        self.assertTrue("near_duplicates" in out_dict)
        self.assertTrue("label_errors" in out_dict)
        for v in out_dict.values():
            self.assertTrue("indices" in v)
            self.assertTrue("scores" in v)
            self.assertIsNotNone(v["indices"])
            self.assertIsNotNone(v["scores"])

    def test_run_with_dataset(self):
        fake_dataset = FakeData(size=20)
        selfclean = SelfClean()
        out_dict = selfclean.run_on_dataset(
            dataset=fake_dataset,
        )
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
