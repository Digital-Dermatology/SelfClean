import os
import re
import shutil
import tempfile
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

    def test_run_with_files_dino_in_workdir(self):
        temp_work_dir = tempfile.TemporaryDirectory()
        selfclean = SelfClean()
        out_dict = selfclean.run_on_image_folder(
            input_path=testfiles_path,
            pretraining_type=PretrainingType.DINO,
            work_dir=temp_work_dir.name,
            epochs=1,
            num_workers=4,
        )
        self.assertTrue("irrelevants" in out_dict)
        self.assertTrue("near_duplicates" in out_dict)
        self.assertTrue("label_errors" in out_dict)
        for v in out_dict.values():
            self.assertTrue("indices" in v)
            self.assertTrue("scores" in v)
            self.assertIsNotNone(v["indices"])
            self.assertIsNotNone(v["scores"])

    def test_run_with_files_dino_wo_pretraining(self):
        selfclean = SelfClean()
        out_dict = selfclean.run_on_image_folder(
            input_path=testfiles_path,
            pretraining_type=PretrainingType.DINO,
            ssl_pre_training=False,
            num_workers=4,
        )
        self.assertTrue("irrelevants" in out_dict)
        self.assertTrue("near_duplicates" in out_dict)
        self.assertTrue("label_errors" in out_dict)
        for v in out_dict.values():
            self.assertTrue("indices" in v)
            self.assertTrue("scores" in v)
            self.assertIsNotNone(v["indices"])
            self.assertIsNotNone(v["scores"])

    def test_run_with_files_dino(self):
        selfclean = SelfClean()
        out_dict = selfclean.run_on_image_folder(
            input_path=testfiles_path,
            pretraining_type=PretrainingType.DINO,
            epochs=1,
            num_workers=4,
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
            num_workers=4,
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
            epochs=1,
            num_workers=4,
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
            epochs=1,
            num_workers=4,
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
