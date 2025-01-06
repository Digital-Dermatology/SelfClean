import os
import re
import shutil
import tempfile
import unittest
from pathlib import Path

from torchvision.datasets import FakeData

from selfclean.cleaner.issue_manager import IssueTypes
from selfclean.cleaner.selfclean import PretrainingType, SelfClean
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
        self._check_output(out_dict)

    def test_run_with_files_dino_with_output_path(self):
        temp_work_dir = tempfile.TemporaryDirectory()
        selfclean = SelfClean(output_path=str(Path(temp_work_dir.name) / "output"))
        out_dict = selfclean.run_on_image_folder(
            input_path=testfiles_path,
            pretraining_type=PretrainingType.DINO,
            work_dir=temp_work_dir.name,
            epochs=1,
            num_workers=4,
        )
        self._check_output(out_dict)

    def test_run_with_files_dino_single_issue_type(self):
        temp_work_dir = tempfile.TemporaryDirectory()
        selfclean = SelfClean()
        out_dict = selfclean.run_on_image_folder(
            input_path=testfiles_path,
            pretraining_type=PretrainingType.DINO,
            work_dir=temp_work_dir.name,
            epochs=1,
            num_workers=4,
            issues_to_detect=[IssueTypes.OFF_TOPIC_SAMPLES],
        )
        self._check_output(out_dict, issue_types=["off_topic_samples"])

    def test_run_with_files_dino_wo_pretraining(self):
        selfclean = SelfClean()
        out_dict = selfclean.run_on_image_folder(
            input_path=testfiles_path,
            pretraining_type=PretrainingType.DINO,
            ssl_pre_training=False,
            num_workers=4,
        )
        self._check_output(out_dict)

    def test_run_with_files_dino(self):
        selfclean = SelfClean()
        out_dict = selfclean.run_on_image_folder(
            input_path=testfiles_path,
            pretraining_type=PretrainingType.DINO,
            epochs=1,
            num_workers=4,
        )
        self._check_output(out_dict)

    def test_run_with_files_imagenet(self):
        selfclean = SelfClean()
        out_dict = selfclean.run_on_image_folder(
            input_path=testfiles_path,
            pretraining_type=PretrainingType.IMAGENET,
            num_workers=4,
        )
        self._check_output(out_dict)

    def test_run_with_files_imagenet_vit(self):
        selfclean = SelfClean()
        out_dict = selfclean.run_on_image_folder(
            input_path=testfiles_path,
            pretraining_type=PretrainingType.IMAGENET_VIT,
            epochs=1,
            num_workers=4,
        )
        self._check_output(out_dict)

    def test_run_with_dataset(self):
        fake_dataset = FakeData(size=20)
        selfclean = SelfClean()
        out_dict = selfclean.run_on_dataset(
            dataset=fake_dataset,
            epochs=1,
            num_workers=4,
        )
        self._check_output(out_dict, check_path_exists=False)

    def test_run_with_plotting(self):
        fake_dataset = FakeData(size=20)
        selfclean = SelfClean(
            memmap=False,
            plot_distribution=True,
            plot_top_N=7,
        )
        out_dict = selfclean.run_on_dataset(
            dataset=fake_dataset,
            epochs=1,
            num_workers=4,
        )
        self._check_output(out_dict, check_path_exists=False)

    def _check_output(
        self,
        out_dict,
        issue_types=["off_topic_samples", "near_duplicates", "label_errors"],
        check_path_exists: bool = True,
    ):
        for issue_type in issue_types:
            # check the output dataframe
            _df = out_dict.get_issues(issue_type, return_as_df=True)
            self.assertTrue(
                "indices" in _df.columns
                or ("indices_1" in _df.columns and "indices_2" in _df.columns)
            )
            self.assertTrue("scores" in _df.columns)
            if check_path_exists:
                self.assertTrue(
                    "path" in _df.columns
                    or (
                        "path_indices_1" in _df.columns
                        and "path_indices_2" in _df.columns
                    )
                )
            self.assertTrue(
                "label" in _df.columns
                or (
                    "label_indices_1" in _df.columns
                    and "label_indices_2" in _df.columns
                )
            )
            self.assertEqual(_df.isna().sum().sum(), 0)
            # check the output array
            v = out_dict.get_issues(issue_type)
            self.assertIsNotNone(v)
            self.assertTrue("indices" in v)
            self.assertTrue("scores" in v)
            self.assertIsNotNone(v["indices"])
            self.assertIsNotNone(v["scores"])


if __name__ == "__main__":
    unittest.main()
