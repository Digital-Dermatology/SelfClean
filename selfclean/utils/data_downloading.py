import os
import tarfile
from pathlib import Path
from typing import Union

import pandas as pd
import requests
from torchvision import datasets

OXFORD_PETS_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"

imagenette_labels = {
    "n02979186": "cassette_player",
    "n03417042": "garbage_truck",
    "n01440764": "tench",
    "n02102040": "English_springer",
    "n03028079": "church",
    "n03888257": "parachute",
    "n03394916": "French_horn",
    "n03000684": "chain_saw",
    "n03445777": "golf_ball",
    "n03425413": "gas_pump",
}


def class_name_from_file(img_path: str) -> str:
    return "_".join(Path(img_path).stem.split("_")[:-1])


def get_oxford_pets3t(
    root_path: Union[Path, str] = "oxford_pets3t",
    return_dataframe: bool = False,
    **kwargs,
):
    root_path = Path(root_path)
    if not (root_path / "images").exists():
        root_path.mkdir(parents=True, exist_ok=True)
        response = requests.get(OXFORD_PETS_URL, stream=True)
        tar_path = root_path / "images.tar.gz"
        with open(tar_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        with tarfile.open(tar_path) as tar:
            tar.extractall(root_path)
        os.remove(tar_path)
    else:
        print(f"Oxford PetIIIT already downloaded to `{root_path}`.")

    dataset = datasets.ImageFolder(root=str(root_path), **kwargs)
    classes = list(
        set([class_name_from_file(samples[0]) for samples in dataset.samples])
    )
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    targets = [class_to_idx.get(class_name_from_file(x[0])) for x in dataset.samples]
    samples = [
        (sample[0], new_target) for sample, new_target in zip(dataset.samples, targets)
    ]

    dataset.classes = classes
    dataset.class_to_idx = class_to_idx
    dataset.targets = targets
    dataset.samples = samples

    if return_dataframe:
        return create_dataframe_from_dataset(samples, dataset)
    return dataset


def get_imagenette(
    root_path: Union[Path, str] = "imagenette",
    return_dataframe: bool = False,
    **kwargs,
):
    root_path = Path(root_path)
    if not (root_path / "imagenette2-160").exists():
        root_path.mkdir(parents=True, exist_ok=True)
        response = requests.get(IMAGENETTE_URL, stream=True)
        tar_path = root_path / "imagenette2-160.tgz"
        with open(tar_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        with tarfile.open(tar_path) as tar:
            tar.extractall(root_path)
        os.remove(tar_path)
    else:
        print(f"ImageNette already downloaded to `{root_path}`.")

    root_path = root_path / "imagenette2-160"
    dataset = datasets.ImageFolder(root=str(root_path), **kwargs)
    classes = list(set([samples[0].split("/")[4] for samples in dataset.samples]))
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    targets = [class_to_idx.get(x[0].split("/")[4]) for x in dataset.samples]
    samples = [
        (sample[0], new_target) for sample, new_target in zip(dataset.samples, targets)
    ]

    dataset.classes = [imagenette_labels.get(x).lower() for x in classes]
    dataset.class_to_idx = class_to_idx
    dataset.targets = targets
    dataset.samples = samples

    if return_dataframe:
        return create_dataframe_from_dataset(samples, dataset)
    return dataset


def create_dataframe_from_dataset(samples, dataset):
    df = pd.DataFrame(samples, columns=["img_path", "label"])
    df["label_name"] = df["label"].apply(lambda x: dataset.classes[x])
    df["img_path"] = df["img_path"].astype(str)
    return dataset, df
