import gc
import os
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode

from ..cleaner.selfclean_cleaner import SelfCleanCleaner
from ..ssl_library.src.augmentations.ibot import iBOTDataAugmentation
from ..ssl_library.src.pkg import Embedder, embed_dataset
from ..ssl_library.src.trainers.dino_trainer import DINOTrainer
from ..ssl_library.src.utils.logging import set_log_level
from ..ssl_library.src.utils.utils import (
    cleanup,
    fix_random_seeds,
    init_distributed_mode,
)
from ..utils.utils import set_dataset_transformation

DINO_STANDARD_HYPERPARAMETERS = {
    "optim": "adamw",
    "lr": 0.0005,
    "min_lr": "1e-6",
    "weight_decay": 0.04,
    "weight_decay_end": 0.4,
    "warmup_epochs": 10,
    "momentum_teacher": 0.996,
    "clip_grad": 3.0,
    "apply_l2_norm": True,
    "save_every_n_epochs": 10,
    "model": {
        "out_dim": 4096,
        "emb_dim": 192,
        "base_model": "vit_tiny",
        "model_type": "VIT",
        "use_bn_in_head": False,
        "norm_last_layer": True,
        "student": {
            "drop_path_rate": 0.1,
            "pretrained": True,
        },
        "teacher": {
            "drop_path_rate": 0.1,
            "pretrained": True,
        },
        "eval": {"n_last_blocks": 4, "avgpool_patchtokens": False},
    },
    "dataset": {
        "augmentations": {
            "global_crops_scale": "(0.7, 1.)",
            "local_crops_scale": "(0.05, 0.4)",
            "global_crops_number": 2,
            "local_crops_number": 12,
            "random_rotation": True,
        }
    },
    "loss": {
        "warmup_teacher_temp": 0.04,
        "teacher_temp": 0.04,
        "warmup_teacher_temp_epochs": 30,
    },
    "optimizer": {"freeze_last_layer": 1},
}


class PretrainingType(Enum):
    IMAGENET = "imagenet"
    IMAGENET_VIT = "imagenet_vit_tiny"
    DINO = "dino"


class SelfClean:
    def __init__(
        self,
        # distance calculation
        distance_function_path: str = "sklearn.metrics.pairwise.",
        distance_function_name: str = "cosine_similarity",
        chunk_size: int = 100,
        precision_type_distance: type = np.float32,
        # memory management
        memmap: bool = True,
        memmap_path: Union[Path, str, None] = None,
        # plotting
        plot_distribution: bool = False,
        plot_top_N: Optional[int] = None,
        output_path: Optional[str] = None,
        figsize: tuple = (10, 8),
        # utils
        random_seed: int = 42,
        # logging
        log_level: str = "INFO",
        **kwargs,
    ):
        set_log_level(min_log_level=log_level)
        fix_random_seeds(seed=random_seed)
        self.memmap = memmap
        self.memmap_path = memmap_path
        self.model = None
        self.cleaner = SelfCleanCleaner(
            distance_function_path=distance_function_path,
            distance_function_name=distance_function_name,
            chunk_size=chunk_size,
            precision_type_distance=precision_type_distance,
            memmap=memmap,
            memmap_path=memmap_path,
            plot_distribution=plot_distribution,
            plot_top_N=plot_top_N,
            output_path=output_path,
            figsize=figsize,
            log_level=log_level,
            **kwargs,
        )
        self.base_transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def run_on_image_folder(
        self,
        input_path: Union[str, Path],
        epochs: int = 100,
        batch_size: int = 64,
        ssl_pre_training: bool = True,
        work_dir: Optional[str] = None,
        num_workers: int = os.cpu_count(),
        pretraining_type: PretrainingType = PretrainingType.DINO,
        hyperparameters: dict = DINO_STANDARD_HYPERPARAMETERS,
        # embedding
        n_layers: int = 1,
        apply_l2_norm: bool = True,
        # logging
        dataset_name: Optional[str] = None,
        wandb_logging: bool = False,
        wandb_project_name: str = "SelfClean",
    ):
        input_path = Path(input_path)
        if not input_path.exists():
            raise ValueError(f"Input path does not exist: {input_path}")
        dataset = ImageFolder(root=input_path)
        return self._run(
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            ssl_pre_training=ssl_pre_training,
            work_dir=work_dir,
            num_workers=num_workers,
            pretraining_type=pretraining_type,
            hyperparameters=hyperparameters,
            n_layers=n_layers,
            apply_l2_norm=apply_l2_norm,
            additional_run_info=(
                input_path.stem if dataset_name is None else dataset_name
            ),
            wandb_logging=wandb_logging,
            wandb_project_name=wandb_project_name,
        )

    def run_on_dataset(
        self,
        dataset,
        epochs: int = 100,
        batch_size: int = 64,
        ssl_pre_training: bool = True,
        work_dir: Optional[str] = None,
        num_workers: int = os.cpu_count(),
        pretraining_type: PretrainingType = PretrainingType.DINO,
        hyperparameters: dict = DINO_STANDARD_HYPERPARAMETERS,
        # embedding
        n_layers: int = 1,
        apply_l2_norm: bool = True,
        # logging
        dataset_name: Optional[str] = None,
        wandb_logging: bool = False,
        wandb_project_name: str = "SelfClean",
    ):
        return self._run(
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            ssl_pre_training=ssl_pre_training,
            work_dir=work_dir,
            num_workers=num_workers,
            pretraining_type=pretraining_type,
            hyperparameters=hyperparameters,
            n_layers=n_layers,
            apply_l2_norm=apply_l2_norm,
            additional_run_info=(
                type(dataset).__name__ if dataset_name is None else dataset_name
            ),
            wandb_logging=wandb_logging,
            wandb_project_name=wandb_project_name,
        )

    def _run(
        self,
        dataset,
        epochs: int = 100,
        batch_size: int = 64,
        ssl_pre_training: bool = True,
        work_dir: Optional[str] = None,
        num_workers: int = os.cpu_count(),
        pretraining_type: PretrainingType = PretrainingType.DINO,
        hyperparameters: dict = DINO_STANDARD_HYPERPARAMETERS,
        # embedding
        n_layers: int = 1,
        apply_l2_norm: bool = True,
        # logging
        additional_run_info: str = "",
        wandb_logging: bool = False,
        wandb_project_name: str = "SelfClean",
    ):
        if self.model is None:
            if pretraining_type is PretrainingType.DINO:
                self.model = self.train_dino(
                    dataset=dataset,
                    epochs=epochs,
                    batch_size=batch_size,
                    ssl_pre_training=ssl_pre_training,
                    work_dir=work_dir,
                    hyperparameters=hyperparameters,
                    num_workers=num_workers,
                    additional_run_info=additional_run_info,
                    wandb_logging=wandb_logging,
                    wandb_project_name=wandb_project_name,
                )
            elif (
                pretraining_type is PretrainingType.IMAGENET
                or pretraining_type is PretrainingType.IMAGENET_VIT
            ):
                self.model = Embedder.load_pretrained(pretraining_type.value)
            else:
                raise ValueError(f"Unknown pretraining type: {pretraining_type}")

        set_dataset_transformation(dataset=dataset, transform=self.base_transform)
        torch_dataset = DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=False,
            shuffle=False,
        )
        emb_space, labels = embed_dataset(
            torch_dataset=torch_dataset,
            model=self.model,
            n_layers=n_layers,
            normalize=apply_l2_norm,
            memmap=self.memmap,
            memmap_path=self.memmap_path,
            tqdm_desc="Creating dataset representation",
            return_only_embedding_and_labels=True,
        )
        # for default datasets we can set the paths manually
        paths = None
        if hasattr(dataset, "_image_files") and paths is None:
            paths = dataset._image_files

        self.cleaner.fit(
            emb_space=np.asarray(emb_space),
            labels=np.asarray(labels),
            paths=np.asarray(paths) if paths is not None else paths,
            dataset=dataset,
            class_labels=dataset.classes if hasattr(dataset, "classes") else None,
        )
        return self.cleaner.predict()

    def train_dino(
        self,
        dataset: Dataset,
        epochs: int = 100,
        batch_size: int = 64,
        ssl_pre_training: bool = True,
        work_dir: Optional[str] = None,
        hyperparameters: dict = DINO_STANDARD_HYPERPARAMETERS,
        num_workers: int = os.cpu_count(),
        # logging
        additional_run_info: str = "",
        wandb_logging: bool = False,
        wandb_project_name: str = "SelfClean",
    ):
        assert all(
            key in hyperparameters for key in DINO_STANDARD_HYPERPARAMETERS
        ), "`hyperparameters` need to contain all standard hyperparameters."

        hyperparameters["epochs"] = epochs
        hyperparameters["batch_size"] = batch_size
        hyperparameters["ssl_pre_training"] = ssl_pre_training
        if work_dir is not None:
            hyperparameters["work_dir"] = work_dir

        init_distributed_mode()
        ssl_augmentation = iBOTDataAugmentation(
            **hyperparameters["dataset"]["augmentations"]
        )
        set_dataset_transformation(dataset=dataset, transform=ssl_augmentation)
        if torch.cuda.is_available():
            sampler = DistributedSampler(dataset, shuffle=True)
            kwargs = {"sampler": sampler}
        else:
            kwargs = {"shuffle": True}
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
            **kwargs,
        )
        trainer = DINOTrainer(
            train_dataset=train_loader,
            config=hyperparameters,
            additional_run_info=additional_run_info,
            wandb_logging=wandb_logging,
            wandb_project_name=wandb_project_name,
        )
        model = trainer.fit()
        del trainer, train_loader
        gc.collect()
        cleanup()
        return model
