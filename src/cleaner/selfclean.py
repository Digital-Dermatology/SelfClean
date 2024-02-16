import gc
from enum import Enum
from pathlib import Path
from typing import Union

from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode

from src.cleaner.selfclean_cleaner import SelfCleanCleaner
from ssl_library.src.augmentations.ibot import iBOTDataAugmentation
from ssl_library.src.pkg import Embedder, embed_dataset
from ssl_library.src.trainers.dino_trainer import DINOTrainer
from ssl_library.src.utils.utils import cleanup, init_distributed_mode

DINO_STANDARD_HYPERPARAMETERS = {
    "batch_size": 4,
    "epochs": 50,
    "optim": "adamw",
    "lr": 0.0005,
    "min_lr": "1e-6",
    "weight_decay": 0.04,
    "weight_decay_end": 0.4,
    "warmup_epochs": 10,
    "momentum_teacher": 0.996,
    "clip_grad": 3.0,
    "use_lr_scheduler": True,
    "use_wd_scheduler": True,
    "seed": 42,
    "fine_tune_from": "None",
    "save_every_n_epochs": 50,
    "embed_vis_every_n_epochs": 50,
    "visualize_attention": True,
    "imgs_to_visualize": 5,
    "apply_l2_norm": False,
    "model": {
        "out_dim": 4096,
        "emb_dim": 192,
        "base_model": "vit_tiny",
        "model_type": "VIT",
        "use_bn_in_head": False,
        "norm_last_layer": True,
        "student": {
            "weights": "IMAGENET1K_V1",
            "patch_size": 16,
            "drop_path_rate": 0.1,
        },
        "teacher": {"weights": "IMAGENET1K_V1", "drop_path_rate": 0.1},
        "eval": {"n_last_blocks": 4, "avgpool_patchtokens": False},
    },
    "dataset": {
        "augmentations": {
            "global_crops_scale": "(0.7, 1.)",
            "local_crops_scale": "(0.05, 0.4)",
            "global_crops_number": 2,
            "local_crops_number": 2,
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
    IMAGENET_VIT = "imagenet_vit"
    DINO = "dino"


class SelfClean:
    def __init__(
        self,
    ):
        self.cleaner = None
        self.model = None
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
        batch_size: int = 32,
        num_workers: int = 48,
        drop_last: bool = False,
        pin_memory: bool = True,
        pretraining_type: PretrainingType = PretrainingType.DINO,
        # embedding
        n_layers: int = 1,
        apply_l2_norm: bool = True,
        memmap: bool = False,
        memmap_path: Union[Path, str, None] = None,
        # logging
        wandb_logging: bool = False,
        wandb_project_name: str = "SelfClean",
    ):
        input_path = Path(input_path)
        if not input_path.exists():
            raise ValueError(f"Input path does not exist: {input_path}")
        dataset = ImageFolder(root=input_path)
        return self._run(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
            pretraining_type=pretraining_type,
            n_layers=n_layers,
            apply_l2_norm=apply_l2_norm,
            memmap=memmap,
            memmap_path=memmap_path,
            additional_run_info=input_path.stem,
            wandb_logging=wandb_logging,
            wandb_project_name=wandb_project_name,
        )

    def run_on_dataset(
        self,
        dataset,
        batch_size: int = 32,
        num_workers: int = 48,
        drop_last: bool = False,
        pin_memory: bool = True,
        pretraining_type: PretrainingType = PretrainingType.DINO,
        # embedding
        n_layers: int = 1,
        apply_l2_norm: bool = True,
        memmap: bool = False,
        memmap_path: Union[Path, str, None] = None,
        # logging
        wandb_logging: bool = False,
        wandb_project_name: str = "SelfClean",
    ):
        return self._run(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
            pretraining_type=pretraining_type,
            n_layers=n_layers,
            apply_l2_norm=apply_l2_norm,
            memmap=memmap,
            memmap_path=memmap_path,
            additional_run_info=type(dataset).__name__,
            wandb_logging=wandb_logging,
            wandb_project_name=wandb_project_name,
        )

    def _run(
        self,
        dataset,
        batch_size: int = 32,
        num_workers: int = 48,
        drop_last: bool = False,
        pin_memory: bool = True,
        pretraining_type: PretrainingType = PretrainingType.DINO,
        # embedding
        n_layers: int = 1,
        apply_l2_norm: bool = True,
        memmap: bool = False,
        memmap_path: Union[Path, str, None] = None,
        # logging
        additional_run_info: str = "",
        wandb_logging: bool = False,
        wandb_project_name: str = "SelfClean",
    ):
        if self.model is None:
            if pretraining_type is PretrainingType.DINO:
                self.model = self.train_dino(
                    dataset=dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    drop_last=drop_last,
                    pin_memory=pin_memory,
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

        dataset.transform = self.base_transform
        torch_dataset = DataLoader(
            dataset,
            batch_size=128,
            drop_last=False,
            shuffle=False,
        )
        emb_space, labels, images, paths = embed_dataset(
            torch_dataset=torch_dataset,
            model=self.model,
            n_layers=n_layers,
            normalize=apply_l2_norm,
            memmap=memmap,
            memmap_path=memmap_path,
        )

        self.cleaner = SelfCleanCleaner(
            memmap=memmap,
            memmap_path=memmap_path,
        )
        self.cleaner.fit(
            emb_space=emb_space,
            images=images,
            labels=labels,
            paths=paths,
            class_labels=dataset.classes if hasattr(dataset, "classes") else None,
        )
        return self.cleaner.predict()

    def train_dino(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        num_workers: int = 48,
        drop_last: bool = False,
        pin_memory: bool = True,
        # logging
        additional_run_info: str = "",
        wandb_logging: bool = False,
        wandb_project_name: str = "SelfClean",
    ):
        init_distributed_mode()
        dataset.transform = iBOTDataAugmentation(
            **DINO_STANDARD_HYPERPARAMETERS["dataset"]["augmentations"]
        )
        sampler = DistributedSampler(dataset, shuffle=True)
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
        )
        trainer = DINOTrainer(
            train_dataset=train_loader,
            config=DINO_STANDARD_HYPERPARAMETERS,
            additional_run_info=additional_run_info,
            wandb_logging=wandb_logging,
            wandb_project_name=wandb_project_name,
        )
        model = trainer.fit()
        del trainer
        gc.collect()
        cleanup()
        return model