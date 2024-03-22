from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from torch.utils.data import Dataset


class BaseCleaner(ABC):
    @abstractmethod
    def fit(
        self,
        emb_space: np.ndarray,
        labels: Optional[np.ndarray] = None,
        paths: Optional[np.ndarray] = None,
        dataset: Optional[Dataset] = None,
        class_labels: Optional[list] = None,
    ):
        raise NotImplementedError()

    @abstractmethod
    def predict(self) -> dict:
        raise NotImplementedError()
