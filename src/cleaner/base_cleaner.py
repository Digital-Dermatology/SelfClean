from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseCleaner(ABC):
    @abstractmethod
    def fit(
        self,
        emb_space: np.ndarray,
        labels: Optional[np.ndarray] = None,
        images: Optional[np.ndarray] = None,
        paths: Optional[np.ndarray] = None,
        class_labels: Optional[list] = None,
    ):
        raise NotImplementedError()

    @abstractmethod
    def predict(self):
        raise NotImplementedError()
