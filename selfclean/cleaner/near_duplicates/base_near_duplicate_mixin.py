from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class BaseNearDuplicateMixin(ABC):
    @abstractmethod
    def get_near_duplicate_ranking(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()
