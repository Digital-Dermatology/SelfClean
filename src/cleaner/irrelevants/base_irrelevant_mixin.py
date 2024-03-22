from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class BaseIrrelevantMixin(ABC):
    @abstractmethod
    def get_irrelevant_ranking(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()
