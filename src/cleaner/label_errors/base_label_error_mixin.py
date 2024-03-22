from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class BaseLabelErrorMixin(ABC):
    @abstractmethod
    def get_label_error_ranking(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        raise NotImplementedError()
