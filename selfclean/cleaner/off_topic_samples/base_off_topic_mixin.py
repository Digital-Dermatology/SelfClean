from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class BaseOffTopicMixin(ABC):
    @abstractmethod
    def get_off_topic_ranking(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()
