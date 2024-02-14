from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseLabelErrorMixin(ABC):
    @abstractmethod
    def get_label_error_ranking(self) -> List[Tuple[float, int]]:
        raise NotImplementedError()
