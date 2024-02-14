from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


class BaseLabelErrorMixin(ABC):
    @abstractmethod
    def get_label_error_ranking(self) -> Optional[List[Tuple[float, int]]]:
        raise NotImplementedError()
