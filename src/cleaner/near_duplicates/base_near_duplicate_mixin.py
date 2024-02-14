from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseNearDuplicateMixin(ABC):
    @abstractmethod
    def get_near_duplicate_ranking(self) -> List[Tuple[float, int]]:
        raise NotImplementedError()
