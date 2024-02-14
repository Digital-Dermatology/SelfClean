from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseIrrelevantMixin(ABC):
    @abstractmethod
    def get_irrelevant_ranking(self) -> List[Tuple[float, int]]:
        raise NotImplementedError()
