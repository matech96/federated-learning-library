from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class AbstractModelState(ABC):
    # pytype: disable=bad-return-type
    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> AbstractModelState:
        pass

    @abstractmethod
    def save(self, path: Path):
        pass

    # pytype: enable=bad-return-type
