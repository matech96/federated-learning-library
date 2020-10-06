from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class FLLModelState(ABC):
    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> FLLModelState:
        pass

    @abstractmethod
    def save(self, path: Path):
        pass
