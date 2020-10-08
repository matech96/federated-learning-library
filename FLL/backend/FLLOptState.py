from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class FLLOptState(ABC):
    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> FLLOptState:
        pass

    @abstractmethod
    def save(self, path: Path):
        pass
