from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class AbstractOptState(ABC):
    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> AbstractOptState:
        """Loads the state from the disk.

        Args:
            path (Path): The location of the state on the disk.

        Returns:
            AbstractOptState: the loaded state
        """

    @abstractmethod
    def save(self, path: Path):
        """Saves the state to the disk.

        Args:
            path (Path): Path to save location.
        """
