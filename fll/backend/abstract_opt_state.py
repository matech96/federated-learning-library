from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class AbstractOptState(ABC):
    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> AbstractOptState:
        """Loads the state from the disk.

        :param path: The location of the state on the disk.
        :return: the loaded state
        """

    @abstractmethod
    def save(self, path: Path):
        """Saves the state to the disk.

        :param path: Path to save location.
        """
