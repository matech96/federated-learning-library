from pathlib import Path

import torch as th

from .. import AbstractModelState
from .utils import state_dict_eq


class TorchModelState(AbstractModelState):
    """An implementation of AbstractModelState for Torch."""
    def __init__(self, state: dict):
        self.state = state

    @classmethod
    def load(cls, path: Path) -> AbstractModelState:
        return TorchModelState(th.load(path))

    # noinspection PyTypeHints
    def __eq__(self, other: AbstractModelState):  # type: ignore[override]
        assert isinstance(other, TorchModelState)
        return state_dict_eq(self.state, other.state)

    def save(self, path: Path):
        th.save(self.state, path)
