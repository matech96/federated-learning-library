from pathlib import Path

import torch as th

from .. import AbstractModelState
from .utils import tensor_dict_eq


class TorchModelState(AbstractModelState):
    """An implementation of AbstractModelState for Torch."""
    def __init__(self, state: dict):
        self.state = state

    @classmethod
    def load(cls, path: Path) -> AbstractModelState:
        return TorchModelState(th.load(path))

    def __eq__(self, other):
        assert isinstance(other, TorchModelState)
        return tensor_dict_eq(self.state, other.state)

    def save(self, path: Path):
        th.save(self.state, path)
