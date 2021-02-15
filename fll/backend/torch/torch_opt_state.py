from pathlib import Path
from typing import Dict

import torch as th

from .. import AbstractOptState
from .utils import tensor_dict_eq


class TorchOptState(AbstractOptState):
    """An implementation of AbstractOpt for Torch."""
    def __init__(self, state: Dict):
        self.state = state

    @classmethod
    def load(cls, path: Path) -> AbstractOptState:
        return TorchOptState(th.load(path))

    def __eq__(self, other) -> bool:
        assert isinstance(other, TorchOptState)
        return tensor_dict_eq(self.state, other.state)

    def save(self, path: Path):
        th.save(self.state, path)
