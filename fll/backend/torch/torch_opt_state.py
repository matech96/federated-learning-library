from pathlib import Path
from typing import Dict

import torch as th

from .. import AbstractOptState
from .utils import state_dict_eq


class TorchOptState(AbstractOptState):
    """An implementation of AbstractOpt for Torch."""
    def __init__(self, state: Dict):
        self.state = state

    @classmethod
    def load(cls, path: Path) -> AbstractOptState:
        return TorchOptState(th.load(path))

    # noinspection PyTypeHints
    def __eq__(self, other: AbstractOptState) -> bool:  # type: ignore[override]
        assert isinstance(other, TorchOptState)
        return state_dict_eq(self.state, other.state)

    def save(self, path: Path):
        th.save(self.state, path)
