from pathlib import Path
from typing import Dict

import torch as th

from .. import AbstractModelState
from .utils import state_dict_eq


class TorchModelState(AbstractModelState):
    def __init__(self, state: Dict):
        self.state = state

    @classmethod
    def load(cls, path: Path) -> AbstractModelState:
        return TorchModelState(th.load(path))

    def __eq__(self, other: "TorchModelState"):  # type: ignore[override]
        return state_dict_eq(self.state, other.state)

    def save(self, path: Path):
        th.save(self.state, path)
