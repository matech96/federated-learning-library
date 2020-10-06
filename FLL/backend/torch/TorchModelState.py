from pathlib import Path
from typing import Dict

import torch as th

from . import FLLModelState


class TorchModelState(FLLModelState):
    def __init__(self, state: Dict):
        self.state = state

    @classmethod
    def load_state(cls, path: Path) -> FLLModelState:
        return TorchModelState(th.load(path))

    def save_state(self, path: Path):
        th.save(self.state, path)
