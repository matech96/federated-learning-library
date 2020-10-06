from pathlib import Path
from typing import Dict

import torch as th

from .. import FLLModelState


class TorchModelState(FLLModelState):
    def __init__(self, state: Dict):
        self.state = state

    @classmethod
    def load(cls, path: Path) -> FLLModelState:
        return TorchModelState(th.load(path))

    def __eq__(self, other):
        return all(
            [
                (sk == ok) and th.all(sv == ov)
                for (sk, sv), (ok, ov) in zip(self.state.items(), other.state.items())
            ]
        )

    def save(self, path: Path):
        th.save(self.state, path)
