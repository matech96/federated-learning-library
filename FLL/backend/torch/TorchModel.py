from torch import nn

from .. import FLLModel
from . import TorchModelState


class TorchModel(FLLModel):
    def __init__(self, model: nn.Module):
        self.model = model

    def get_state(self) -> TorchModelState:
        return TorchModelState(self.model.state_dict())

    def load_state(self, state: TorchModelState):
        self.model.load_state_dict(state.state)
