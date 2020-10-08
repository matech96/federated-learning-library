from torch import nn

from .. import FLLModel, FLLModelState
from . import TorchModelState


class TorchModel(FLLModel):
    def __init__(self, model: nn.Module):
        self.model = model

    def get_state(self) -> FLLModelState:
        return TorchModelState(self.model.state_dict())

    def load_state(self, state: FLLModelState):
        if not isinstance(state, TorchModelState):
            raise ValueError()
        self.model.load_state_dict(state.state)
