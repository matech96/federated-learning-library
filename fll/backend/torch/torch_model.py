from torch import nn

from .torch_model_state import TorchModelState
from .. import AbstractModel, AbstractModelState


class TorchModel(AbstractModel):
    def __init__(self, model: nn.Module):
        self.model = model

    def get_state(self) -> AbstractModelState:
        return TorchModelState(self.model.state_dict())

    def load_state(self, state: AbstractModelState):
        if not isinstance(state, TorchModelState):
            raise ValueError()
        self.model.load_state_dict(state.state)
