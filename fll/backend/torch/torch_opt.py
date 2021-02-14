from torch import optim

from .torch_opt_state import TorchOptState
from .. import AbstractOptState, AbstractOpt


class TorchOpt(AbstractOpt):
    """An implementation of AbstractOpt for Torch."""
    def __init__(self, opt: optim.Optimizer):
        self.opt = opt

    def get_state(self) -> AbstractOptState:
        return TorchOptState(self.opt.state_dict())

    def load_state(self, state: AbstractOptState):
        assert isinstance(state, TorchOptState)
        self.opt.load_state_dict(state.state)
