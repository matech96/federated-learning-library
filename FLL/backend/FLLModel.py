from abc import ABC, abstractmethod

from . import FLLModelState


class FLLModel(ABC):
    @abstractmethod
    def get_state(self) -> FLLModelState:
        pass

    @abstractmethod
    def load_state(cls, state: FLLModelState):
        pass
