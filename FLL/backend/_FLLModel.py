from abc import ABC, abstractmethod

from . import FLLModelState


class FLLModel(ABC):
    # pytype: disable=bad-return-type
    @abstractmethod
    def get_state(self) -> FLLModelState:
        pass

    @abstractmethod
    def load_state(self, state: FLLModelState):
        pass

    # pytype: enable=bad-return-type
