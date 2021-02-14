from abc import ABC, abstractmethod

from .abstract_model_state import AbstractModelState


class AbstractModel(ABC):
    """Abstract class for a model."""

    @abstractmethod
    def get_state(self) -> AbstractModelState:
        """Returns the state of the model.

        :return: State of the model
        """

    @abstractmethod
    def load_state(self, state: AbstractModelState):
        pass
