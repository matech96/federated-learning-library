from abc import ABC, abstractmethod

from .abstract_opt_state import AbstractOptState


class AbstractOpt(ABC):
    """Abstract class for an optimizer."""

    @abstractmethod
    def get_state(self) -> AbstractOptState:
        """Returns the state of the optimizer.

        :return: State of the optimizer
        """

    @abstractmethod
    def load_state(self, state: AbstractOptState):
        pass
