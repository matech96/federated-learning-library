from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional


class AbstractBackendFactory(ABC):
    """ A factory, that puts the deep learning framework specific object into a container, that can be freely passed
    around in the rest of the code.

    .. seealso:: AbstractBackendOperations
    """

    @classmethod
    @abstractmethod
    def create_data_loader(cls, data_loader) -> AbstractDataLoader:
        pass

    @classmethod
    @abstractmethod
    def create_loss(cls, loss, name: str) -> AbstractLoss:
        pass

    @classmethod
    @abstractmethod
    def create_metric(cls, metric, name: str) -> AbstractMetric:
        pass

    @classmethod
    @abstractmethod
    def create_model(cls, model) -> AbstractModel:
        pass

    @classmethod
    @abstractmethod
    def create_model_state(cls, model_state) -> AbstractModelState:
        pass

    @classmethod
    @abstractmethod
    def create_opt(cls, opt) -> AbstractOpt:
        pass

    @classmethod
    @abstractmethod
    def create_opt_state(cls, opt_state) -> AbstractOptState:
        pass


class AbstractBackendOperations(ABC):
    @staticmethod
    @abstractmethod
    def train_epoch(model: AbstractModel, opt: AbstractOpt, loss: AbstractLoss,  # noqa: R0913
                    data: AbstractDataLoader, metrics: List[AbstractMetric]) -> Dict[str, float]:  # noqa: R0913
        """Trains the model for 1 round.

        :param model: The model to be trained.
        :param opt: The optimization algorithm.
        :param loss: The loss function.
        :param data: The training data.
        :param metrics: The metrics to be measured.
        :return: Dictionary, where the key is the name of the metric and the value is a float or int.
        """

    @staticmethod
    @abstractmethod
    def eval(model: AbstractModel, data: AbstractDataLoader, metrics: List[AbstractMetric]) -> Dict[str, float]:
        """Evaluates the model on the provided data.

        :param model: Model to be evaluated.
        :param data: Data to evaluate on. (Should include preprocessing.)
        :param metrics: Metrics of the evaluation.
        :return: Dictionary, where the key is the name of the metric and the value is a float or int.
        """

    @staticmethod
    @abstractmethod
    def cumulative_avg_model_state(state_0: Optional[AbstractModelState], state_1: AbstractModelState,
                                   n_states_0: int) -> AbstractModelState:
        """This function is useful to calculate the average of many model states, but without needing to keep all of
        them in memory. With this function you only need two model states simultaneously to calculate the average.

        :param state_0: State form previous cumulative steps. Can be None. If None, return state_1
        :param state_1: New state.
        :param n_states_0: Number of states averaged in state_0.
        :return: The uniform average of all the states in state_0 and state_1.
        """

    @staticmethod
    @abstractmethod
    def cumulative_avg_opt_state(state_0: Optional[AbstractOptState], state_1: AbstractOptState,
                                 n_states_0: int) -> AbstractOptState:
        """This function is useful to calculate the average of many optimizer states, but without needing to keep all of
        them in memory. With this function you only need two optimizer states simultaneously to calculate the average.

        :param state_0: State form previous cumulative steps. Can be None. If None, return state_1
        :param state_1: New state.
        :param n_states_0: Number of states averaged in state_0.
        :return: The uniform average of all the states in state_0 and state_1.
        """


class AbstractDataLoader(ABC):
    """Abstract class for a data loader."""

    def __init__(self, data_loader):
        self.data_loader = data_loader


class AbstractLoss(ABC):
    """Abstract class for a loss function."""

    def __init__(self, name: str):
        self.name = name


class AbstractMetric(ABC):
    """Abstract class for a metric."""

    def __init__(self, name: str):
        self.name = name


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


class AbstractModelState(ABC):
    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> AbstractModelState:
        pass

    @abstractmethod
    def save(self, path: Path):
        pass


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


class AbstractOptState(ABC):
    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> AbstractOptState:
        """Loads the state from the disk.

        :param path: The location of the state on the disk.
        :return: the loaded state
        """

    @abstractmethod
    def save(self, path: Path):
        """Saves the state to the disk.

        :param path: Path to save location.
        """
