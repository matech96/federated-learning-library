"""Our code supports multiple deep learning frameworks. This module describes the interfaces, that we have to create
for a framework."""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple


class AbstractBackendFactory(ABC):
    """An abstract factory, that puts the deep learning framework specific object into a container, that can be
    freely passed around in the rest of the code. Needs to be inherited for a specific framework.

    .. seealso:: :class:`~AbstractBackendOperations`
    """

    @classmethod
    @abstractmethod
    def create_data_loader(cls, data_loader) -> AbstractDataLoader:
        """Stores the data loader in an :class:`~AbstractDataLoader`.

        :param data_loader: Deep learning framework specific data loader.
        :return: Wrapped data loader.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def create_loss(cls, loss, name: str) -> AbstractLoss:
        """Stores the loss function in an :class:`~AbstractLoss`.

        :param loss: Deep learning framework specific loss function.
        :param name: Name of the loss function.
        :return: Wrapped loss function.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def create_metric(cls, metric, name: str) -> AbstractMetric:
        """Stores the metric in an :class:`~AbstractMetric`.

        :param metric: Deep learning framework specific metric.
        :param name: Name of the metric.
        :return: Wrapped metric.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def create_model(cls, model) -> AbstractModel:
        """Stores the model in an :class:`~AbstractModel`.

        :param model: Deep learning framework specific model.
        :return: Wrapped model.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def create_model_state(cls, model_state) -> AbstractModelState:
        """Stores the model state in an :class:`~AbstractModelState`.

        :param model_state: Deep learning framework specific model state.
        :return: Wrapped model state.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def create_opt(cls, opt) -> AbstractOpt:
        """Stores the optimizer in an :class:`~AbstractOpt`.

        :param opt: Deep learning framework specific optimizer.
        :return: Wrapped optimizer.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def create_opt_state(cls, opt_state) -> AbstractOptState:
        """Stores the optimizer state in an :class:`~AbstractOptState`.

        :param opt_state: Deep learning framework specific optimizer state.
        :return: Wrapped optimizer state.
        """
        raise NotImplementedError()


class AbstractBackendOperations(ABC):
    """The deep learning framework specific calculations - that can't be linked to a specific class - are collected in
    this class. Needs to be inherited for a specific framework.

    .. seealso:: :class:`~AbstractBackendFactory`
    """

    @classmethod
    @abstractmethod
    def train_epoch(cls, model: AbstractModel, opt: AbstractOpt, loss: AbstractLoss,  # noqa: R0913
                    data_loader: AbstractDataLoader, metrics: List[AbstractMetric]) -> Dict[str, float]:  # noqa: R0913
        """Trains the model for 1 round.

        :param model: The model to be trained.
        :param opt: The optimization algorithm.
        :param loss: The loss function.
        :param data_loader: The training data.
        :param metrics: The metrics to be measured.
        :return: Dictionary, where the key is the name of the metric and the value is a float or int.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def eval(cls, model: AbstractModel, data_loader: AbstractDataLoader, metrics: List[AbstractMetric]) \
            -> Dict[str, float]:
        """Evaluates the model on the provided data.

        :param model: Model to be evaluated.
        :param data_loader: Data to evaluate on. (Should include preprocessing.)
        :param metrics: Metrics of the evaluation.
        :return: Dictionary, where the key is the name of the metric and the value is a float or int.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def cumulative_avg_model_state(cls, state_0: Optional[AbstractModelState], state_1: AbstractModelState,
                                   n_states_0: int) -> AbstractModelState:
        """This function is useful to calculate the average of many model states, but without needing to keep all of
        them in memory. With this function you only need two model states simultaneously to calculate the average.

        :param state_0: State form previous cumulative steps. Can be None. If None, return state_1
        :param state_1: New state.
        :param n_states_0: Number of states averaged in state_0.
        :return: The uniform average of all the states in state_0 and state_1.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def cumulative_avg_opt_state(cls, state_0: Optional[AbstractOptState], state_1: AbstractOptState,
                                 n_states_0: int) -> AbstractOptState:
        """This function is useful to calculate the average of many optimizer states, but without needing to keep all of
        them in memory. With this function you only need two optimizer states simultaneously to calculate the average.

        :param state_0: State form previous cumulative steps. Can be None. If None, return state_1
        :param state_1: New state.
        :param n_states_0: Number of states averaged in state_0.
        :return: The uniform average of all the states in state_0 and state_1.
        """
        raise NotImplementedError()


class AbstractModelOptFactory(ABC):
    """Abstract class for creating a framework specific model and optimizer."""
    def __init__(self, model_cls: Callable, opt_cls: Callable):
        """

        :param model_cls: Callable, that returns the framework specific model.
        :param opt_cls: Callable, that returns the framework specific optimizer.
        """
        self.model_cls = model_cls
        self.opt_cls = opt_cls

    @abstractmethod
    def make_objects(self) -> Tuple[AbstractModel, AbstractOpt]:
        """

        :return: The created model and optimizer in a tuple.
        """
        raise NotImplementedError()


class AbstractDataLoader(ABC):
    """Abstract class for containing a framework specific data loader. A data loader can yield a batch of data for
    training or inference. In federated learning, each clients has its own data loader. Needs to be inherited for a
    specific framework.

    .. seealso:: :class:`~AbstractBackendFactory`
    """


class AbstractLoss(ABC):
    """Abstract class for containing a framework specific, callable loss function and its name. Needs to be inherited
    for a specific framework.

    .. seealso:: :class:`~AbstractBackendFactory`
    """

    def __init__(self, name: str):
        """

        :param name: Name of the loss function.
        """
        self.name = name


class AbstractMetric(ABC):
    """Abstract class for containing a framework specific, callable metric function. Needs to be inherited for a
    specific framework.

    .. seealso:: :class:`~AbstractBackendFactory`
    """

    def __init__(self, name: str):
        """

        :param name: Name of the metric.
        """
        self.name = name


class AbstractModel(ABC):
    """Abstract class for containing a framework specific a model. Needs to be inherited for a specific framework.

    .. seealso:: :class:`~AbstractBackendFactory`
    """

    @abstractmethod
    def get_state(self) -> AbstractModelState:
        """Returns the state of the model.

        :return: State of the model.
        """

    @abstractmethod
    def load_state(self, state: AbstractModelState):
        """Loads the state of the model.

        :param state: State of the model.
        """


class AbstractModelState(ABC):
    """Abstract class for containing a framework specific model state. The model state is a snapshot of the model
    taken during it's training. The model state doesn't include the optimizer. Needs to be inherited for a specific
    framework.

    .. seealso:: :class:`~AbstractBackendFactory` :class:`~AbstractOptState`
    """

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> AbstractModelState:
        """Loads the model state from a file.

        :param path: Path to the file.
        :return: A new object with the loaded model state.
        """

    @abstractmethod
    def save(self, path: Path):
        """Save to a file.

        :param path: Path to the file.
        """


class AbstractOpt(ABC):
    """Abstract class for containing a framework specific optimizer. Needs to be inherited for a specific framework.

    .. seealso:: :class:`~AbstractBackendFactory`
    """

    @abstractmethod
    def get_state(self) -> AbstractOptState:
        """Returns the state of the optimizer.

        :return: State of the optimizer.
        """

    @abstractmethod
    def load_state(self, state: AbstractOptState):
        """Loads the state of the optimizer.

        :return: State of the optimizer.
        """


class AbstractOptState(ABC):
    """Abstract class for containing a framework specific optimizer state. The optimizer state is a snapshot of the
    optimizer taken during training. In the case of momentum optimizer, the optimizer state is the momentum value.
    Needs to be inherited for a specific framework.

    .. seealso:: :class:`~AbstractBackendFactory` :class:`~AbstractModelState`
    """

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> AbstractOptState:
        """Loads the state from the disk.

        :param path: Path to the file.
        :return: the loaded state
        """

    @abstractmethod
    def save(self, path: Path):
        """Saves the state to the disk.

        :param path: Path to the file.
        """
