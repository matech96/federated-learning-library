from abc import ABC, abstractmethod

from .abstract_data_loader import AbstractDataLoader
from .abstract_loss import AbstractLoss
from .abstract_metric import AbstractMetric
from .abstract_model import AbstractModel
from .abstract_model_state import AbstractModelState
from .abstract_opt import AbstractOpt
from .abstract_opt_state import AbstractOptState


class AbstractBackendFactory(ABC):
    """ A factory, that puts the deep learning framework specific object into a container, that can be freely passed around in the rest of the code.

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
