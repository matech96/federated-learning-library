from abc import ABC, abstractmethod

from typing import Dict

from ..backend import AbstractModel, AbstractOpt, AbstractLoss, AbstractDataLoader, AbstractMetric


class AbstractBackendOperations(ABC):
    @classmethod
    @abstractmethod
    def train_epoch(cls, model: AbstractModel, opt: AbstractOpt, loss: AbstractLoss,
              data: AbstractDataLoader, metrics: AbstractMetric) -> Dict[str, float]:
        """
        Trains the model for 1 round.
        :param model: The model to be trained.
        :param opt: The optimization algorithm.
        :param loss: The loss function.
        :param data: The training data.
        :param metrics: The metrics to be measured.
        :return: A dictionary containing the names of the metrics an
        """
