from abc import ABC, abstractmethod

from typing import List, Dict

from .backend import AbstractDataLoader


class DataSet(ABC):
    """A class containing data loaders for the training, eval and test sets."""

    def __init__(self, train_loader_list: List[AbstractDataLoader], eval_loader: AbstractDataLoader,
                 test_loader: AbstractDataLoader):
        """
        Instantiates a DataSet.
        :param train_loader_list: List of instances of AbstractDataLoader. Each loader in the list will be associated
        with a nique client.
        :param eval_loader: AbstractDataLoader for evaluation.
        :param test_loader: AbstractDataLoader for testing.
        """
        self.train_loader_list = train_loader_list
        self.eval_loader = eval_loader
        self.test_loader = test_loader

    @classmethod
    @abstractmethod
    def is_learning(cls, metrics: Dict[str, float]) -> bool:
        """
        From the metrics, it determines if the model is learning. For example in classification, if the accuracy is
        higher, than random accuracy.
        :param metrics: Dictionary, where the key is the name of the metric and the value is a float or int.
        :return: True, if the model is performing better, than random weights.
        """
