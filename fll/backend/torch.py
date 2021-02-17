from __future__ import annotations
from pathlib import Path
from typing import Callable, Dict, Union, Any, cast

import torch as th
from torch import nn, optim

from fll.backend.abstract import AbstractBackendFactory, AbstractDataLoader, AbstractLoss, AbstractMetric, \
    AbstractModel, AbstractModelState, AbstractOptState, AbstractOpt


class TorchBackendFactory(AbstractBackendFactory):
    """Puts the PyTorch specific object into a container, that can be freely passed around in the rest of the code.

    """

    @classmethod
    def create_data_loader(cls, data_loader: th.utils.data.DataLoader) -> TorchDataLoader:
        """Stores the data loader in an :class:`~TorchDataLoader`.

        :param data_loader: Deep learning framework specific data loader.
        :return: Wrapped data loader.
        """
        return TorchDataLoader(data_loader)

    @classmethod
    def create_loss(cls, loss: Callable[[th.Tensor, th.Tensor], th.Tensor], name: str) -> TorchLoss:
        """Stores the loss function in an :class:`~TorchLoss`.

        >>> import torch as th
        >>> from torch import nn
        >>> from fll.backend.torch import TorchBackendFactory
        >>> l = TorchBackendFactory.create_loss(nn.MSELoss(), "mse")
        >>> l.loss(th.tensor([1.0]), th.tensor([1.0]))
        tensor(0.)

        :param loss: PyTorch loss function.
        :param name: Name of the loss function.
        :return: Wrapped loss function.
        """
        return TorchLoss(loss, name)

    @classmethod
    def create_metric(cls, metric: Callable[[th.Tensor, th.Tensor], th.Tensor], name: str) -> TorchMetric:
        """Stores the metric in an :class:`~TorchMetric`.

        :param metric: Metric, created using PyTorch.
        :param name: Name of the metric.
        :return: Wrapped metric.
        """
        return TorchMetric(metric, name)

    @classmethod
    def create_model(cls, model: th.nn.Module) -> TorchModel:
        """Stores the model in an :class:`~TorchModel`.

        :param model: Deep learning framework specific model.
        :return: Wrapped model.
        """
        return TorchModel(model)

    @classmethod
    def create_model_state(cls, model_state: dict) -> TorchModelState:
        """Stores the model state in an :class:`~TorchModelState`.

        :param model_state: Deep learning framework specific model state.
        :return: Wrapped model state.
        """
        return TorchModelState(model_state)

    @classmethod
    def create_opt(cls, opt: th.optim.Optimizer) -> TorchOpt:
        """Stores the optimizer in an :class:`~TorchOpt`.

        :param opt: Deep learning framework specific optimizer.
        :return: Wrapped optimizer.
        """
        return TorchOpt(opt)

    @classmethod
    def create_opt_state(cls, opt_state: dict) -> TorchOptState:
        """Stores the optimizer state in an :class:`~TorchOptState`.

        :param opt_state: Deep learning framework specific optimizer state.
        :return: Wrapped optimizer state.
        """
        return TorchOptState(opt_state)


class TorchDataLoader(AbstractDataLoader):
    """An implementation of AbstractDataLoader for Torch."""

    def __init__(self, data_loader: th.utils.data.DataLoader):
        assert isinstance(data_loader, th.utils.data.DataLoader)
        self.data_loader = data_loader


class TorchLoss(AbstractLoss):
    """An implementation of AbstractLoss for Torch."""

    def __init__(self, loss: Callable[[th.Tensor, th.Tensor], th.Tensor], name: str):
        super().__init__(name)
        self.loss = loss


class TorchMetric(AbstractMetric):
    """An implementation of AbstractMetric for Torch."""

    def __init__(self, metric: Callable[[th.Tensor, th.Tensor], th.Tensor], name: str):
        super().__init__(name)
        self.metric = metric


class TorchModel(AbstractModel):
    """An implementation of AbstractModel for Torch."""

    def __init__(self, model: nn.Module):
        assert isinstance(model, nn.Module)
        self.model = model

    def get_state(self) -> AbstractModelState:
        return TorchModelState(self.model.state_dict())

    def load_state(self, state: AbstractModelState):
        assert isinstance(state, TorchModelState)
        self.model.load_state_dict(state.state)


class TorchModelState(AbstractModelState):
    """An implementation of AbstractModelState for Torch."""

    def __init__(self, state: dict):
        self.state = state

    @classmethod
    def load(cls, path: Path) -> AbstractModelState:
        return TorchModelState(th.load(path))

    def __eq__(self, other):
        assert isinstance(other, TorchModelState)
        return tensor_dict_eq(self.state, other.state)

    def save(self, path: Path):
        th.save(self.state, path)


class TorchOpt(AbstractOpt):
    """An implementation of AbstractOpt for Torch."""

    def __init__(self, opt: optim.Optimizer):
        self.opt = opt

    def get_state(self) -> AbstractOptState:
        return TorchOptState(self.opt.state_dict())

    def load_state(self, state: AbstractOptState):
        assert isinstance(state, TorchOptState)
        self.opt.load_state_dict(state.state)


class TorchOptState(AbstractOptState):
    """An implementation of AbstractOpt for Torch."""

    def __init__(self, state: Dict):
        self.state = state

    @classmethod
    def load(cls, path: Path) -> AbstractOptState:
        return TorchOptState(th.load(path))

    def __eq__(self, other) -> bool:
        assert isinstance(other, TorchOptState)
        return tensor_dict_eq(self.state, other.state)

    def save(self, path: Path):
        th.save(self.state, path)


def tensor_dict_eq(dict1: dict, dict2: dict) -> bool:
    """Checks the equivalence between 2 dictionaries, that can contain torch Tensors as value. The dictionary can be
    nested with other dictionaries or lists, they will be checked recursively.

    :param dict1: Dictionary to compare.
    :param dict2: Dictionary to compare.
    :return: True, if dict1 and dict2 are equal, false otherwise.
    """
    if len(dict1) != len(dict2):
        return False

    for (key1, value1), (key2, value2) in zip(dict1.items(), dict2.items()):
        key_equal = key1 == key2
        value_equal = tensor_container_element_eq(value1, value2)

        if (not key_equal) or (not value_equal):
            return False

    return True


def tensor_list_eq(list1: list, list2: list) -> bool:
    """Checks the equivalence between 2 lists, that can contain torch Tensors as value. The list can be nested with
    other dictionaries and lists, they will be checked recursively. The dictionaries can have torch Tensors as value,
    not as key!

    :param list1: List to compare.
    :param list2: List to compare.
    :return: True, if list1 and list2 are equal, false otherwise.
    """
    if len(list1) != len(list2):
        return False

    for value1, value2 in zip(list1, list2):
        value_equal = tensor_container_element_eq(value1, value2)
        if not value_equal:
            return False

    return True


def tensor_container_element_eq(value1: Union[dict, list, th.Tensor, Any],
                                value2: Union[dict, list, th.Tensor, Any]) -> bool:
    """ Checks equivalence between the two values and returns a single bool value, if the input is torch Tensor. If
    the input is a dictionary or list, it is recursively checked. The key of a (nested) dictionary can't be Tensor.

    :param value1: Value to compare. Can be dictionary, list, tensor or some other type that has the equal operator (==)
    return a single bool value.
    :param value2: Value to compare. Can be dictionary, list, tensor or some other type that
    has the equal operator (==) return a single bool value.
    :return: True, if v1 and v2 are equal, false otherwise.
    """
    if isinstance(value1, dict) and isinstance(value2, dict):
        return tensor_dict_eq(value1, value2)
    if isinstance(value1, list) and isinstance(value2, list):
        return tensor_list_eq(value1, value2)
    if isinstance(value1, th.Tensor) and isinstance(value2, th.Tensor):
        is_equal = th.all(th.tensor(value1 == value2)).item()
        return cast(bool, is_equal)

    return value1 == value2
