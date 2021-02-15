from typing import Callable

import torch as th

from .. import AbstractBackendFactory
from .torch_data_loader import TorchDataLoader
from .torch_loss import TorchLoss
from .torch_metric import TorchMetric
from .torch_model import TorchModel
from .torch_model_state import TorchModelState
from .torch_opt import TorchOpt
from .torch_opt_state import TorchOptState


class TorchBackendFactory(AbstractBackendFactory):
    @classmethod
    def create_data_loader(cls, data_loader: th.utils.data.DataLoader) -> TorchDataLoader:
        return TorchDataLoader(data_loader)

    @classmethod
    def create_loss(cls, loss: Callable[[th.Tensor, th.Tensor], th.Tensor], name: str) -> TorchLoss:
        return TorchLoss(loss, name)

    @classmethod
    def create_metric(cls, metric: Callable[[th.Tensor, th.Tensor], th.Tensor], name: str) -> TorchMetric:
        return TorchMetric(metric, name)

    @classmethod
    def create_model(cls, model: th.nn.Module) -> TorchModel:
        return TorchModel(model)

    @classmethod
    def create_model_state(cls, model_state: dict) -> TorchModelState:
        return TorchModelState(model_state)

    @classmethod
    def create_opt(cls, opt: th.optim.Optimizer) -> TorchOpt:
        return TorchOpt(opt)

    @classmethod
    def create_opt_state(cls, opt_state: dict) -> TorchOptState:
        return TorchOptState(opt_state)
