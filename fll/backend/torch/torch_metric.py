from typing import Callable
import torch as th

from .. import AbstractMetric


class TorchMetric(AbstractMetric):
    """An implementation of AbstractMetric for Torch."""
    def __init__(self, metric: Callable[[th.Tensor, th.Tensor], th.Tensor]):
        self.metric = metric
