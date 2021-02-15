from typing import Callable

import torch as th

from .. import AbstractLoss


class TorchLoss(AbstractLoss):
    """An implementation of AbstractLoss for Torch."""

    def __init__(self, loss: Callable[[th.Tensor, th.Tensor], th.Tensor], name: str):
        super().__init__(name)
        self.loss = loss
