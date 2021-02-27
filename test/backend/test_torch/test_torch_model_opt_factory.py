from typing import Callable

import torch as th
from torch import optim, nn

from fll.backend.torchbackend import TorchModelOptFactory, TorchBackendOperations, TorchBackendFactory

from .test_torch_util import SignModelComplex, SignDataSet


class TestTorchModelOptFactory:
    def test_learning(self):
        model_opt_factory = TorchModelOptFactory(SignModelComplex, optim.Adam)
        ds = SignDataSet()
        dl = th.utils.data.DataLoader(ds, batch_size=2)
        dl = TorchBackendFactory.create_data_loader(dl)
        model, opt = model_opt_factory.make_objects()
        loss = TorchBackendFactory.create_loss(th.nn.BCEWithLogitsLoss(), 'bce')

        prev_loss = None
        for _ in range(10):
            res = TorchBackendOperations.train_epoch(model=model, opt=opt, loss=loss, data_loader=dl, metrics=[])
            loss_value = res['loss']
            if prev_loss is not None:
                assert loss_value < prev_loss
            prev_loss = loss_value
