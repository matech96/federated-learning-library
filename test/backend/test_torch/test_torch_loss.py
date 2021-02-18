from torch import nn

from fll.backend.torchbackend import TorchLoss


class TestTorchLoss:
    def test_name(self):
        name = 'mse'
        loss = TorchLoss(nn.MSELoss(), name)
        assert loss.name == name
