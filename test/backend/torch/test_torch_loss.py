from torch import nn

from fll.backend.torch import TorchLoss


class TestTorchLoss:
    def test_name(self):
        name = 'mse'
        loss = TorchLoss(nn.MSELoss(), name)
        assert loss.name == name
