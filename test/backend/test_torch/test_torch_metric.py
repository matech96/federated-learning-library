from torch import nn

from fll.backend.torchbackend import TorchMetric


class TestTorchMetric:
    def test_name(self):
        name = 'mse'
        metric = TorchMetric(nn.MSELoss(), name)
        assert metric.name == name
