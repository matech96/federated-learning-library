from torch import nn

from fll.backend.torch import TorchMetric


class TestTorchMetric:
    def test_name(self):
        name = 'mse'
        metric = TorchMetric(nn.MSELoss(), name)
        assert metric.name == name
