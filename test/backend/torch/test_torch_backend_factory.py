import torch as th
from torch import nn

from fll.backend.torch import TorchBackendFactory
from fll.backend.torch.utils import tensor_dict_eq

from . import util


class TestTorchBackendFactory:
    def test_loss(self):
        mse = nn.MSELoss()
        name = "mse"
        a = th.tensor([1, 2, 3], dtype=th.float)
        b = th.tensor([4, 5, 6], dtype=th.float)

        loss = TorchBackendFactory.create_loss(mse, name)
        assert mse(a, b) == loss.loss(a, b)
        assert name == loss.name

    def test_metric(self):
        mse = nn.MSELoss()
        name = "mse"
        a = th.tensor([1, 2, 3], dtype=th.float)
        b = th.tensor([4, 5, 6], dtype=th.float)

        metric = TorchBackendFactory.create_metric(mse, name)
        assert mse(a, b) == metric.metric(a, b)
        assert name == metric.name

    def test_model(self):
        model = util.SimpleModel()
        a = th.tensor([1, 2, 3], dtype=th.float)
        warped_model = TorchBackendFactory.create_model(model)
        assert model(a) == warped_model.model(a)

    def test_model_state(self):
        model = util.SimpleModel()
        state = model.state_dict()
        warped_state = TorchBackendFactory.create_model_state(state)
        assert tensor_dict_eq(state, warped_state.state)

    def test_opt(self):
        provider = util.SimpleProvider()

        opt = TorchBackendFactory.create_opt(provider.opt)
        provider.train_one_iteration()

        assert provider.opt == opt.opt

    def test_opt_state(self):
        provider = util.SimpleProvider()

        opt_state_warp = TorchBackendFactory.create_opt_state(provider.opt.state_dict())
        opt_warp_state = TorchBackendFactory.create_opt(provider.opt).get_state()

        assert opt_warp_state == opt_state_warp
