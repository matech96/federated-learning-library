from torch import optim

from fll import ModelOptStateManager, ModelOptState
from fll.backend.torchbackend import TorchModelOptFactory, TorchBackendOperations
from test.backend.test_torch.test_torch_util import SignModelComplex, SignProvider


class TestModelOptStateManager:
    def test_model_exists(self):
        factory = TorchModelOptFactory(SignModelComplex, optim.Adam)
        manager = ModelOptStateManager(False, factory)
        assert manager.model is None
        assert manager.opt is None
        with manager:
            assert manager.model is not None
            assert manager.opt is not None
        assert manager.model is None
        assert manager.opt is None

        manager = ModelOptStateManager(True, factory)
        assert manager.model is None
        assert manager.opt is None
        with manager:
            assert manager.model is not None
            assert manager.opt is not None
        assert manager.model is not None
        assert manager.opt is not None

    def test_state(self):
        prov = SignProvider()
        factory = TorchModelOptFactory(SignModelComplex, optim.Adam)
        manager = ModelOptStateManager(False, factory)
        with manager:
            TorchBackendOperations.train_epoch(manager.model, manager.opt, prov.loss, prov.dl, [])
            model_state = manager.model.get_state()
            opt_state = manager.opt.get_state()
        with manager:
            model_state2 = manager.model.get_state()
            opt_state2 = manager.opt.get_state()
            assert model_state != model_state2
            assert opt_state != opt_state2
        state = ModelOptState(model_state, opt_state)
        manager.set_state(state)
        with manager:
            model_state2 = manager.model.get_state()
            opt_state2 = manager.opt.get_state()
            assert model_state == model_state2
            assert opt_state == opt_state2

        manager = ModelOptStateManager(True, factory)
        with manager:
            TorchBackendOperations.train_epoch(manager.model, manager.opt, prov.loss, prov.dl, [])
            model_state = manager.model.get_state()
            opt_state = manager.opt.get_state()
        with manager:
            model_state2 = manager.model.get_state()
            opt_state2 = manager.opt.get_state()
            assert model_state == model_state2
            assert opt_state == opt_state2
            TorchBackendOperations.train_epoch(manager.model, manager.opt, prov.loss, prov.dl, [])
            model_state2 = manager.model.get_state()
            opt_state2 = manager.opt.get_state()
            assert model_state != model_state2
            assert opt_state != opt_state2
        state = ModelOptState(model_state, opt_state)
        manager.set_state(state)
        with manager:
            model_state2 = manager.model.get_state()
            opt_state2 = manager.opt.get_state()
            assert model_state == model_state2
            assert opt_state == opt_state2
