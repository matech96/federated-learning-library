from copy import deepcopy
import torch as th

from fll.backend.torchbackend import TorchOptState

from . import util


class TestTorchOptState:
    def test_save_load_diff_hyperparameter(self, tmp_path):
        m = util.SimpleModel()
        o1 = th.optim.Adam(params=m.parameters())
        state1 = TorchOptState(deepcopy(o1.state_dict()))
        path1 = tmp_path / "opt1.pt"
        state1.save(path1)

        o2 = th.optim.Adam(params=m.parameters(), lr=1)
        state2 = TorchOptState(deepcopy(o2.state_dict()))
        path2 = tmp_path / "opt2.pt"
        state2.save(path2)

        loaded_state1 = TorchOptState.load(path1)
        loaded_state2 = TorchOptState.load(path2)

        assert state1 == loaded_state1
        assert state1 != loaded_state2
        assert state2 != loaded_state1

    def test_save_load_training(self, tmp_path):
        provider = util.SimpleProvider()

        state1 = TorchOptState(deepcopy(provider.opt.state_dict()))
        path1 = tmp_path / "opt1.pt"
        state1.save(path1)

        provider.train_one_iteration()

        state2 = TorchOptState(deepcopy(provider.opt.state_dict()))
        path2 = tmp_path / "opt2.pt"
        state2.save(path2)

        provider.train_one_iteration()
        state3 = TorchOptState(deepcopy(provider.opt.state_dict()))
        path3 = tmp_path / "opt3.pt"
        state3.save(path3)

        loaded_state1 = TorchOptState.load(path1)
        loaded_state2 = TorchOptState.load(path2)
        loaded_state3 = TorchOptState.load(path3)

        assert loaded_state1 == state1
        assert loaded_state2 == state2
        assert loaded_state3 == state3
        assert loaded_state1 != state2
        assert loaded_state2 != state3
        assert loaded_state3 != state1
