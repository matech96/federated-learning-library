from FLL.backend.torch import TorchModelState

from ... import models


class TestTorchModelState:
    def test_save_load(self, tmp_path):
        m = models.Simple()
        state = TorchModelState(m.state_dict())
        path = tmp_path / "model.pt"
        state.save(path)

        m2 = models.Simple()
        state2 = TorchModelState(m2.state_dict())
        path2 = tmp_path / "model2.pt"
        state2.save(path2)

        loaded_state = TorchModelState.load(path)

        assert state == loaded_state
        assert state2 != loaded_state
