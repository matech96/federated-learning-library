from fll.backend.torchbackend import TorchModel

from . import test_torch_util


class TestTorchModel:
    def test_state_storage(self):
        m = TorchModel(test_torch_util.SimpleModel())
        m2 = TorchModel(test_torch_util.SimpleModel())
        assert m.get_state() == m.get_state()
        assert m.get_state() != m2.get_state()

    def test_load_save(self):
        m = TorchModel(test_torch_util.SimpleModel())
        s = m.get_state()
        m2 = TorchModel(test_torch_util.SimpleModel())
        m2.load_state(s)
        assert m.get_state() == m2.get_state()
