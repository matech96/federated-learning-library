from fll.backend.torch import TorchModel

from . import util


class TestTorchModel:
    def test_state_storage(self):
        m = TorchModel(util.SimpleModel())
        m2 = TorchModel(util.SimpleModel())
        assert m.get_state() == m.get_state()
        assert m.get_state() != m2.get_state()

    def test_load_save(self):
        m = TorchModel(util.SimpleModel())
        s = m.get_state()
        m2 = TorchModel(util.SimpleModel())
        m2.load_state(s)
        assert m.get_state() == m2.get_state()
