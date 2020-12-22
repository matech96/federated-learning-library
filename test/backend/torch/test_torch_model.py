from fll.backend.torch import TorchModel

from ... import models


class TestTorchModel:
    def test_state_storage(self):
        m = TorchModel(models.Simple())
        m2 = TorchModel(models.Simple())
        assert m.get_state() == m.get_state()
        assert m.get_state() == m2.get_state()  # TODO !=

    def test_load_save(self):
        m = TorchModel(models.Simple())
        s = m.get_state()
        m2 = TorchModel(models.Simple())
        m2.load_state(s)
        assert m.get_state() == m2.get_state()
