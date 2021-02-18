import torch as th
from fll.backend.torchbackend import TorchOpt

from . import util


class TestTorchOpt:
    def test_state_storage(self):
        m = util.SimpleModel()
        o1 = th.optim.Adam(params=m.parameters())
        o2 = th.optim.Adam(params=m.parameters(), lr=1)
        warped_o1 = TorchOpt(o1)
        warped_o2 = TorchOpt(o2)
        assert warped_o1.get_state() == warped_o1.get_state()
        assert warped_o1.get_state() != warped_o2.get_state()

    def test_load_save(self):
        m = util.SimpleModel()
        o = th.optim.Adam(params=m.parameters())
        o2 = th.optim.Adam(params=m.parameters(), lr=1)
        warped_o = TorchOpt(o)
        warped_o2 = TorchOpt(o2)
        s = warped_o.get_state()
        warped_o2.load_state(s)
        assert warped_o.get_state() == warped_o2.get_state()
