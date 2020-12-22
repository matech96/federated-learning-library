# import torch as th
#
# from fll.backend.torch import TorchOptState
#
# from ... import models
#
#
# class TestTorchModelState:
#     def test_save_load(self, tmp_path):
#         m = models.Simple()
#         o = th.optim.Adam(params=m.parameters())
#         state = TorchOptState(o.state_dict())
#         path = tmp_path / "opt.pt"
#         state.save(path)
#
#         loaded_state = TorchOptState.load(path)
#
#         assert state == loaded_state
