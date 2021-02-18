import torch as th

from fll.backend.torchbackend import TorchBackendOperations
from .test_torch_util import SignProvider


class TestTorchBackendOperations:
    def test_train_epoch(self):
        prov = SignProvider()
        for i in range(20):
            res = TorchBackendOperations.train_epoch(prov.model, prov.opt, prov.loss, prov.dl, prov.metrics)
            if i == 0:
                assert res["Accuracy"] < 0.9
                first_loss = res["loss"]
        assert res["Accuracy"] > 0.9999
        assert res["loss"] < first_loss

    def test_eval(self):
        th.manual_seed(0)
        prov = SignProvider(1000)
        res = TorchBackendOperations.eval(prov.model, prov.dl, prov.metrics)
        assert 0.49 < res["Accuracy"] < 0.5
        for _ in range(10):
            TorchBackendOperations.train_epoch(prov.model, prov.opt, prov.loss, prov.dl, prov.metrics)
        res = TorchBackendOperations.eval(prov.model, prov.dl, prov.metrics)
        assert 0.98 < res["Accuracy"]

        prov.opt.opt.param_groups[0]['lr'] = 0.0  # set learning rate to 0
        res_train = TorchBackendOperations.train_epoch(prov.model, prov.opt, prov.loss, prov.dl, prov.metrics)
        assert res["Accuracy"] == res_train["Accuracy"]  # check if eval and train function report the same accuracy
