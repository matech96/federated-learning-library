from fll.backend.torchbackend import TorchBackendOperations
from .util import SignProvider


class TestTorchBackendOperations:
    def test_train_epoch(self):
        prov = SignProvider()
        for i in range(20):
            res = TorchBackendOperations.train_epoch(prov.model, prov.opt, prov.loss, prov.dl, prov.metrics)
            if i == 0:
                assert res["Accuracy"] < 0.9
                first_loss = res["loss"]
        assert res["Accuracy"] > 0.9999, res["Accuracy"]
        assert res["loss"] < first_loss
