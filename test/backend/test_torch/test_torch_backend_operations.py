import torch as th

from fll.backend.torchbackend import TorchBackendOperations, TorchBackendFactory
from .test_torch_util import SignProvider


class TestTorchBackendOperations:
    def test_train_epoch(self):
        th.manual_seed(0)
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

    def test_cumulative_avg_opt_state_random_model(self):
        provs = [SignProvider(is_complex_model=True) for _ in range(10)]
        wrapped_states = [prov.model.get_state() for prov in provs]
        avg_state = avg_model_state_dicts([ws.state for ws in wrapped_states])

        n_processed_states = 0
        running_states = None
        for s in wrapped_states:
            running_states = TorchBackendOperations.cumulative_avg_model_state(running_states, s, n_processed_states)
            n_processed_states += 1

        assert TorchBackendFactory.create_model_state(avg_state) == running_states

    def test_cumulative_avg_opt_state_trained_model(self):
        provs = [SignProvider(is_complex_model=True) for _ in range(10)]
        cum_state = None
        n_states = 0
        for prov in provs:
            TorchBackendOperations.train_epoch(prov.model, prov.opt, prov.loss, prov.dl, prov.metrics)
            new_state = prov.model.get_state()
            cum_state = TorchBackendOperations.cumulative_avg_model_state(cum_state, new_state, n_states)
            n_states += 1

        state_dicts = [prov.model.get_state().state for prov in provs]
        avg_state = avg_model_state_dicts(state_dicts)
        assert TorchBackendFactory.create_model_state(avg_state) == cum_state


def avg_model_state_dicts(state_dicts):
    final_state_dict = {}
    with th.no_grad():
        for parameter_name in state_dicts[0].keys():
            if (not isinstance(state_dicts[0][parameter_name], th.Tensor)) or (
                    state_dicts[0][parameter_name].dtype == th.int64
            ):
                final_state_dict[parameter_name] = state_dicts[0][parameter_name]
                for state_dict in state_dicts:
                    assert final_state_dict[parameter_name] == state_dict[parameter_name]
                continue

            final_state_dict[parameter_name] = th.mean(
                th.stack(
                    [
                        model_parameters[parameter_name]
                        for model_parameters in state_dicts
                    ]
                ),
                dim=0,
            )
    return final_state_dict


def avg_models(models):
    final_model = models[0].copy()

    parameters = [model.state_dict() for model in models]
    final_state_dict = avg_model_state_dicts(parameters)
    final_model.load_state_dict(final_state_dict)

    return final_model
