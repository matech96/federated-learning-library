import copy
from functools import partial
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

    def test_cumulative_avg_model_state_same_model(self):
        prov = SignProvider(is_complex_model=True)
        models = [copy.deepcopy(prov.model) for _ in range(10)]
        n_processed_states = 0
        running_states = None
        for wrapped_model in models:
            wrapped_state = wrapped_model.get_state()
            running_states = TorchBackendOperations.cumulative_avg_model_state(running_states, wrapped_state,
                                                                               n_processed_states)
            n_processed_states += 1

        assert prov.model.get_state() == running_states

    def test_cumulative_avg_model_state_random_model(self):
        provs = [SignProvider(is_complex_model=True) for _ in range(10)]
        wrapped_states = [prov.model.get_state() for prov in provs]
        avg_state = avg_state_dicts([ws.state for ws in wrapped_states])

        n_processed_states = 0
        running_states = None
        for s in wrapped_states:
            running_states = TorchBackendOperations.cumulative_avg_model_state(running_states, s, n_processed_states)
            n_processed_states += 1

        assert TorchBackendFactory.create_model_state(avg_state) == running_states

    def test_cumulative_avg_model_state_trained_model(self):
        provs = [SignProvider(is_complex_model=True) for _ in range(10)]
        cum_state = None
        n_states = 0
        for prov in provs:
            TorchBackendOperations.train_epoch(prov.model, prov.opt, prov.loss, prov.dl, prov.metrics)
            new_state = prov.model.get_state()
            cum_state = TorchBackendOperations.cumulative_avg_model_state(cum_state, new_state, n_states)
            n_states += 1

        state_dicts = [prov.model.get_state().state for prov in provs]
        avg_state = avg_state_dicts(state_dicts)
        assert TorchBackendFactory.create_model_state(avg_state) == cum_state

    def test_cumulative_avg_opt_state_same_model(self):
        for opt_class in [th.optim.Adam, partial(th.optim.SGD, momentum=0.9, nesterov=True), th.optim.Adagrad]:
            prov = SignProvider(is_complex_model=True, opt_class=opt_class)
            models = [copy.deepcopy(prov.model) for _ in range(10)]
            opts = [TorchBackendFactory.create_opt(opt_class(lr=0.1, params=model.model.parameters())) for model in
                    models]
            n_processed_states = 0
            running_opt_states = None
            for wrapped_model, wrapped_opt in zip(models, opts):
                TorchBackendOperations.train_epoch(wrapped_model, wrapped_opt, prov.loss, prov.dl, prov.metrics)
                wrapped_opt_state = wrapped_opt.get_state()
                running_opt_states = TorchBackendOperations.cumulative_avg_opt_state(running_opt_states,
                                                                                     wrapped_opt_state,
                                                                                     n_processed_states)
                n_processed_states += 1

            for opt in opts:
                assert opt.get_state() == running_opt_states


def avg_state_dicts(state_dicts):
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
