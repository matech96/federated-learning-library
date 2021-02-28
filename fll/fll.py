"""FLL provides an easy way to experiment with federated learning."""

from typing import NamedTuple, Optional

from fll.backend.abstract import AbstractModelOptFactory, AbstractModelState, AbstractOptState, AbstractModel, \
    AbstractOpt

ModelOptState = NamedTuple('ModelOptState', [('model_state', AbstractModelState), ('opt_state', AbstractOptState)])


class ModelOptStateManager:
    """Resource manager for a model and an associated optimizer."""

    def __init__(self, is_cached: bool, factory: AbstractModelOptFactory):
        """

        :param is_cached: Keep the model and optimizer, even if not in a with statement.
        :param factory: Creates the model and the optimizer.
        """
        self.is_cached = is_cached
        self.factory = factory

        self.model: Optional[AbstractModel] = None
        self.opt: Optional[AbstractOpt] = None

        self._state_to_load: Optional[ModelOptState] = None

    def __enter__(self):
        if self.model is None:
            assert self.opt is None
            self.model, self.opt = self.factory.make_objects()
            if self._state_to_load is not None:
                self.model.load_state(self._state_to_load.model_state)
                self.opt.load_state(self._state_to_load.opt_state)
        else:
            assert self.opt is not None

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.is_cached:
            self.model = None
            self.opt = None

    def set_state(self, state: ModelOptState):
        """Sets the state for them model and the optimizer.

        :param state: State of the model and the optimizer to be loaded.
        """
        if self.model is None:
            assert self.opt is None
            self._state_to_load = state
        else:
            assert self.opt is not None
            self.model.load_state(state.model_state)
            self.opt.load_state(state.opt_state)
