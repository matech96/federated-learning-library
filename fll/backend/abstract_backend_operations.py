from abc import ABC, abstractmethod

from typing import List, Dict, Optional

from ..backend import AbstractModel, AbstractModelState, AbstractOpt, AbstractOptState, AbstractLoss, \
    AbstractDataLoader, AbstractMetric


class AbstractBackendOperations(ABC):
    @classmethod
    @abstractmethod
    def train_epoch(cls, model: AbstractModel, opt: AbstractOpt, loss: AbstractLoss,
                    data: AbstractDataLoader, metrics: List[AbstractMetric]) -> Dict[str, float]:
        """Trains the model for 1 round.

        :param model: The model to be trained.
        :param opt: The optimization algorithm.
        :param loss: The loss function.
        :param data: The training data.
        :param metrics: The metrics to be measured.
        :return: Dictionary, where the key is the name of the metric and the value is a float or int.
        """

    @classmethod
    @abstractmethod
    def eval(cls, model: AbstractModel, data: AbstractDataLoader, metrics: List[AbstractMetric]) -> Dict[str, float]:
        """Evaluates the model on the provided data.

        :param model: Model to be evaluated.
        :param data: Data to evaluate on. (Should include preprocessing.)
        :param metrics: Metrics of the evaluation.
        :return: Dictionary, where the key is the name of the metric and the value is a float or int.
        """

    @classmethod
    @abstractmethod
    def cumulative_avg_model_state(cls, state_0: Optional[AbstractModelState], state_1: AbstractModelState,
                                   n_states_0: int) -> AbstractModelState:
        """This function is useful to calculate the average of many model states, but without needing to keep all of
        them in memory. With this function you only need two model states simultaneously to calculate the average.

        :param state_0: State form previous cumulative steps. Can be None. If None, return state_1
        :param state_1: New state.
        :param n_states_0: Number of states averaged in state_0.
        :return: The uniform average of all the states in state_0 and state_1.
        """

    @classmethod
    @abstractmethod
    def cumulative_avg_opt_state(cls, state_0: Optional[AbstractOptState], state_1: AbstractOptState,
                                 n_states_0: int) -> AbstractOptState:
        """This function is useful to calculate the average of many optimizer states, but without needing to keep all of
        them in memory. With this function you only need two optimizer states simultaneously to calculate the average.

        :param state_0: State form previous cumulative steps. Can be None. If None, return state_1
        :param state_1: New state.
        :param n_states_0: Number of states averaged in state_0.
        :return: The uniform average of all the states in state_0 and state_1.
        """
