from abc import ABC, abstractmethod


class AbstractBackendFactory(ABC):
    @classmethod
    @abstractmethod
    def create_data_loader(cls, data_loader):
        pass

    @classmethod
    @abstractmethod
    def create_loss(cls, loss):
        pass

    @classmethod
    @abstractmethod
    def create_metric(cls, metric):
        pass

    @classmethod
    @abstractmethod
    def create_model(cls, model):
        pass

    @classmethod
    @abstractmethod
    def create_model_state(cls, model_state):
        pass

    @classmethod
    @abstractmethod
    def create_opt(cls, opt):
        pass

    @classmethod
    @abstractmethod
    def create_opt_state(cls, opt_state):
        pass
