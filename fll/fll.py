from typing import Callable

from .backend.abstract import AbstractModel, AbstractOpt, AbstractBackendFactory


class ModelOptFactory:
    def __init__(self, model_cls: Callable[[], AbstractModel], opt_cls: Callable[[], AbstractOpt]):
        self.model_cls = model_cls
        self.opt_cls = opt_cls

    def make_objects(self):
        model = self.model_cls()
        opt = self.opt_cls()
