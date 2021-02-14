import torch as th

from .. import AbstractDataLoader


class TorchDataLoader(AbstractDataLoader):
    """An implementation of AbstractDataLoader for Torch."""
    def __init__(self, data_loader):
        assert isinstance(data_loader, th.utils.data.DataLoader)
        super().__init__(data_loader)
