from abc import ABC


class AbstractDataLoader(ABC):
    """Abstract class for a data loader."""
    def __init__(self, data_loader):
        self.data_loader = data_loader
