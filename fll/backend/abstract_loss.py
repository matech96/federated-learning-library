from abc import ABC


class AbstractLoss(ABC):
    """Abstract class for a loss function."""

    def __init__(self, name: str):
        self.name = name
