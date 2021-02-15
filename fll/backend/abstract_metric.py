from abc import ABC


class AbstractMetric(ABC):
    """Abstract class for a metric."""

    def __init__(self, name: str):
        self.name = name
