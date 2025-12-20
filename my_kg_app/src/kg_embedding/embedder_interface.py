"""Abstract interface for embedders."""
from abc import ABC, abstractmethod

class EmbedderInterface(ABC):
    @abstractmethod
    def embed(self, items):
        raise NotImplementedError
