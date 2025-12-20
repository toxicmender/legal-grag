"""Embedding interfaces and simple implementations consolidated here."""
from abc import ABC, abstractmethod
from typing import Any, List


class EmbedderInterface(ABC):
    @abstractmethod
    def embed(self, items: List[Any]) -> List[Any]:
        raise NotImplementedError


class FallbackEmbedder(EmbedderInterface):
    """Deterministic fallback - useful for tests or when no real embedder is
    available."""

    def embed(self, items: List[Any]) -> List[float]:
        return [0.0 for _ in items]


class EmbeddingCache:
    def __init__(self):
        self._store = {}

    def get(self, key):
        return self._store.get(key)

    def set(self, key, value):
        self._store[key] = value

    def clear(self):
        self._store.clear()


def normalize_for_embedding(text: str) -> str:
    return text.strip().lower()


class MAYPLWrapper(EmbedderInterface):
    """Placeholder wrapper for a primary embedder implementation."""

    def embed(self, items: List[Any]) -> List[Any]:
        # Real implementation would call out to MAYPL or other provider
        return []
