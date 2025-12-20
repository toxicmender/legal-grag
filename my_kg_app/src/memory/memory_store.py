"""Simple memory store abstraction (placeholder)."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any

class MemoryStore(ABC):
    @abstractmethod
    def save(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        pass

    @abstractmethod
    def load(self, session_id: str) -> List[Dict[str, Any]]:
        pass

class InMemoryStore(MemoryStore):
    def __init__(self):
        self._store = {}

    def save(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        self._store[session_id] = messages

    def load(self, session_id: str) -> List[Dict[str, Any]]:
        return self._store.get(session_id, [])
