"""Abstract storage interface and optional Neo4j adapter for KGs."""
from abc import ABC, abstractmethod
from typing import Optional

try:
    from neo4j import GraphDatabase
    _NEO4J_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    GraphDatabase = object
    _NEO4J_AVAILABLE = False


class KGStorageInterface(ABC):
    """Abstract interface for persisting and loading KnowledgeGraph objects.

    Implementations should accept whatever backend is appropriate (Neo4j,
    in-memory, file, etc.) and expose `save_graph` / `load_graph` methods.
    """

    @abstractmethod
    def save_graph(self, kg) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_graph(self, query: str):
        raise NotImplementedError


class Neo4jKGStorage(KGStorageInterface):
    """Lightweight Neo4j adapter that wraps the official driver.

    This class is intentionally minimal; real projects should implement
    proper session/transaction management and mapping of KG objects.
    """

    def __init__(self, uri: str, user: Optional[str] = None, password: Optional[str] = None):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        if _NEO4J_AVAILABLE:
            # auth=(user, password) may be None if not provided
            self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def save_graph(self, kg) -> None:
        # Placeholder: implement mapping of `kg` to Neo4j nodes/relations
        return None

    def load_graph(self, query: str):
        # Placeholder: execute a query and return a KnowledgeGraph-like object
        from .construction import KnowledgeGraph

        return KnowledgeGraph()


class InMemoryKGStorage(KGStorageInterface):
    """Simple in-memory storage for KnowledgeGraph objects keyed by an
    identifier. Useful for tests and local development.

    Usage:
        store = InMemoryKGStorage()
        store.save_graph(kg, key="mykg")
        kg2 = store.load_graph(key="mykg")
    """

    def __init__(self):
        # store mapping from key -> KnowledgeGraph-like object
        self._store = {}

    def save_graph(self, kg, key: str = "default") -> None:
        self._store[key] = kg

    def load_graph(self, query: str = None, key: str = "default"):
        # If a key provided, return stored KG; otherwise return empty KG
        from .construction import KnowledgeGraph

        if key in self._store:
            return self._store[key]
        return KnowledgeGraph()
