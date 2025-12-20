"""Simplified repository API for the graph store.

Defines an abstract interface and a Neo4j implementation.
"""

# Existing simple implementation kept for now
# src/graph_store/repository.py

from abc import ABC, abstractmethod
from typing import Any, Dict
from kg.storage import KGStorageInterface

class GraphRepository(ABC):
    """Repository interface for storing and retrieving graph data."""
    @abstractmethod
    def connect(self) -> None:
        pass

    @abstractmethod
    def add_graph(self, graph_data: Any) -> None:
        """Add nodes & edges to the store."""
        pass

    @abstractmethod
    def query_subgraph(self, criteria: Dict) -> Any:
        """Retrieve subgraph matching some criteria (e.g. by entities, types)."""
        pass

    @abstractmethod
    def close(self) -> None:
        pass

class Neo4jRepository(GraphRepository, KGStorageInterface):
    def __init__(self, uri: str, user: str, password: str):
        # store config
        self._uri = uri
        self._user = user
        self._password = password
        self._driver = None

    def connect(self) -> None:
        from neo4j import GraphDatabase
        self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))

    def add_graph(self, graph_data: Any) -> None:
        # placeholder: convert GraphData to Neo4j nodes & edges and write
        raise NotImplementedError()

    # KGStorageInterface compatibility
    def save_graph(self, kg) -> None:
        """Adapter method to save a KnowledgeGraph via repository.

        This is a simple shim that reuses `add_graph`. Real implementations
        would map the `kg` object to the repository's expected `graph_data`.
        """
        # For now, forward to add_graph (expects GraphData-like structure)
        self.add_graph(kg)

    def load_graph(self, query: str):
        """Adapter method to query and return a KnowledgeGraph-like object.

        Real implementations should translate the `query`/criteria and
        reconstruct a `KnowledgeGraph` object.
        """
        return self.query_subgraph({"query": query})

    def query_subgraph(self, criteria: Dict) -> Any:
        # placeholder: execute a query and return subgraph
        raise NotImplementedError()

    def close(self) -> None:
        if self._driver:
            self._driver.close()
