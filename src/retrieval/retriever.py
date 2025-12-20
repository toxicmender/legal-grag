"""Graph retriever: query graph_store and return candidate subgraphs.

This module contains simple retrieval helpers. Replace with repository-backed
implementations that query Neo4j or another graph store.
"""
from kg.storage import KGStorageInterface
from kg.construction import KnowledgeGraph


def retrieve(query: str, top_k: int = 10) -> list:
    # Placeholder: query the graph store and return candidate nodes/subgraphs
    return []


def retrieve_subgraph(query: str) -> dict:
    """Compatibility helper: return a single subgraph for generation flows."""
    return {"nodes": [], "edges": []}

class KGRetriever:
    def __init__(self, storage: KGStorageInterface, embedder=None):
        """Storage should implement `KGStorageInterface` so this retriever
        is independent of the actual backend (Neo4j, in-memory, file, etc.).
        """
        self.storage = storage
        self.embedder = embedder

    def retrieve_subgraph(self, query: str, top_k: int = 5) -> KnowledgeGraph:
        """
        Given a user query (natural language), retrieve a relevant subgraph from storage.
        Optionally use embedding-based similarity ranking if embedder provided.
        """
        # Example placeholder:
        # 1. Extract query entities (maybe via simple NER or LLM).
        # 2. Query Neo4j for neighborhoods around those entities.
        # 3. If embedder available, embed candidate subgraphs and rank by similarity to query.
        kg = self.storage.load_graph(query=query)  # placeholder
        return kg
