"""KG builder interfaces and simple converter.

This module consolidates the previous `kg_builder` pieces into a single
place. It's intentionally small and easily replaced with more advanced
implementations later.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any


class GraphData:
    def __init__(self, nodes: List[Dict] = None, edges: List[Dict] = None):
        self.nodes = nodes or []
        self.edges = edges or []


class KGBuilder(ABC):
    """Abstract interface for knowledge-graph builders."""

    @abstractmethod
    def build_from_text(self, text: str) -> GraphData:
        pass

    @abstractmethod
    def normalize(self, graph: GraphData) -> GraphData:
        pass


class IText2KGBuilder(KGBuilder):
    """Minimal concrete adapter placeholder for an external tool."""

    def build_from_text(self, text: str) -> GraphData:
        raise NotImplementedError()

    def normalize(self, graph: GraphData) -> GraphData:
        return graph


def convert_text_to_kg(text: str) -> Dict[str, Any]:
    """Simple converter that produces a minimal KG-like dict.

    Replace this with a call to a real extractor / converter.
    """
    return {"nodes": [], "edges": []}


ENTITY_TYPES = ["Person", "Organization", "LegalCase"]
