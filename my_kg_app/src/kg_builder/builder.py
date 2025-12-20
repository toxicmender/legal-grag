"""Simplified KG builder facade.

Defines interfaces and concrete implementations for building knowledge graphs from text.
"""

from abc import ABC, abstractmethod
from typing import List, Any, Dict

class GraphData:
    """
    Simple container for nodes/edges extracted from text.
    e.g. {'nodes': [...], 'edges': [...]}
    """
    def __init__(self, nodes: List[Dict], edges: List[Dict]):
        self.nodes = nodes
        self.edges = edges

class KGBuilder(ABC):
    """Interface for a knowledge-graph builder."""
    @abstractmethod
    def build_from_text(self, text: str) -> GraphData:
        """Extract entities/relations from text into GraphData."""
        pass

    @abstractmethod
    def normalize(self, graph: GraphData) -> GraphData:
        """Optional: normalize entities / merge duplicates / standardize types."""
        pass

class IText2KGBuilder(KGBuilder):
    """Concrete builder wrapping iText2KG (or similar library)."""
    def build_from_text(self, text: str) -> GraphData:
        # placeholder: call out to iText2KG, convert to GraphData
        raise NotImplementedError()

    def normalize(self, graph: GraphData) -> GraphData:
        # optional normalization logic
        return graph
