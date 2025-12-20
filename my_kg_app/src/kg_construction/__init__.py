"""kg_construction package - adapters and builders for KG construction.

Provides adapters that keep compatibility with previous `kg_builder` modules.
"""
from .graph_builder import build_kg_from_text

__all__ = ["build_kg_from_text"]
