"""
Knowledge Graph embedding and representation learning module.

This module handles graph representation learning, including entity and relation embeddings.
"""

from .embedder_interface import EmbedderInterface
from .maypl_wrapper import MAYPLWrapper
from .fallback_embedder import FallbackEmbedder
from .cache import EmbeddingCache

__all__ = [
    'EmbedderInterface',
    'MAYPLWrapper',
    'FallbackEmbedder',
    'EmbeddingCache',
]

