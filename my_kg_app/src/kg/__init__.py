"""Unified knowledge-graph package.

This package consolidates builder, construction and embedding utilities in a
single cohesive API. Import modules under `kg` (for example `kg.builder` or
`kg.build_kg_from_text`) for access to the refactored implementation.
"""
from .builder import (
    GraphData,
    KGBuilder,
    IText2KGBuilder,
    convert_text_to_kg,
    ENTITY_TYPES,
)
from .construction import distill, extract, build_kg_from_text
from .embedding import (
    EmbedderInterface,
    FallbackEmbedder,
    EmbeddingCache,
    normalize_for_embedding,
    MAYPLWrapper,
)

__all__ = [
    "GraphData",
    "KGBuilder",
    "IText2KGBuilder",
    "convert_text_to_kg",
    "ENTITY_TYPES",
    "distill",
    "extract",
    "build_kg_from_text",
    "EmbedderInterface",
    "FallbackEmbedder",
    "EmbeddingCache",
    "normalize_for_embedding",
    "MAYPLWrapper",
]
