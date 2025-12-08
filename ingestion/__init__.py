"""
Document ingestion and parsing module.

This module handles the ingestion of various document formats (PDF, DOC, text)
and provides utilities for parsing, chunking, and metadata extraction.
"""

from .loader import DocumentLoader
from .parser import DocumentParser
from .chunker import DocumentChunker
from .metadata import MetadataExtractor

__all__ = [
    'DocumentLoader',
    'DocumentParser',
    'DocumentChunker',
    'MetadataExtractor',
]

