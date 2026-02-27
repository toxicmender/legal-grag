"""
Document chunker for splitting large documents into manageable pieces.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class Chunk:
    """Represents a chunk of text from a document."""
    text: str
    start_index: int
    end_index: int
    chunk_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DocumentChunker:
    """
    Chunks large documents into smaller, manageable pieces.
    
    Supports various chunking strategies:
    - Fixed-size chunks
    - Sentence-based chunking
    - Paragraph-based chunking
    - Semantic chunking (using embeddings)
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document chunker.
        
        Args:
            chunk_size: Maximum size of each chunk (in characters or tokens).
            chunk_overlap: Number of characters/tokens to overlap between chunks.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_by_size(self, text: str) -> List[Chunk]:
        """
        Chunk text by fixed size.
        
        Args:
            text: Text to chunk.
            
        Returns:
            List of Chunk objects.
        """
        if not text:
            return []

        chunks: List[Chunk] = []
        text_length = len(text)
        start = 0
        chunk_index = 0

        step = self.chunk_size - self.chunk_overlap

        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunk_text = text[start:end]

            chunks.append(
                Chunk(
                    text=chunk_text,
                    start_index=start,
                    end_index=end,
                    chunk_id=str(chunk_index),
                    metadata={
                        "chunk_index": chunk_index,
                        "total_length": text_length,
                    },
                )
            )

            chunk_index += 1
            start = end

            if start >= text_length or step <= 0:
                break

            start = max(0, start - self.chunk_overlap)

        return chunks
    
    def chunk_by_sentences(self, text: str) -> List[Chunk]:
        """
        Chunk text by sentences.
        
        Args:
            text: Text to chunk.
            
        Returns:
            List of Chunk objects.
        """
        # TODO: Implement sentence-based chunking
        raise NotImplementedError("Sentence-based chunking not yet implemented")
    
    def chunk_by_paragraphs(self, text: str) -> List[Chunk]:
        """
        Chunk text by paragraphs.
        
        Args:
            text: Text to chunk.
            
        Returns:
            List of Chunk objects.
        """
        # TODO: Implement paragraph-based chunking
        raise NotImplementedError("Paragraph-based chunking not yet implemented")
    
    def chunk_semantic(self, text: str, embeddings: Optional[List] = None) -> List[Chunk]:
        """
        Chunk text semantically using embeddings.
        
        Args:
            text: Text to chunk.
            embeddings: Optional pre-computed embeddings.
            
        Returns:
            List of Chunk objects.
        """
        # TODO: Implement semantic chunking
        raise NotImplementedError("Semantic chunking not yet implemented")

