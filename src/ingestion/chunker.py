"""Chunking utilities for splitting long documents into passages."""

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = max(0, end - overlap)
    return chunks
