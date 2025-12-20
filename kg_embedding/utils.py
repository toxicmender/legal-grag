"""
Helper functions for embedding operations.

Includes normalization, batching, and other utility functions.
"""

from typing import List
import numpy as np


def normalize_embeddings(embeddings: np.ndarray, norm: str = "l2") -> np.ndarray:
    """
    Normalize embeddings to unit length.
    
    Args:
        embeddings: Array of embeddings to normalize.
        norm: Norm type ('l2', 'l1', 'max').
        
    Returns:
        Normalized embeddings array.
    """
    if norm == "l2":
        norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms
    elif norm == "l1":
        norms = np.sum(np.abs(embeddings), axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms
    elif norm == "max":
        max_vals = np.max(np.abs(embeddings), axis=-1, keepdims=True)
        max_vals = np.where(max_vals == 0, 1, max_vals)
        return embeddings / max_vals
    else:
        raise ValueError(f"Unknown norm type: {norm}")


def batch_items(items: List, batch_size: int) -> List[List]:
    """
    Split a list of items into batches.
    
    Args:
        items: List of items to batch.
        batch_size: Size of each batch.
        
    Returns:
        List of batches.
    """
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        emb1: First embedding array.
        emb2: Second embedding array.
        
    Returns:
        Cosine similarity score.
    """
    dot_product = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def euclidean_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two embeddings.
    
    Args:
        emb1: First embedding array.
        emb2: Second embedding array.
        
    Returns:
        Euclidean distance.
    """
    return np.linalg.norm(emb1 - emb2)


def compute_similarity_matrix(embeddings: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """
    Compute pairwise similarity matrix for embeddings.
    
    Args:
        embeddings: Array of embeddings (n_samples, embedding_dim).
        metric: Similarity metric ('cosine', 'euclidean').
        
    Returns:
        Similarity matrix (n_samples, n_samples).
    """
    n_samples = embeddings.shape[0]
    similarity_matrix = np.zeros((n_samples, n_samples))
    
    if metric == "cosine":
        # Normalize embeddings first
        normalized = normalize_embeddings(embeddings, norm="l2")
        similarity_matrix = np.dot(normalized, normalized.T)
    elif metric == "euclidean":
        for i in range(n_samples):
            for j in range(n_samples):
                similarity_matrix[i, j] = euclidean_distance(embeddings[i], embeddings[j])
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return similarity_matrix

