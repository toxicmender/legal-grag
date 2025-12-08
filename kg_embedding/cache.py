"""
Caching and persistence for embeddings.
"""

from typing import Dict, Optional, List
import numpy as np
import pickle
from pathlib import Path


class EmbeddingCache:
    """
    Cache for storing and retrieving embeddings.
    
    Provides persistence for embeddings to avoid recomputation.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the embedding cache.
        
        Args:
            cache_dir: Directory to store cached embeddings.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".cache/embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache: Dict[str, np.ndarray] = {}
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """
        Get an embedding from cache.
        
        Args:
            key: Cache key (e.g., entity_id or relation_id).
            
        Returns:
            Cached embedding array or None if not found.
        """
        # Check memory cache first
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    embedding = pickle.load(f)
                self.memory_cache[key] = embedding
                return embedding
            except Exception as e:
                # TODO: Log error
                return None
        
        return None
    
    def set(self, key: str, embedding: np.ndarray, persist: bool = True) -> None:
        """
        Store an embedding in cache.
        
        Args:
            key: Cache key.
            embedding: Embedding array to cache.
            persist: Whether to persist to disk.
        """
        # Store in memory cache
        self.memory_cache[key] = embedding
        
        # Persist to disk if requested
        if persist:
            cache_file = self.cache_dir / f"{key}.pkl"
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(embedding, f)
            except Exception as e:
                # TODO: Log error
                pass
    
    def get_batch(self, keys: List[str]) -> Dict[str, Optional[np.ndarray]]:
        """
        Get multiple embeddings from cache.
        
        Args:
            keys: List of cache keys.
            
        Returns:
            Dictionary mapping keys to embeddings (or None if not found).
        """
        return {key: self.get(key) for key in keys}
    
    def set_batch(self, embeddings: Dict[str, np.ndarray], persist: bool = True) -> None:
        """
        Store multiple embeddings in cache.
        
        Args:
            embeddings: Dictionary mapping keys to embeddings.
            persist: Whether to persist to disk.
        """
        for key, embedding in embeddings.items():
            self.set(key, embedding, persist)
    
    def clear(self, memory_only: bool = False) -> None:
        """
        Clear the cache.
        
        Args:
            memory_only: If True, only clear memory cache, not disk cache.
        """
        self.memory_cache.clear()
        
        if not memory_only:
            # Clear disk cache
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    # TODO: Log error
                    pass
    
    def exists(self, key: str) -> bool:
        """
        Check if an embedding exists in cache.
        
        Args:
            key: Cache key.
            
        Returns:
            True if embedding exists, False otherwise.
        """
        if key in self.memory_cache:
            return True
        
        cache_file = self.cache_dir / f"{key}.pkl"
        return cache_file.exists()

