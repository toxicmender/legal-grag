"""
Unit tests for knowledge graph embedding module.
"""

import unittest
import numpy as np
from kg_embedding.maypl_wrapper import MAYPLWrapper
from kg_embedding.fallback_embedder import FallbackEmbedder
from kg_embedding.cache import EmbeddingCache
from kg_embedding.utils import normalize_embeddings, cosine_similarity


class TestMAYPLWrapper(unittest.TestCase):
    """Tests for MAYPLWrapper."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.embedder = MAYPLWrapper(embedding_dim=128)
    
    def test_embedder_initialization(self):
        """Test embedder initialization."""
        self.assertEqual(self.embedder.get_embedding_dim(), 128)
    
    # TODO: Add more tests for MAYPL embedding


class TestFallbackEmbedder(unittest.TestCase):
    """Tests for FallbackEmbedder."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.embedder = FallbackEmbedder(library="pykg2vec", model_name="TransE")
    
    def test_embedder_initialization(self):
        """Test embedder initialization."""
        self.assertEqual(self.embedder.library, "pykg2vec")
        self.assertEqual(self.embedder.model_name, "TransE")
    
    # TODO: Add more tests for fallback embedding


class TestEmbeddingCache(unittest.TestCase):
    """Tests for EmbeddingCache."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cache = EmbeddingCache(cache_dir=".test_cache")
    
    def test_cache_operations(self):
        """Test cache operations."""
        key = "test_entity"
        embedding = np.random.rand(128)
        
        # Test set and get
        self.cache.set(key, embedding)
        retrieved = self.cache.get(key)
        self.assertIsNotNone(retrieved)
        np.testing.assert_array_equal(embedding, retrieved)
        
        # Test exists
        self.assertTrue(self.cache.exists(key))
        self.assertFalse(self.cache.exists("nonexistent"))
    
    # TODO: Add more tests for cache operations


class TestEmbeddingUtils(unittest.TestCase):
    """Tests for embedding utility functions."""
    
    def test_normalize_embeddings(self):
        """Test embedding normalization."""
        embeddings = np.random.rand(10, 128)
        normalized = normalize_embeddings(embeddings, norm="l2")
        
        # Check that norms are approximately 1
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)
    
    def test_cosine_similarity(self):
        """Test cosine similarity computation."""
        emb1 = np.array([1, 0, 0])
        emb2 = np.array([1, 0, 0])
        
        similarity = cosine_similarity(emb1, emb2)
        self.assertAlmostEqual(similarity, 1.0)
        
        emb3 = np.array([0, 1, 0])
        similarity = cosine_similarity(emb1, emb3)
        self.assertAlmostEqual(similarity, 0.0)


if __name__ == '__main__':
    unittest.main()

