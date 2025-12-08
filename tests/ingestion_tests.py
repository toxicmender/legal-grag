"""
Unit tests for document ingestion module.
"""

import unittest
from ingestion.loader import DocumentLoader
from ingestion.parser import DocumentParser
from ingestion.chunker import DocumentChunker
from ingestion.metadata import MetadataExtractor


class TestDocumentLoader(unittest.TestCase):
    """Tests for DocumentLoader."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = DocumentLoader()
    
    def test_loader_initialization(self):
        """Test loader initialization."""
        self.assertIsNotNone(self.loader)
    
    # TODO: Add more tests for document loading


class TestDocumentParser(unittest.TestCase):
    """Tests for DocumentParser."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parser = DocumentParser()
    
    def test_parser_initialization(self):
        """Test parser initialization."""
        self.assertIsNotNone(self.parser)
    
    # TODO: Add more tests for document parsing


class TestDocumentChunker(unittest.TestCase):
    """Tests for DocumentChunker."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
    
    def test_chunker_initialization(self):
        """Test chunker initialization."""
        self.assertEqual(self.chunker.chunk_size, 1000)
        self.assertEqual(self.chunker.chunk_overlap, 200)
    
    # TODO: Add more tests for document chunking


class TestMetadataExtractor(unittest.TestCase):
    """Tests for MetadataExtractor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = MetadataExtractor()
    
    def test_extractor_initialization(self):
        """Test extractor initialization."""
        self.assertIsNotNone(self.extractor)
    
    # TODO: Add more tests for metadata extraction


if __name__ == '__main__':
    unittest.main()

