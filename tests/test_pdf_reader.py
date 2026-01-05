"""
Unit tests for PDFReader module.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from ingestion.pdf_reader import PDFReader
from ingestion.parser import DocumentParser


class TestPDFReader:
    """Tests for PDFReader class."""
    
    def test_initialization(self):
        """Test PDFReader initialization."""
        reader = PDFReader()
        assert reader is not None
        assert reader.base_dir == Path("dataset/Central_Government_Acts")
        assert isinstance(reader.parser, DocumentParser)
    
    def test_initialization_with_custom_parser(self):
        """Test PDFReader initialization with custom parser."""
        custom_parser = Mock(spec=DocumentParser)
        reader = PDFReader(parser=custom_parser)
        assert reader.parser is custom_parser
    
    def test_list_pdfs_no_directory(self, tmp_path):
        """Test list_pdfs when directory doesn't exist."""
        reader = PDFReader()
        # Temporarily change base_dir to non-existent directory
        reader.base_dir = tmp_path / "nonexistent"
        pdfs = reader.list_pdfs()
        assert pdfs == []
    
    def test_list_pdfs_empty_directory(self, tmp_path):
        """Test list_pdfs with empty directory."""
        reader = PDFReader()
        reader.base_dir = tmp_path
        pdfs = reader.list_pdfs()
        assert pdfs == []
    
    def test_list_pdfs_with_files(self, tmp_path):
        """Test list_pdfs finds PDF files."""
        reader = PDFReader()
        reader.base_dir = tmp_path
        
        # Create some PDF files
        (tmp_path / "test1.pdf").touch()
        (tmp_path / "test2.pdf").touch()
        (tmp_path / "not_a_pdf.txt").touch()
        
        pdfs = reader.list_pdfs()
        assert len(pdfs) == 2
        assert all(p.suffix == '.pdf' for p in pdfs)
    
    def test_resolve_path_absolute(self, tmp_path):
        """Test _resolve_path with absolute path."""
        reader = PDFReader()
        abs_path = tmp_path / "test.pdf"
        abs_path.touch()
        
        resolved = reader._resolve_path(abs_path)
        assert resolved == abs_path
    
    def test_resolve_path_relative_to_base(self, tmp_path):
        """Test _resolve_path with path relative to base_dir."""
        reader = PDFReader()
        reader.base_dir = tmp_path
        
        pdf_file = tmp_path / "test.pdf"
        pdf_file.touch()
        
        resolved = reader._resolve_path("test.pdf")
        assert resolved == pdf_file
    
    @patch('ingestion.pdf_reader.DocumentParser')
    def test_read_pdf_success(self, mock_parser_class, tmp_path):
        """Test successful PDF reading."""
        mock_parser = Mock()
        mock_parser.parse_pdf.return_value = {
            'text': 'Sample PDF text',
            'metadata': {},
            'pages': [],
            'page_count': 1
        }
        mock_parser_class.return_value = mock_parser
        
        reader = PDFReader(parser=mock_parser)
        reader.base_dir = tmp_path
        
        pdf_file = tmp_path / "test.pdf"
        pdf_file.touch()
        
        text = reader.read_pdf("test.pdf")
        assert text == 'Sample PDF text'
        mock_parser.parse_pdf.assert_called_once()
    
    @patch('ingestion.pdf_reader.DocumentParser')
    def test_read_pdf_with_metadata(self, mock_parser_class, tmp_path):
        """Test reading PDF with metadata."""
        mock_parser = Mock()
        expected_result = {
            'text': 'Sample text',
            'metadata': {'title': 'Test'},
            'pages': [{'page_number': 1, 'text': 'Sample text'}],
            'page_count': 1
        }
        mock_parser.parse_pdf.return_value = expected_result
        mock_parser_class.return_value = mock_parser
        
        reader = PDFReader(parser=mock_parser)
        reader.base_dir = tmp_path
        
        pdf_file = tmp_path / "test.pdf"
        pdf_file.touch()
        
        result = reader.read_pdf_with_metadata("test.pdf")
        assert result == expected_result
    
    @patch('ingestion.pdf_reader.DocumentParser')
    def test_read_pdf_file_not_found(self, mock_parser_class):
        """Test read_pdf raises error for missing file."""
        mock_parser = Mock()
        mock_parser.parse_pdf.side_effect = FileNotFoundError("File not found")
        mock_parser_class.return_value = mock_parser
        
        reader = PDFReader(parser=mock_parser)
        
        with pytest.raises(FileNotFoundError):
            reader.read_pdf("nonexistent.pdf")
    
    @patch('ingestion.pdf_reader.DocumentParser')
    def test_read_all_pdfs(self, mock_parser_class, tmp_path):
        """Test reading all PDFs from directory."""
        mock_parser = Mock()
        mock_parser.parse_pdf.return_value = {
            'text': 'Sample text',
            'metadata': {},
            'pages': [],
            'page_count': 1
        }
        mock_parser_class.return_value = mock_parser
        
        reader = PDFReader(parser=mock_parser)
        reader.base_dir = tmp_path
        
        # Create PDF files
        (tmp_path / "pdf1.pdf").touch()
        (tmp_path / "pdf2.pdf").touch()
        
        results = reader.read_all_pdfs()
        assert len(results) == 2
        assert 'pdf1.pdf' in results
        assert 'pdf2.pdf' in results
    
    @patch('ingestion.pdf_reader.DocumentParser')
    def test_read_all_pdfs_with_error(self, mock_parser_class, tmp_path):
        """Test read_all_pdfs handles errors gracefully."""
        mock_parser = Mock()
        mock_parser.parse_pdf.side_effect = [
            {'text': 'Success', 'metadata': {}, 'pages': [], 'page_count': 1},
            RuntimeError("Parse error")
        ]
        mock_parser_class.return_value = mock_parser
        
        reader = PDFReader(parser=mock_parser)
        reader.base_dir = tmp_path
        
        (tmp_path / "good.pdf").touch()
        (tmp_path / "bad.pdf").touch()
        
        results = reader.read_all_pdfs()
        assert len(results) == 2
        assert 'good.pdf' in results
        assert 'bad.pdf' in results
        assert 'error' in results['bad.pdf']

