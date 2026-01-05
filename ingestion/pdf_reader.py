"""
PDF Reader module for reading PDFs from Central Government Acts directory.

This module provides a convenient interface for reading PDF files specifically
from the dataset/Central_Government_Acts directory using PyMuPDF.
"""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from .parser import DocumentParser


class PDFReader:
    """
    PDF Reader for Central Government Acts.
    
    Provides methods to read PDF files from the Central_Government_Acts directory
    and return structured text with page-level information.
    """
    
    def __init__(self, parser: Optional[DocumentParser] = None):
        """
        Initialize the PDF reader.
        
        Args:
            parser: Optional DocumentParser instance. If not provided, creates a new one.
        """
        self.parser = parser or DocumentParser()
        self.base_dir = Path("dataset/Central_Government_Acts")
    
    def read_pdf(self, pdf_path: Union[str, Path]) -> str:
        """
        Read a PDF file and return its text content.
        
        Args:
            pdf_path: Path to the PDF file (can be relative to base_dir or absolute).
            
        Returns:
            Extracted text content from the PDF.
            
        Raises:
            FileNotFoundError: If the PDF file does not exist.
            ValueError: If the file is not a PDF.
            RuntimeError: If there's an error parsing the PDF.
        """
        pdf_path = self._resolve_path(pdf_path)
        parsed_result = self.parser.parse_pdf(pdf_path)
        return parsed_result.get('text', '')
    
    def read_pdf_with_metadata(self, pdf_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Read a PDF file and return text with full metadata.
        
        Args:
            pdf_path: Path to the PDF file (can be relative to base_dir or absolute).
            
        Returns:
            Dictionary containing:
                - text: Extracted text content
                - metadata: Document metadata
                - pages: List of page-level text dictionaries
                - page_count: Number of pages
                - file_path: Path to the PDF file
                
        Raises:
            FileNotFoundError: If the PDF file does not exist.
            ValueError: If the file is not a PDF.
            RuntimeError: If there's an error parsing the PDF.
        """
        pdf_path = self._resolve_path(pdf_path)
        return self.parser.parse_pdf(pdf_path)
    
    def list_pdfs(self) -> List[Path]:
        """
        List all PDF files in the Central_Government_Acts directory.
        
        Returns:
            List of Path objects for all PDF files found.
        """
        if not self.base_dir.exists():
            return []
        
        pdf_files = list(self.base_dir.glob("*.pdf"))
        return sorted(pdf_files)
    
    def read_all_pdfs(self) -> Dict[str, Dict[str, Any]]:
        """
        Read all PDF files from the Central_Government_Acts directory.
        
        Returns:
            Dictionary mapping PDF filenames to their parsed results.
            Each value contains text, metadata, pages, and page_count.
        """
        pdf_files = self.list_pdfs()
        results = {}
        
        for pdf_path in pdf_files:
            try:
                parsed_result = self.read_pdf_with_metadata(pdf_path)
                results[pdf_path.name] = parsed_result
            except Exception as e:
                # Log error but continue with other files
                results[pdf_path.name] = {
                    'error': str(e),
                    'text': '',
                    'metadata': {},
                    'pages': [],
                    'page_count': 0
                }
        
        return results
    
    def _resolve_path(self, pdf_path: Union[str, Path]) -> Path:
        """
        Resolve PDF path - check if it's absolute, relative to base_dir, or relative to current dir.
        
        Args:
            pdf_path: Path to resolve.
            
        Returns:
            Resolved Path object.
        """
        pdf_path = Path(pdf_path)
        
        # If absolute path, use as is
        if pdf_path.is_absolute():
            return pdf_path
        
        # Check if it's relative to base_dir
        base_dir_path = self.base_dir / pdf_path
        if base_dir_path.exists():
            return base_dir_path
        
        # Check if it's relative to current directory
        if pdf_path.exists():
            return pdf_path.resolve()
        
        # If not found, return the base_dir_path (will raise error in parser)
        return base_dir_path

