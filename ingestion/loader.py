"""
High-level document loader.

Handles ingestion of documents in various formats (PDF, DOC, text)
and returns raw text for further processing.
"""

from typing import Union, Optional, List, Dict, Any
from pathlib import Path
from .parser import DocumentParser


class DocumentLoader:
    """
    High-level document loader for ingesting documents.
    
    Supports multiple formats: PDF, DOC, DOCX, TXT, etc.
    Returns raw text extracted from documents.
    """
    
    def __init__(self, parser: Optional[DocumentParser] = None):
        """
        Initialize the document loader.
        
        Args:
            parser: Optional parser instance for document parsing.
                    If not provided, a new DocumentParser will be created.
        """
        self.parser = parser or DocumentParser()
    
    def load(self, file_path: Union[str, Path]) -> str:
        """
        Load a document and return raw text.
        
        Args:
            file_path: Path to the document file.
            
        Returns:
            Raw text content of the document.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is not supported.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        # Parse the document
        parsed_result = self.parser.parse(file_path)
        
        # Return the extracted text
        return parsed_result.get('text', '')
    
    def load_with_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a document and return text with full parsing metadata.
        
        Args:
            file_path: Path to the document file.
            
        Returns:
            Dictionary containing:
                - text: Raw text content
                - metadata: Document metadata
                - pages: Page-level text (for PDFs)
                - page_count: Number of pages (for PDFs)
                - file_path: Path to the document
                
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is not supported.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        # Parse the document
        return self.parser.parse(file_path)
    
    def load_from_url(self, url: str) -> str:
        """
        Load a document from a URL and return raw text.
        
        Args:
            url: URL to the document.
            
        Returns:
            Raw text content of the document.
        """
        # TODO: Implement URL-based document loading
        raise NotImplementedError("URL-based loading not yet implemented")
    
    def load_batch(self, file_paths: List[Union[str, Path]]) -> List[str]:
        """
        Load multiple documents in batch.
        
        Args:
            file_paths: List of paths to document files.
            
        Returns:
            List of raw text contents for each document.
        """
        results = []
        for file_path in file_paths:
            try:
                text = self.load(file_path)
                results.append(text)
            except Exception as e:
                # Log error but continue with other files
                print(f"Error loading {file_path}: {e}")
                results.append("")  # Append empty string for failed loads
        return results

