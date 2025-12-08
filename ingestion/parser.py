"""
Document parser using PyMuPDF (or other libraries) to extract text from PDFs and documents.
"""

from typing import Optional, Dict, Any, Union, List
from pathlib import Path
import fitz  # PyMuPDF


class DocumentParser:
    """
    Parser for extracting text from various document formats.
    
    Uses PyMuPDF (fitz) for PDF parsing and other libraries for DOC/DOCX formats.
    """
    
    def __init__(self):
        """Initialize the document parser."""
        pass
    
    def parse_pdf(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse a PDF file and extract text.
        
        Args:
            file_path: Path to the PDF file.
            
        Returns:
            Dictionary containing:
                - text: Extracted text content
                - metadata: Document metadata
                - pages: List of page-level text
                - page_count: Number of pages
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        if not file_path.suffix.lower() == '.pdf':
            raise ValueError(f"File is not a PDF: {file_path}")
        
        try:
            # Open PDF document
            doc = fitz.open(str(file_path))
            
            # Extract text from all pages
            full_text = []
            pages_text = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                pages_text.append({
                    'page_number': page_num + 1,
                    'text': page_text
                })
                full_text.append(page_text)
            
            # Combine all text
            combined_text = "\n\n".join(full_text)
            
            # Extract metadata
            metadata = doc.metadata
            
            # Get page count
            page_count = len(doc)
            
            # Close document
            doc.close()
            
            return {
                'text': combined_text,
                'metadata': metadata,
                'pages': pages_text,
                'page_count': page_count,
                'file_path': str(file_path)
            }
            
        except Exception as e:
            raise RuntimeError(f"Error parsing PDF {file_path}: {str(e)}") from e
    
    def parse_doc(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse a DOC/DOCX file and extract text.
        
        Args:
            file_path: Path to the DOC/DOCX file.
            
        Returns:
            Dictionary containing extracted text and metadata.
        """
        # TODO: Implement DOC/DOCX parsing
        raise NotImplementedError("DOC parsing not yet implemented")
    
    def parse_text(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse a plain text file.
        
        Args:
            file_path: Path to the text file.
            
        Returns:
            Dictionary containing text content.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Text file not found: {file_path}")
        
        try:
            # Read text file with UTF-8 encoding, fallback to latin-1 if needed
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except UnicodeDecodeError:
                # Fallback to latin-1 if UTF-8 fails
                with open(file_path, 'r', encoding='latin-1') as f:
                    text = f.read()
            
            return {
                'text': text,
                'metadata': {},
                'file_path': str(file_path)
            }
            
        except Exception as e:
            raise RuntimeError(f"Error parsing text file {file_path}: {str(e)}") from e
    
    def parse(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Auto-detect file format and parse accordingly.
        
        Args:
            file_path: Path to the document file.
            
        Returns:
            Dictionary containing extracted text and metadata.
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            return self.parse_pdf(file_path)
        elif suffix in ['.txt', '.text']:
            return self.parse_text(file_path)
        elif suffix in ['.doc', '.docx']:
            return self.parse_doc(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Supported formats: .pdf, .txt, .doc, .docx")

