"""Document loader helpers (placeholder).

Implement PDF loading (e.g., PyMuPDF) here.
"""

from abc import ABC, abstractmethod
from typing import Any

class DocumentLoader(ABC):
    """Abstract loader for documents."""
    @abstractmethod
    def load(self, path: str) -> Any:
        """
        Load raw document (PDF, DOCX, HTML, etc.)
        Return a raw document object for parsing.
        """
        pass

class PDFLoader(DocumentLoader):
    """Concrete loader using PyMuPDF (or similar) for PDFs."""
    def load(self, path: str) -> Any:
        import fitz  # PyMuPDF
        doc = fitz.open(path)
        return doc
