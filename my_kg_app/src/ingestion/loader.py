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

    @abstractmethod
    def extract_text(self, doc: Any) -> str:
        """Extract text from loaded document."""
        pass

    @abstractmethod
    def extract_metadata(self, doc: Any) -> dict:
        """Extract metadata from loaded document."""
        pass

class PDFLoader(DocumentLoader):
    """Concrete loader using PyMuPDF (or similar) for PDFs."""
    def load(self, path: str) -> Any:
        import fitz  # PyMuPDF
        doc = fitz.open(path)
        return doc

    def extract_text(self, doc: Any) -> str:
        """Extract text from loaded PDF document."""
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    def extract_metadata(self, doc: Any) -> dict:
        """Extract metadata from loaded PDF document."""
        metadata = doc.metadata
        return metadata
