"""
High-level document loader.

Handles ingestion of documents in various formats (PDF, DOC, text)
and returns raw text for further processing.
"""

from typing import Union, Optional, List, Dict, Any
from pathlib import Path
from .parser import DocumentParser
from .chunker import DocumentChunker, Chunk


class DocumentLoader:
    """
    High-level document loader for ingesting documents.
    
    Supports multiple formats: PDF, DOC, DOCX, TXT, etc.
    Returns raw text extracted from documents.
    """
    
    def __init__(
        self,
        parser: Optional[DocumentParser] = None,
        chunker: Optional[DocumentChunker] = None,
        max_chars_without_chunking: int = 20000,
        max_pages_without_chunking: int = 10,
    ):
        """
        Initialize the document loader.
        
        Args:
            parser: Optional parser instance for document parsing.
                    If not provided, a new DocumentParser will be created.
            chunker: Optional DocumentChunker for chunk-based loading.
                     If not provided, a new DocumentChunker will be created
                     when chunk-based methods are used.
            max_chars_without_chunking: Maximum character count before
                                        chunking is applied automatically.
            max_pages_without_chunking: Maximum page count before
                                        chunking is applied automatically.
        """
        self.parser = parser or DocumentParser()
        self._chunker = chunker
        self.max_chars_without_chunking = max_chars_without_chunking
        self.max_pages_without_chunking = max_pages_without_chunking

    @property
    def chunker(self) -> DocumentChunker:
        """
        Lazily initialized DocumentChunker instance.
        """
        if self._chunker is None:
            self._chunker = DocumentChunker()
        return self._chunker
    
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

    def load_chunks(
        self,
        file_path: Union[str, Path],
        use_metadata: bool = True,
    ) -> List[Chunk]:
        """
        Load a document and return it as chunks, using metadata to decide
        whether chunking is required.

        For smaller documents, this returns a single chunk containing
        the full text. For larger documents (based on metadata such as
        page_count and character length), the text is split into multiple
        fixed-size chunks.

        Args:
            file_path: Path to the document file.
            use_metadata: Whether to use parsing metadata (e.g. page_count)
                          to decide if chunking is necessary.

        Returns:
            List of Chunk objects.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is not supported.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        parsed_result = self.parser.parse(file_path)
        text = parsed_result.get("text", "") or ""
        page_count = parsed_result.get("page_count")

        if not text:
            return []

        should_chunk = False

        if use_metadata:
            if page_count is not None and page_count > self.max_pages_without_chunking:
                should_chunk = True

        if len(text) > self.max_chars_without_chunking:
            should_chunk = True

        if not should_chunk:
            return [
                Chunk(
                    text=text,
                    start_index=0,
                    end_index=len(text),
                    chunk_id="0",
                    metadata={
                        "file_path": str(file_path),
                        "page_count": page_count,
                        "chunk_index": 0,
                        "is_chunked": False,
                    },
                )
            ]

        chunks = self.chunker.chunk_by_size(text)
        total_chunks = len(chunks)

        for idx, chunk in enumerate(chunks):
            if chunk.metadata is None:
                chunk.metadata = {}
            chunk.metadata.update(
                {
                    "file_path": str(file_path),
                    "page_count": page_count,
                    "chunk_index": idx,
                    "total_chunks": total_chunks,
                    "is_chunked": True,
                }
            )

        return chunks

