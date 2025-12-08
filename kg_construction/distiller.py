"""
Document Distiller for extracting semantic blocks and sections from documents.

Prepares text for knowledge graph construction by splitting into semantic sections.
"""

from typing import List, Dict, Any, Optional
import re


class DocumentDistiller:
    """
    Document Distiller that extracts semantic blocks and sections from documents.
    
    Converts raw text into structured semantic units that can be used
    for entity and relation extraction, particularly for itext2kg.
    """
    
    def __init__(self, max_section_length: int = 2000):
        """
        Initialize the document distiller.
        
        Args:
            max_section_length: Maximum length of a section in characters.
        """
        self.max_section_length = max_section_length
    
    def distill(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Distill document into semantic blocks/sections for itext2kg processing.
        
        Args:
            text: Raw text content of the document.
            metadata: Optional metadata dictionary.
            
        Returns:
            List of semantic blocks, each containing:
                - text: Block text content
                - section_type: Type of section (e.g., 'paragraph', 'clause', 'section')
                - position: Position in document
                - metadata: Block-specific metadata
        """
        # First, try to extract structural sections
        sections = self.extract_sections(text)
        
        # If no structural sections found, use semantic blocks
        if not sections:
            sections = self.extract_semantic_blocks(text)
        
        # Add metadata to each section
        for i, section in enumerate(sections):
            section['position'] = i
            if metadata:
                section['metadata'].update(metadata)
        
        return sections
    
    def extract_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract structural sections from document.
        
        Args:
            text: Raw text content.
            
        Returns:
            List of section dictionaries with structure information.
        """
        sections = []
        
        # Split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        for para in paragraphs:
            # Check if paragraph is a heading (all caps, short, or numbered)
            is_heading = (
                len(para) < 100 and 
                (para.isupper() or re.match(r'^\d+\.?\s+[A-Z]', para))
            )
            
            section_type = 'heading' if is_heading else 'paragraph'
            
            sections.append({
                'text': para,
                'section_type': section_type,
                'metadata': {}
            })
        
        return sections
    
    def extract_semantic_blocks(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract semantic blocks (meaningful units) from document.
        
        Splits text into chunks that are suitable for itext2kg processing.
        
        Args:
            text: Raw text content.
            
        Returns:
            List of semantic block dictionaries.
        """
        blocks = []
        
        # Split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_block = ""
        for sentence in sentences:
            # If adding this sentence would exceed max length, start new block
            if len(current_block) + len(sentence) > self.max_section_length and current_block:
                blocks.append({
                    'text': current_block.strip(),
                    'section_type': 'semantic_block',
                    'metadata': {}
                })
                current_block = sentence
            else:
                current_block += " " + sentence if current_block else sentence
        
        # Add the last block
        if current_block.strip():
            blocks.append({
                'text': current_block.strip(),
                'section_type': 'semantic_block',
                'metadata': {}
            })
        
        return blocks

