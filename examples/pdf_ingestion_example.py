"""
Example script demonstrating PDF document ingestion and parsing.

This script shows how to use the ingestion module to load and parse PDF documents.
"""

from pathlib import Path
from ingestion.loader import DocumentLoader
from ingestion.parser import DocumentParser
from ingestion.metadata import MetadataExtractor


def example_pdf_ingestion(pdf_path: str):
    """
    Example of ingesting a PDF document.
    
    Args:
        pdf_path: Path to the PDF file.
    """
    print("=" * 60)
    print("PDF Document Ingestion Example")
    print("=" * 60)
    
    # Initialize components
    loader = DocumentLoader()
    parser = DocumentParser()
    metadata_extractor = MetadataExtractor()
    
    # Check if file exists
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        return
    
    print(f"\nProcessing PDF: {pdf_file.name}")
    print(f"File size: {pdf_file.stat().st_size / 1024:.2f} KB")
    
    # Method 1: Load just the text
    print("\n--- Method 1: Load text only ---")
    try:
        text = loader.load(pdf_path)
        print(f"Extracted text length: {len(text)} characters")
        print(f"First 200 characters:\n{text[:200]}...")
    except Exception as e:
        print(f"Error loading text: {e}")
    
    # Method 2: Load with full metadata
    print("\n--- Method 2: Load with full metadata ---")
    try:
        result = loader.load_with_metadata(pdf_path)
        print(f"Text length: {len(result['text'])} characters")
        print(f"Page count: {result.get('page_count', 'N/A')}")
        print(f"Metadata keys: {list(result.get('metadata', {}).keys())}")
        
        # Show PDF metadata if available
        if 'metadata' in result and result['metadata']:
            pdf_meta = result['metadata']
            print("\nPDF Metadata:")
            if pdf_meta.get('title'):
                print(f"  Title: {pdf_meta['title']}")
            if pdf_meta.get('author'):
                print(f"  Author: {pdf_meta['author']}")
            if pdf_meta.get('subject'):
                print(f"  Subject: {pdf_meta['subject']}")
            if pdf_meta.get('creationDate'):
                print(f"  Creation Date: {pdf_meta['creationDate']}")
        
        # Show first page text if available
        if 'pages' in result and result['pages']:
            first_page = result['pages'][0]
            print(f"\nFirst page text (first 200 chars):")
            print(f"{first_page['text'][:200]}...")
            
    except Exception as e:
        print(f"Error loading with metadata: {e}")
    
    # Method 3: Direct parsing
    print("\n--- Method 3: Direct PDF parsing ---")
    try:
        parsed = parser.parse_pdf(pdf_path)
        print(f"Parsed {parsed['page_count']} pages")
        print(f"Total text length: {len(parsed['text'])} characters")
    except Exception as e:
        print(f"Error parsing PDF: {e}")
    
    # Method 4: Extract metadata
    print("\n--- Method 4: Extract metadata ---")
    try:
        metadata = metadata_extractor.extract_from_file(pdf_path)
        print("Extracted metadata:")
        for key, value in metadata.items():
            if key != 'pdf_metadata':  # Skip nested dict for cleaner output
                print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error extracting metadata: {e}")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pdf_ingestion_example.py <path_to_pdf>")
        print("\nExample:")
        print("  python pdf_ingestion_example.py documents/sample.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    example_pdf_ingestion(pdf_path)

