"""
Bulk ingestion script for processing multiple documents.
"""

import argparse
from pathlib import Path
from typing import List, Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.loader import DocumentLoader
from ingestion.parser import DocumentParser
from ingestion.metadata import MetadataExtractor
from kg_construction.extractor import EntityRelationExtractor
from kg_construction.graph_builder import GraphBuilder


def bulk_ingest(
    input_dir: str,
    output_dir: Optional[str] = None,
    file_pattern: str = "*.*",
    recursive: bool = True
) -> None:
    """
    Bulk ingest documents from a directory.
    
    Args:
        input_dir: Directory containing documents to ingest.
        output_dir: Optional output directory for processed documents.
        file_pattern: File pattern to match (e.g., "*.pdf").
        recursive: Whether to search recursively.
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return
    
    # Initialize components
    loader = DocumentLoader()
    parser = DocumentParser()
    metadata_extractor = MetadataExtractor()
    extractor = EntityRelationExtractor()
    graph_builder = GraphBuilder()
    
    # Find files
    if recursive:
        files = list(input_path.rglob(file_pattern))
    else:
        files = list(input_path.glob(file_pattern))
    
    print(f"Found {len(files)} files to process.")
    
    # Process each file
    for i, file_path in enumerate(files, 1):
        print(f"\nProcessing {i}/{len(files)}: {file_path.name}")
        
        try:
            # Load document
            text = loader.load(file_path)
            
            # Extract metadata
            metadata = metadata_extractor.extract_from_file(file_path)
            
            # Extract entities and relations
            entities = extractor.extract_entities(text)
            relations = extractor.extract_relations(text, entities=entities)
            
            # Build graph
            graph_builder.build_from_entities_relations(entities, relations)
            
            print(f"  ✓ Processed: {len(entities)} entities, {len(relations)} relations")
            
        except Exception as e:
            print(f"  ✗ Error processing {file_path.name}: {e}")
            continue
    
    print(f"\n✓ Bulk ingestion complete. Processed {len(files)} files.")


def main():
    """Main entry point for bulk ingestion script."""
    parser = argparse.ArgumentParser(description="Bulk ingest documents into knowledge graph")
    parser.add_argument("input_dir", help="Input directory containing documents")
    parser.add_argument("--output-dir", help="Output directory for processed documents")
    parser.add_argument("--pattern", default="*.*", help="File pattern to match (default: *.*)")
    parser.add_argument("--no-recursive", action="store_true", help="Don't search recursively")
    
    args = parser.parse_args()
    
    bulk_ingest(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        file_pattern=args.pattern,
        recursive=not args.no_recursive
    )


if __name__ == "__main__":
    main()

