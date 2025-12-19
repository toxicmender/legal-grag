"""
Example script demonstrating knowledge graph construction using itext2kg.

This script shows how to use the kg_construction module with parsed text
from the ingestion module to build a knowledge graph.
"""

import os
from typing import Optional
from pathlib import Path
from ingestion.loader import DocumentLoader
from ingestion.parser import DocumentParser
from kg_construction.distiller import DocumentDistiller
from kg_construction.extractor import EntityRelationExtractor
from kg_construction.graph_builder import GraphBuilder


def example_kg_construction(pdf_path: str, openai_api_key: Optional[str] = None):
    """
    Example of building a knowledge graph from a PDF document.

    Args:
        pdf_path: Path to the PDF file.
        openai_api_key: Optional OpenAI API key. If not provided, will use OPENAI_API_KEY env var.
    """
    print("=" * 60)
    print("Knowledge Graph Construction Example (using itext2kg)")
    print("=" * 60)

    # Step 1: Ingest and parse the document
    print("\n--- Step 1: Document Ingestion ---")
    loader = DocumentLoader()
    _parser = DocumentParser()

    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        return

    print(f"Loading PDF: {pdf_file.name}")
    try:
        # Load text from PDF
        text = loader.load(pdf_path)
        print(f"✓ Extracted {len(text)} characters of text")

        # Get full parsing results for metadata
        parsed_result = loader.load_with_metadata(pdf_path)
        print(f"✓ Document has {parsed_result.get('page_count', 'unknown')} pages")

    except Exception as e:
        print(f"✗ Error loading document: {e}")
        return

    # Step 2: Distill document into semantic sections
    print("\n--- Step 2: Document Distillation ---")
    distiller = DocumentDistiller()
    try:
        sections = distiller.distill(text, metadata=parsed_result.get('metadata', {}))
        print(f"✓ Created {len(sections)} semantic sections")
        print(f"  Section types: {set(s.get('section_type') for s in sections)}")
    except Exception as e:
        print(f"✗ Error distilling document: {e}")
        return

    # Step 3: Extract entities and relations using itext2kg
    print("\n--- Step 3: Entity and Relation Extraction (itext2kg) ---")

    # Get API key
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("✗ Error: OpenAI API key is required for itext2kg")
        print("  Set OPENAI_API_KEY environment variable or pass as parameter")
        return

    try:
        extractor = EntityRelationExtractor(openai_api_key=api_key)

        print("Extracting entities and relations...")
        result = extractor.extract_from_text(text)

        entities = result['entities']
        relations = result['relations']

        print(f"✓ Extracted {len(entities)} entities")
        print(f"✓ Extracted {len(relations)} relations")

        # Show sample entities
        if entities:
            print("\nSample entities:")
            for entity in entities[:5]:
                print(f"  - {entity.name} ({entity.entity_type})")

        # Show sample relations
        if relations:
            print("\nSample relations:")
            for relation in relations[:5]:
                source = next((e.name for e in entities if e.id == relation.source_entity_id), "?")
                target = next((e.name for e in entities if e.id == relation.target_entity_id), "?")
                print(f"  - {source} --[{relation.relation_type}]--> {target}")

    except Exception as e:
        print(f"✗ Error extracting entities/relations: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 4: Build knowledge graph
    print("\n--- Step 4: Knowledge Graph Construction ---")
    try:
        graph_builder = GraphBuilder()
        graph = graph_builder.build_from_entities_relations(entities, relations)

        stats = graph.get_statistics()
        print("✓ Knowledge graph built successfully!")
        print(f"  Entities: {stats['entity_count']}")
        print(f"  Relations: {stats['relation_count']}")
        print(f"  Entity types: {stats['entity_types']}")
        print(f"  Relation types: {stats['relation_types']}")

        # Show graph structure
        print("\nGraph structure:")
        entity_types = {}
        for entity in graph.entities.values():
            entity_types[entity.entity_type] = entity_types.get(entity.entity_type, 0) + 1

        for etype, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {etype}: {count} entities")

    except Exception as e:
        print(f"✗ Error building knowledge graph: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("Knowledge graph construction completed!")
    print("=" * 60)

    return graph


if __name__ == "__main__":
    import sys
    from typing import Optional

    if len(sys.argv) < 2:
        print("Usage: python kg_construction_example.py <path_to_pdf> [openai_api_key]")
        print("\nExample:")
        print("  python kg_construction_example.py documents/sample.pdf")
        print("  python kg_construction_example.py documents/sample.pdf sk-...")
        print("\nNote: OpenAI API key can also be set via OPENAI_API_KEY environment variable")
        sys.exit(1)

    pdf_path = sys.argv[1]
    api_key = sys.argv[2] if len(sys.argv) > 2 else None

    example_kg_construction(pdf_path, api_key)

