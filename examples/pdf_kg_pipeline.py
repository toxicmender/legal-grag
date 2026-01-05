"""
Complete pipeline example: PDF reading → Lexical Analysis → Knowledge Graph → Visualization.

This script demonstrates the full pipeline for processing PDFs from the
Central_Government_Acts directory, performing lexical analysis, creating
knowledge graphs using itext2kg, and visualizing the results.
"""

import os
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.pdf_reader import PDFReader
from ingestion.lexical_analyzer import LexicalAnalyzer
from kg_construction.extractor import EntityRelationExtractor
from kg_construction.graph_builder import GraphBuilder
from kg_construction.visualization import GraphVisualizer


def process_pdf_pipeline(
    pdf_path: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    output_dir: str = "output",
    visualize: bool = True
) -> None:
    """
    Process a PDF through the complete pipeline.
    
    Args:
        pdf_path: Path to PDF file. If None, processes all PDFs in Central_Government_Acts.
        openai_api_key: OpenAI API key for itext2kg. If None, uses OPENAI_API_KEY env var.
        output_dir: Directory to save visualization outputs.
        visualize: Whether to generate visualizations.
    """
    print("=" * 80)
    print("PDF to Knowledge Graph Pipeline")
    print("=" * 80)
    
    # Get API key
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key is required for itext2kg")
        print("  Set OPENAI_API_KEY environment variable or pass as parameter")
        return
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialize components
    pdf_reader = PDFReader()
    lexical_analyzer = LexicalAnalyzer()
    
    # Determine which PDFs to process
    if pdf_path:
        pdf_files = [Path(pdf_path)]
    else:
        pdf_files = pdf_reader.list_pdfs()
        if not pdf_files:
            print(f"No PDF files found in {pdf_reader.base_dir}")
            return
    
    print(f"\nFound {len(pdf_files)} PDF file(s) to process\n")
    
    # Process each PDF
    for pdf_file in pdf_files:
        print("-" * 80)
        print(f"Processing: {pdf_file.name}")
        print("-" * 80)
        
        try:
            # Step 1: Read PDF
            print("\n[Step 1] Reading PDF...")
            pdf_data = pdf_reader.read_pdf_with_metadata(pdf_file)
            text = pdf_data['text']
            print(f"✓ Extracted {len(text)} characters from {pdf_data['page_count']} pages")
            
            if not text.strip():
                print("⚠ Warning: No text extracted from PDF. Skipping...")
                continue
            
            # Step 2: Lexical Analysis
            print("\n[Step 2] Performing lexical analysis...")
            analysis = lexical_analyzer.analyze(text)
            stats = analysis['statistics']
            print(f"✓ Tokenized: {stats['word_count']} words, {stats['sentence_count']} sentences")
            print(f"✓ Unique words: {stats['unique_words']}")
            print(f"✓ Average word length: {stats['average_word_length']}")
            
            # Step 3: Knowledge Graph Creation
            print("\n[Step 3] Creating knowledge graph with itext2kg...")
            extractor = EntityRelationExtractor(openai_api_key=api_key)
            
            # Use processed text for better results
            result = extractor.extract_from_text(analysis['processed_text'])
            entities = result['entities']
            relations = result['relations']
            
            print(f"✓ Extracted {len(entities)} entities")
            print(f"✓ Extracted {len(relations)} relations")
            
            if not entities:
                print("⚠ Warning: No entities extracted. Skipping visualization...")
                continue
            
            # Show sample entities
            print("\nSample entities:")
            for entity in entities[:5]:
                print(f"  - {entity.name} ({entity.entity_type})")
            
            # Step 4: Build Knowledge Graph
            print("\n[Step 4] Building knowledge graph...")
            graph_builder = GraphBuilder()
            graph = graph_builder.build_from_entities_relations(entities, relations)
            
            graph_stats = graph.get_statistics()
            print(f"✓ Graph built: {graph_stats['entity_count']} entities, "
                  f"{graph_stats['relation_count']} relations")
            print(f"✓ Entity types: {graph_stats['entity_types']}")
            print(f"✓ Relation types: {graph_stats['relation_types']}")
            
            # Step 5: Visualization
            if visualize:
                print("\n[Step 5] Generating visualization...")
                visualizer = GraphVisualizer(figsize=(14, 10))
                
                # Create output filename
                pdf_stem = pdf_file.stem
                viz_path = output_path / f"{pdf_stem}_kg_visualization.png"
                
                # Visualize with filtering for readability (limit to top entities)
                max_nodes = 50 if len(entities) > 50 else None
                visualizer.visualize(
                    graph,
                    output_path=str(viz_path),
                    layout='spring',
                    show_labels=True,
                    max_nodes=max_nodes
                )
                print(f"✓ Visualization saved to {viz_path}")
            
            print(f"\n✓ Successfully processed {pdf_file.name}")
            
        except Exception as e:
            print(f"\n✗ Error processing {pdf_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 80)
    print("Pipeline completed!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process PDFs through lexical analysis and knowledge graph creation"
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="Path to specific PDF file (default: process all PDFs in dataset)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (default: use OPENAI_API_KEY environment variable)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for visualizations (default: output)"
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip visualization generation"
    )
    
    args = parser.parse_args()
    
    process_pdf_pipeline(
        pdf_path=args.pdf,
        openai_api_key=args.api_key,
        output_dir=args.output_dir,
        visualize=not args.no_visualize
    )

