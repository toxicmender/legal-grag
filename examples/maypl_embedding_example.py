"""
Example script demonstrating knowledge graph embedding using MAYPL.

This script shows how to use the MAYPL wrapper to create embeddings
for entities and relations in a knowledge graph.
"""

import os
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kg_construction.models import Entity, Relation
from kg_construction.graph_builder import GraphBuilder
from kg_embedding.maypl_wrapper import MAYPLWrapper


def example_maypl_embedding(
    entities: list,
    relations: list,
    maypl_path: Optional[str] = None,
    train: bool = True,
    model_path: Optional[str] = None
):
    """
    Example of creating embeddings using MAYPL.
    
    Args:
        entities: List of Entity objects or dictionaries.
        relations: List of Relation objects or dictionaries.
        maypl_path: Optional path to MAYPL code directory.
        train: Whether to train the model (True) or load from checkpoint (False).
        model_path: Path to saved model checkpoint (if loading).
    """
    print("=" * 60)
    print("MAYPL Knowledge Graph Embedding Example")
    print("=" * 60)
    
    # Step 1: Prepare knowledge graph
    print("\n--- Step 1: Prepare Knowledge Graph ---")
    graph_builder = GraphBuilder()
    graph = graph_builder.build_from_entities_relations(entities, relations)
    
    stats = graph.get_statistics()
    print("✓ Knowledge graph prepared:")
    print(f"  Entities: {stats['entity_count']}")
    print(f"  Relations: {stats['relation_count']}")
    print(f"  Entity types: {stats['entity_types']}")
    print(f"  Relation types: {stats['relation_types']}")
    
    # Step 2: Initialize MAYPL wrapper
    print("\n--- Step 2: Initialize MAYPL Wrapper ---")
    
    # Set MAYPL path if provided
    if maypl_path:
        os.environ['MAYPL_PATH'] = maypl_path
        print(f"✓ Set MAYPL_PATH to: {maypl_path}")
    elif 'MAYPL_PATH' not in os.environ:
        print("⚠ Warning: MAYPL_PATH not set. Trying default locations...")
    
    try:
        embedder = MAYPLWrapper(
            embedding_dim=128,
            maypl_path=maypl_path,
            device="cuda" if os.getenv("CUDA_AVAILABLE", "false").lower() == "true" else "cpu"
        )
        print("✓ MAYPL wrapper initialized")
    except ImportError as e:
        print(f"✗ Error: {e}")
        print("\nTo set up MAYPL:")
        print("  1. Run: python scripts/setup_maypl.py")
        print("  2. Or clone manually: git clone https://github.com/bdi-lab/MAYPL.git")
        print("  3. Set MAYPL_PATH environment variable")
        return None
    
    # Step 3: Train or load model
    if train:
        print("\n--- Step 3: Train MAYPL Model ---")
        try:
            graph_data = {
                'entities': list(graph.entities.values()),
                'relations': list(graph.relations.values())
            }
            
            embedder.fit(
                graph_data,
                train_epochs=50,  # Reduced for example
                batch_size=256,
                learning_rate=0.001
            )
            print("✓ Model training completed")
            
            # Save model
            if model_path:
                embedder.save(model_path)
                print(f"✓ Model saved to {model_path}")
        except Exception as e:
            print(f"✗ Error training model: {e}")
            import traceback
            traceback.print_exc()
            return None
    else:
        print("\n--- Step 3: Load MAYPL Model ---")
        if not model_path:
            print("✗ Error: model_path required when train=False")
            return None
        
        try:
            embedder.load(model_path)
            print(f"✓ Model loaded from {model_path}")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return None
    
    # Step 4: Extract embeddings
    print("\n--- Step 4: Extract Embeddings ---")
    try:
        # Embed sample entities
        sample_entities = list(graph.entities.values())[:5]
        entity_embeddings = embedder.embed_entities([
            {'id': e.id, 'name': e.name, 'type': e.entity_type}
            for e in sample_entities
        ])
        
        print(f"✓ Extracted embeddings for {len(sample_entities)} entities")
        print(f"  Embedding shape: {entity_embeddings.shape}")
        print(f"  Sample entity: {sample_entities[0].name}")
        print(f"  Embedding (first 10 dims): {entity_embeddings[0][:10]}")
        
        # Embed sample relations
        sample_relations = list(graph.relations.values())[:5]
        relation_embeddings = embedder.embed_relations([
            {'id': r.id, 'type': r.relation_type}
            for r in sample_relations
        ])
        
        print(f"\n✓ Extracted embeddings for {len(sample_relations)} relations")
        print(f"  Embedding shape: {relation_embeddings.shape}")
        print(f"  Sample relation: {sample_relations[0].relation_type}")
        print(f"  Embedding (first 10 dims): {relation_embeddings[0][:10]}")
        
    except Exception as e:
        print(f"✗ Error extracting embeddings: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    print("\n" + "=" * 60)
    print("MAYPL embedding example completed!")
    print("=" * 60)
    
    return embedder


def create_sample_graph():
    """Create a sample knowledge graph for testing."""
    entities = [
        Entity(id="e1", name="Alice", entity_type="PERSON"),
        Entity(id="e2", name="Bob", entity_type="PERSON"),
        Entity(id="e3", name="Company X", entity_type="ORGANIZATION"),
        Entity(id="e4", name="Contract", entity_type="DOCUMENT"),
    ]
    
    relations = [
        Relation(
            id="r1",
            source_entity_id="e1",
            target_entity_id="e3",
            relation_type="WORKS_FOR"
        ),
        Relation(
            id="r2",
            source_entity_id="e2",
            target_entity_id="e3",
            relation_type="WORKS_FOR"
        ),
        Relation(
            id="r3",
            source_entity_id="e1",
            target_entity_id="e4",
            relation_type="SIGNED"
        ),
        Relation(
            id="r4",
            source_entity_id="e3",
            target_entity_id="e4",
            relation_type="ISSUED"
        ),
    ]
    
    return entities, relations


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MAYPL embedding example")
    parser.add_argument("--maypl-path", help="Path to MAYPL code directory")
    parser.add_argument("--model-path", help="Path to saved model checkpoint")
    parser.add_argument("--load-only", action="store_true", help="Load model instead of training")
    parser.add_argument("--use-sample", action="store_true", help="Use sample graph for testing")
    
    args = parser.parse_args()
    
    if args.use_sample:
        entities, relations = create_sample_graph()
        print("Using sample knowledge graph for demonstration")
    else:
        print("Error: Please provide entities and relations, or use --use-sample flag")
        print("\nExample usage:")
        print("  python maypl_embedding_example.py --use-sample --maypl-path ./maypl")
        sys.exit(1)
    
    example_maypl_embedding(
        entities=entities,
        relations=relations,
        maypl_path=args.maypl_path,
        train=not args.load_only,
        model_path=args.model_path
    )

