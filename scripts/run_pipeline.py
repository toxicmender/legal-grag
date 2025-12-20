"""Orchestration script to run a simple pipeline: ingest -> build KG -> embed -> store.

This is a placeholder that demonstrates wiring between modules.
"""
from src.ingestion.loader import load_pdf
from kg import build_kg_from_text
from src.graph_learning.embedder import embed


def run(path: str):
    text = load_pdf(path)
    kg = build_kg_from_text(text)
    embeddings = embed(kg)
    print("KG built:", kg)
    print("Embeddings:", embeddings)


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: run_pipeline.py <path-to-pdf>")
    else:
        run(sys.argv[1])
