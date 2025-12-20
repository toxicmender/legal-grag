"""Script to bulk ingest documents from a directory (placeholder)."""
from src.ingestion.loader import load_pdf


def bulk_ingest(directory: str):
    print(f"Would ingest files from {directory}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: bulk_ingest.py <directory>")
    else:
        bulk_ingest(sys.argv[1])
