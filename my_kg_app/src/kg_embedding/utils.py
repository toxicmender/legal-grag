"""Utility helpers for embedding workflows."""

def normalize_for_embedding(text: str) -> str:
    return text.strip().lower()
