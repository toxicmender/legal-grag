"""Graph builder facade that delegates to existing `kg_builder` code."""
from src.kg_builder.builder import build_kg_from_text as _build


def build_kg_from_text(text: str) -> dict:
    return _build(text)
