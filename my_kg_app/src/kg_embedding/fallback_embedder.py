"""Simple fallback embedder if MAYPL isn't available."""
from .embedder_interface import EmbedderInterface

class FallbackEmbedder(EmbedderInterface):
    def embed(self, items):
        # deterministic simple fallback (e.g., hashing)
        return [0.0 for _ in items]
