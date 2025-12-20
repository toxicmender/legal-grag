"""Integration helpers tying retrieval and ranking together."""
from .retriever import retrieve
from .ranking import rank


def retrieve_and_rank(query: str, top_k: int = 10):
    candidates = retrieve(query, top_k=top_k)
    return rank(candidates, query)
