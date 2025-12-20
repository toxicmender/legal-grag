"""Prompt templates and chaining utilities for LLM calls."""

def build_prompt(context: str, question: str) -> str:
    return f"Context:\n{context}\nQuestion:\n{question}"
