"""Base prompt templates and helpers."""

def format_prompt(context: str, question: str) -> str:
    return f"Context:\n{context}\n\nQuestion:\n{question}\n"