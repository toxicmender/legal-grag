"""Chain-of-thought prompt templates and utilities."""

def cot_prompt(question: str) -> str:
    return f"Think step-by-step about: {question}"