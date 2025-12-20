"""Unified retrieval+generation engine interfaces.

Defines abstract `Retriever` and `Generator` interfaces and simple placeholders
for common implementations. These are intentionally minimal — replace with
real implementations as needed.
"""
from typing import Any
from abc import ABC, abstractmethod


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> Any:
        pass


class Generator(ABC):
    @abstractmethod
    def generate(self, context: Any, prompt: str) -> str:
        pass


class KGBasedRetriever(Retriever):
    def __init__(self, repo, embedder):
        self.repo = repo
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 5) -> Any:
        raise NotImplementedError()


class LLMGenerator(Generator):
    def __init__(self, llm, prompt_template: str):
        self.llm = llm
        self.prompt_template = prompt_template

    def generate(self, context: Any, prompt: str) -> str:
        raise NotImplementedError()
