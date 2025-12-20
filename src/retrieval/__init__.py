"""Consolidated retrieval package.

This package unifies retrieval and retrieval+generation interfaces and
utilities. Import high-level classes and functions from here; older
`retrieval_generation` module is kept as a compatibility shim.
"""
from .retriever import retrieve
from .integration import retrieve_and_rank
from .ranking import rank
from .graph_to_context import graph_to_context

# Higher-level engine + generation APIs
from .engine import Retriever, Generator, KGBasedRetriever, LLMGenerator  # type: ignore
from .generation import generate_answer  # type: ignore
from .prompt_engineering import build_prompt  # type: ignore

__all__ = [
	"retrieve",
	"retrieve_and_rank",
	"rank",
	"graph_to_context",
	"Retriever",
	"Generator",
	"KGBasedRetriever",
	"LLMGenerator",
	"generate_answer",
	"build_prompt",
]