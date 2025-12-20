"""Compatibility shim: re-export retrieval+generation APIs from `retrieval`.

This module remains for backwards compatibility with code that imports
`retrieval_generation.*`. New code should import from `retrieval` directly.
"""
from retrieval import (
	retrieve,
	retrieve_and_rank,
	rank,
	graph_to_context,
	Retriever,
	Generator,
	KGBasedRetriever,
	LLMGenerator,
	generate_answer,
	build_prompt,
)

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