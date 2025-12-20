"""
Glue code integrating retrieval with LLM and prompt chain.
"""

from typing import Dict, Any, Optional
from .retriever import SubgraphRetriever
from .ranking import SubgraphRanker
from .graph_to_context import GraphToContextConverter


class RetrievalIntegration:
    """
    Integrates retrieval pipeline with LLM and prompt chain.
    
    Orchestrates the full retrieval pipeline:
    1. Query processing
    2. Subgraph retrieval
    3. Ranking
    4. Context conversion
    5. Integration with LLM prompts
    """
    
    def __init__(
        self,
        retriever: Optional[SubgraphRetriever] = None,
        ranker: Optional[SubgraphRanker] = None,
        converter: Optional[GraphToContextConverter] = None
    ):
        """
        Initialize the retrieval integration.
        
        Args:
            retriever: Optional SubgraphRetriever instance.
            ranker: Optional SubgraphRanker instance.
            converter: Optional GraphToContextConverter instance.
        """
        self.retriever = retriever or SubgraphRetriever()
        self.ranker = ranker or SubgraphRanker()
        self.converter = converter or GraphToContextConverter()
    
    def retrieve_context(
        self, 
        query: str, 
        top_k: int = 10,
        return_format: str = "natural_language"
    ) -> Dict[str, Any]:
        """
        Retrieve and convert subgraph context for a query.
        
        Args:
            query: User query string.
            top_k: Number of subgraphs to retrieve.
            return_format: Format for context conversion.
            
        Returns:
            Dictionary containing:
                - context: Textual context
                - subgraphs: List of retrieved subgraphs
                - metadata: Additional metadata
        """
        # Retrieve subgraphs
        subgraphs = self.retriever.retrieve(query, top_k=top_k)
        
        # Rank subgraphs
        ranked_subgraphs = self.ranker.rank_subgraphs(subgraphs, query=query)
        
        # Convert to context
        if return_format == "natural_language":
            context = self.converter.convert_to_natural_language(ranked_subgraphs[0])
        elif return_format == "structured":
            context = self.converter.convert_to_structured(ranked_subgraphs[0])
        elif return_format == "triples":
            context = self.converter.convert_to_triples(ranked_subgraphs[0])
        else:
            context = self.converter.convert(ranked_subgraphs[0])
        
        return {
            'context': context,
            'subgraphs': ranked_subgraphs,
            'metadata': {
                'query': query,
                'num_subgraphs': len(ranked_subgraphs),
                'format': return_format
            }
        }
    
    def prepare_prompt_context(
        self, 
        query: str, 
        top_k: int = 10
    ) -> str:
        """
        Prepare prompt-ready context for LLM.
        
        Args:
            query: User query string.
            top_k: Number of subgraphs to retrieve.
            
        Returns:
            Formatted context string ready for LLM prompt.
        """
        result = self.retrieve_context(query, top_k=top_k)
        return result['context']
    
    def integrate_with_llm(
        self, 
        query: str, 
        llm_client,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Integrate retrieval with LLM to generate response.
        
        Args:
            query: User query string.
            llm_client: LLM client instance.
            top_k: Number of subgraphs to retrieve.
            
        Returns:
            Dictionary containing:
                - response: LLM-generated response
                - context: Retrieved context
                - metadata: Additional metadata
        """
        # Retrieve context
        self.prepare_prompt_context(query, top_k=top_k)
        
        # TODO: Integrate with LLM to generate response
        # This would typically involve:
        # 1. Building prompt with context
        # 2. Calling LLM
        # 3. Processing response
        
        raise NotImplementedError("LLM integration not yet implemented")

