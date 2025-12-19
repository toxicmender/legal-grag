"""
Example script demonstrating LangChain-based chain-of-thought and orchestration.

This script shows how to use the orchestrator for end-to-end reasoning
over knowledge graphs.
"""

import os
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prompting.orchestrator import ReasoningOrchestrator
from prompting.prompt_config import PromptConfig
from retrieval.integration import RetrievalIntegration
from retrieval.retriever import SubgraphRetriever
from kg_construction.models import KnowledgeGraph


def example_reasoning_orchestrator(
    query: str,
    graph: Optional[KnowledgeGraph] = None,
    openai_api_key: Optional[str] = None
):
    """
    Example of using ReasoningOrchestrator for end-to-end reasoning.
    
    Args:
        query: User query.
        graph: Optional knowledge graph for retrieval.
        openai_api_key: Optional OpenAI API key.
    """
    print("=" * 60)
    print("LangChain Reasoning Orchestration Example")
    print("=" * 60)
    
    # Set API key
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    elif not os.getenv("OPENAI_API_KEY"):
        print("Error: OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        return None
    
    # Step 1: Set up retrieval (if graph provided)
    print("\n--- Step 1: Setup Retrieval ---")
    retrieval = None
    if graph:
        retriever = SubgraphRetriever(graph=graph, strategy="embedding")
        from retrieval.ranking import SubgraphRanker
        from retrieval.graph_to_context import GraphToContextConverter
        
        ranker = SubgraphRanker()
        converter = GraphToContextConverter()
        retrieval = RetrievalIntegration(
            retriever=retriever,
            ranker=ranker,
            converter=converter
        )
        print("✓ Retrieval integration configured")
    else:
        print("⚠ No graph provided, skipping retrieval setup")
    
    # Step 2: Configure prompt settings
    print("\n--- Step 2: Configure Prompt Settings ---")
    from prompting.prompt_config import LLMConfig
    
    config = PromptConfig(
        llm_config=LLMConfig(
            model_name="gpt-4",
            temperature=0.7,
            max_tokens=2000
        ),
        enable_chain_of_thought=True,
        max_reasoning_steps=5
    )
    print(f"✓ Using model: {config.llm_config.model_name}")
    print(f"✓ Chain-of-thought enabled: {config.enable_chain_of_thought}")
    
    # Step 3: Initialize orchestrator
    print("\n--- Step 3: Initialize Orchestrator ---")
    try:
        orchestrator = ReasoningOrchestrator(
            retrieval_integration=retrieval,
            config=config
        )
        print("✓ Orchestrator initialized")
    except Exception as e:
        print(f"✗ Error initializing orchestrator: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Step 4: Process query
    print("\n--- Step 4: Process Query ---")
    print(f"Query: {query}")
    
    try:
        result = orchestrator.process_query(
            query=query,
            top_k=5,
            enable_cot=True,
            max_reasoning_steps=5
        )
        
        print("\n✓ Query processed successfully!")
        print(f"\nResponse: {result['response']}")
        print(f"\nReasoning Steps: {len(result['reasoning_chain'])}")
        
        # Show reasoning chain
        if result['reasoning_chain']:
            print("\nReasoning Chain:")
            for i, step in enumerate(result['reasoning_chain'], 1):
                print(f"\n  Step {i}:")
                print(f"    Reasoning: {step.get('reasoning', '')[:200]}...")
                if step.get('entities_used'):
                    print(f"    Entities: {step['entities_used']}")
                if step.get('relations_used'):
                    print(f"    Relations: {step['relations_used']}")
        
        # Show validation if available
        if result.get('validation'):
            validation = result['validation']
            print("\nValidation:")
            print(f"  Valid: {validation.get('is_valid', 'Unknown')}")
            if validation.get('issues'):
                print(f"  Issues: {validation['issues']}")
            if validation.get('suggestions'):
                print(f"  Suggestions: {validation['suggestions']}")
        
        return result
        
    except Exception as e:
        print(f"✗ Error processing query: {e}")
        import traceback
        traceback.print_exc()
        return None


def example_multi_hop_reasoning(
    query: str,
    graph: Optional[KnowledgeGraph] = None,
    openai_api_key: Optional[str] = None
):
    """
    Example of multi-hop reasoning.
    
    Args:
        query: User query requiring multi-hop reasoning.
        graph: Optional knowledge graph.
        openai_api_key: Optional OpenAI API key.
    """
    print("=" * 60)
    print("Multi-Hop Reasoning Example")
    print("=" * 60)
    
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    
    # Setup (similar to above)
    config = PromptConfig()
    retrieval = None
    
    if graph:
        retriever = SubgraphRetriever(graph=graph)
        from retrieval.ranking import SubgraphRanker
        from retrieval.graph_to_context import GraphToContextConverter
        
        retrieval = RetrievalIntegration(
            retriever=retriever,
            ranker=SubgraphRanker(),
            converter=GraphToContextConverter()
        )
    
    orchestrator = ReasoningOrchestrator(
        retrieval_integration=retrieval,
        config=config
    )
    
    print(f"\nProcessing multi-hop query: {query}")
    
    try:
        result = orchestrator.process_multi_hop_query(
            query=query,
            hops=3,
            top_k=5
        )
        
        print("\n✓ Multi-hop reasoning completed!")
        print(f"\nFinal Response: {result['response']}")
        print(f"\nReasoning Path: {' -> '.join(result['path'])}")
        print(f"\nConclusion: {result['conclusion']}")
        
        return result
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LangChain orchestration example")
    parser.add_argument("query", help="Query to process")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--multi-hop", action="store_true", help="Use multi-hop reasoning")
    
    args = parser.parse_args()
    
    if args.multi_hop:
        example_multi_hop_reasoning(
            query=args.query,
            openai_api_key=args.api_key
        )
    else:
        example_reasoning_orchestrator(
            query=args.query,
            openai_api_key=args.api_key
        )

