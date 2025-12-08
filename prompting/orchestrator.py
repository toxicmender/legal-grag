"""
Orchestration module using LangChain for end-to-end reasoning pipeline.

Coordinates retrieval, prompting, chain-of-thought reasoning, and response generation.
"""

from typing import Dict, Any, Optional, List, Callable
import os

try:
    from langchain.chains import SequentialChain, LLMChain
    from langchain.prompts import PromptTemplate, ChatPromptTemplate
    from langchain.agents import AgentExecutor, create_openai_functions_agent
    from langchain.tools import BaseTool, Tool
    from langchain.schema import BaseMessage
    from langchain.callbacks import BaseCallbackHandler
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from .chain_of_thought import ChainOfThought
from .base_prompt import PromptBuilder
from .prompt_config import PromptConfig
from retrieval.integration import RetrievalIntegration


class ReasoningOrchestrator:
    """
    Orchestrates the complete reasoning pipeline using LangChain.
    
    Coordinates:
    1. Query understanding
    2. Context retrieval
    3. Chain-of-thought reasoning
    4. Response generation
    5. Validation and refinement
    """
    
    def __init__(
        self,
        retrieval_integration: Optional[RetrievalIntegration] = None,
        chain_of_thought: Optional[ChainOfThought] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        config: Optional[PromptConfig] = None
    ):
        """
        Initialize the orchestrator.
        
        Args:
            retrieval_integration: Optional RetrievalIntegration instance.
            chain_of_thought: Optional ChainOfThought instance.
            prompt_builder: Optional PromptBuilder instance.
            config: Optional PromptConfig instance.
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is not installed. Please install: pip install langchain openai"
            )
        
        self.retrieval = retrieval_integration
        self.cot = chain_of_thought or ChainOfThought(prompt_builder=prompt_builder, config=config)
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.config = config or PromptConfig()
        
        # Build orchestration chain
        self._build_orchestration_chain()
    
    def process_query(
        self,
        query: str,
        top_k: int = 10,
        enable_cot: bool = True,
        max_reasoning_steps: int = 5
    ) -> Dict[str, Any]:
        """
        Process a user query through the complete reasoning pipeline.
        
        Args:
            query: User query string.
            top_k: Number of subgraphs to retrieve.
            enable_cot: Whether to use chain-of-thought reasoning.
            max_reasoning_steps: Maximum reasoning steps.
            
        Returns:
            Dictionary containing:
                - response: Final response
                - reasoning_chain: Chain-of-thought steps
                - context: Retrieved context
                - metadata: Additional metadata
        """
        # Step 1: Retrieve context
        if self.retrieval:
            retrieval_result = self.retrieval.retrieve_context(query, top_k=top_k)
            context = retrieval_result['context']
            subgraphs = retrieval_result['subgraphs']
        else:
            context = ""
            subgraphs = []
        
        # Step 2: Chain-of-thought reasoning
        if enable_cot and self.config.enable_chain_of_thought:
            reasoning_chain = self.cot.build_reasoning_chain(
                query=query,
                subgraphs=subgraphs,
                max_steps=max_reasoning_steps
            )
        else:
            reasoning_chain = []
        
        # Step 3: Generate response
        response = self._generate_response(query, context, reasoning_chain)
        
        # Step 4: Validate (optional)
        if reasoning_chain:
            validation = self.cot.validate_reasoning_chain(reasoning_chain, query)
            
            # Refine if needed
            if not validation.get('is_valid', True) and validation.get('suggestions'):
                reasoning_chain = self.cot.refine_reasoning_chain(
                    reasoning_chain,
                    feedback=validation
                )
                # Regenerate response with refined reasoning
                response = self._generate_response(query, context, reasoning_chain)
        else:
            validation = None
        
        return {
            'response': response,
            'reasoning_chain': reasoning_chain,
            'context': context,
            'subgraphs': subgraphs,
            'validation': validation,
            'metadata': {
                'query': query,
                'top_k': top_k,
                'reasoning_steps': len(reasoning_chain),
                'used_cot': enable_cot
            }
        }
    
    def process_multi_hop_query(
        self,
        query: str,
        hops: int = 2,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Process a multi-hop reasoning query.
        
        Args:
            query: User query.
            hops: Number of reasoning hops.
            top_k: Number of subgraphs to retrieve per hop.
            
        Returns:
            Dictionary with multi-hop reasoning results.
        """
        if not self.retrieval:
            raise ValueError("Retrieval integration required for multi-hop reasoning")
        
        # Retrieve initial context
        retrieval_result = self.retrieval.retrieve_context(query, top_k=top_k)
        subgraphs = retrieval_result['subgraphs']
        
        # Perform multi-hop reasoning
        multi_hop_result = self.cot.build_multi_hop_reasoning(
            query=query,
            subgraphs=subgraphs,
            hops=hops
        )
        
        # Generate final response
        final_response = self._generate_response(
            query=query,
            context=multi_hop_result['conclusion'],
            reasoning_chain=multi_hop_result['reasoning_chain']
        )
        
        return {
            'response': final_response,
            'reasoning_chain': multi_hop_result['reasoning_chain'],
            'path': multi_hop_result['path'],
            'conclusion': multi_hop_result['conclusion'],
            'metadata': {
                'query': query,
                'hops': hops,
                'top_k': top_k
            }
        }
    
    def _build_orchestration_chain(self):
        """Build the main orchestration chain."""
        # This can be extended with more sophisticated chains
        pass
    
    def _generate_response(
        self,
        query: str,
        context: str,
        reasoning_chain: List[Dict[str, Any]]
    ) -> str:
        """Generate final response from query, context, and reasoning."""
        # Build prompt with reasoning
        if reasoning_chain:
            reasoning_text = self.cot._format_previous_steps(reasoning_chain)
            prompt = self.prompt_builder.build_reasoning_prompt(
                query=query,
                context=context,
                reasoning_steps=[step.get('reasoning', '') for step in reasoning_chain]
            )
        else:
            prompt = self.prompt_builder.build_qa_prompt(
                question=query,
                context=context
            )
        
        # Generate response using LLM
        try:
            response = self.cot.llm.invoke(prompt).content
            return response
        except AttributeError:
            # Fallback for different LLM interfaces
            response = self.cot.llm(prompt)
            return response if isinstance(response, str) else str(response)


class AgentOrchestrator:
    """
    Agent-based orchestrator using LangChain agents for more dynamic reasoning.
    
    Uses LangChain agents with tools for flexible reasoning and information gathering.
    """
    
    def __init__(
        self,
        tools: List[BaseTool],
        retrieval_integration: Optional[RetrievalIntegration] = None,
        config: Optional[PromptConfig] = None
    ):
        """
        Initialize agent-based orchestrator.
        
        Args:
            tools: List of LangChain tools for the agent.
            retrieval_integration: Optional RetrievalIntegration for context retrieval.
            config: Optional PromptConfig.
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required for AgentOrchestrator")
        
        self.tools = tools
        self.retrieval = retrieval_integration
        self.config = config or PromptConfig()
        
        # Create agent
        self.agent = self._create_agent()
    
    def _create_agent(self) -> AgentExecutor:
        """Create LangChain agent with tools."""
        from langchain.chat_models import ChatOpenAI
        
        llm = ChatOpenAI(
            model_name=self.config.llm_config.model_name,
            temperature=self.config.llm_config.temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create agent prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that reasons about legal knowledge graphs."),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create agent
        agent = create_openai_functions_agent(llm, self.tools, prompt)
        
        # Create executor
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=self.config.max_reasoning_steps
        )
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process query using agent-based reasoning.
        
        Args:
            query: User query.
            
        Returns:
            Dictionary with agent response and metadata.
        """
        result = self.agent.invoke({"input": query})
        
        return {
            'response': result.get('output', ''),
            'intermediate_steps': result.get('intermediate_steps', []),
            'metadata': {
                'query': query,
                'agent_type': 'openai_functions'
            }
        }

