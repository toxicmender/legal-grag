"""
Logic to build reasoning chains over subgraphs and user queries using LangChain.

Implements chain-of-thought reasoning with LangChain's capabilities for
multi-step reasoning, agent-based orchestration, and structured output.
"""

from typing import List, Dict, Any, Optional
import os

try:
    from langchain.chat_models import ChatOpenAI, ChatAnthropic
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseLLM = None
    ChatOpenAI = None
    ChatAnthropic = None
    PromptTemplate = None
    LLMChain = None

from .base_prompt import PromptBuilder
from .prompt_config import PromptConfig


class ChainOfThought:
    """
    Implements chain-of-thought reasoning over subgraphs and user queries using LangChain.

    Builds multi-step reasoning chains that leverage graph structure
    and retrieved context. Uses LangChain for orchestration and reasoning.
    """

    def __init__(
        self,
        prompt_builder: Optional[PromptBuilder] = None,
        llm: Optional[Any] = None,
        config: Optional[PromptConfig] = None
    ):
        """
        Initialize the chain-of-thought processor.

        Args:
            prompt_builder: Optional PromptBuilder instance.
            llm: Optional LangChain LLM instance. If not provided, will create from config.
            config: Optional PromptConfig for LLM settings.
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is not installed. Please install it: pip install langchain openai"
            )

        self.prompt_builder = prompt_builder or PromptBuilder()
        self.config = config or PromptConfig()

        # Initialize LLM
        if llm is None:
            self.llm = self._create_llm()
        else:
            self.llm = llm

        # Initialize chains
        self.reasoning_chain = None
        self._build_chains()

    def build_reasoning_chain(
        self,
        query: str,
        subgraphs: List[Dict[str, Any]],
        max_steps: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Build a reasoning chain from query and subgraphs using LangChain.

        Args:
            query: User query.
            subgraphs: List of retrieved subgraphs.
            max_steps: Maximum number of reasoning steps.

        Returns:
            List of reasoning step dictionaries, each containing:
                - step_number: Step number
                - reasoning: Reasoning text
                - entities_used: Entities referenced in this step
                - relations_used: Relations referenced in this step
        """
        # Convert subgraphs to context
        context = self._subgraphs_to_context(subgraphs)

        # Execute reasoning chain
        reasoning_steps = []
        current_context = context
        current_query = query

        for step in range(1, max_steps + 1):
            # Create step-specific prompt (constructed when needed)

            # Execute LLM chain
            try:
                # Use `run` to get the textual response (LLMChain.invoke may return a dict)
                response = self.reasoning_chain.run(
                    step_number=step,
                    query=current_query,
                    context=current_context,
                    previous_steps=self._format_previous_steps(reasoning_steps)
                )

                # Parse response
                step_result = self._parse_reasoning_step(response, step)
                reasoning_steps.append(step_result)

                # Check if we've reached a conclusion
                if step_result.get('is_final', False):
                    break

            except Exception as e:
                print(f"Error in reasoning step {step}: {e}")
                break

        return reasoning_steps

    def extract_reasoning_steps(
        self,
        query: str,
        context: str
    ) -> List[str]:
        """
        Extract reasoning steps from query and context using LangChain.

        Args:
            query: User query.
            context: Context information.

        Returns:
            List of reasoning step strings.
        """
        # Create prompt for step extraction
        extraction_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""Given the following query and context, extract the key reasoning steps needed to answer the query.

Query: {query}

Context: {context}

Extract the reasoning steps as a numbered list. Each step should be clear and specific.
Format:
1. [First step]
2. [Second step]
...

Reasoning steps:"""
        )

        # Create chain
        chain = LLMChain(llm=self.llm, prompt=extraction_prompt)

        # Execute
        result = chain.run(query=query, context=context)

        # Parse steps
        steps = []
        for line in result.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering
                step = line.split('.', 1)[-1].strip() if '.' in line else line[1:].strip()
                if step:
                    steps.append(step)

        return steps

    def build_multi_hop_reasoning(
        self,
        query: str,
        subgraphs: List[Dict[str, Any]],
        hops: int = 2
    ) -> Dict[str, Any]:
        """
        Build multi-hop reasoning across subgraphs using LangChain sequential chains.

        Args:
            query: User query.
            subgraphs: List of retrieved subgraphs.
            hops: Number of reasoning hops.

        Returns:
            Dictionary containing:
                - reasoning_chain: List of reasoning steps
                - path: Path through the graph
                - conclusion: Final conclusion
        """
        # Create sequential chain for multi-hop reasoning
        reasoning_steps = []

        current_context = self._subgraphs_to_context(subgraphs)
        current_query = query

        for hop in range(1, hops + 1):
            # Create hop-specific prompt
            hop_prompt = PromptTemplate(
                input_variables=["query", "context", "hop_number", "previous_reasoning"],
                template="""You are performing multi-hop reasoning step {hop_number} of {total_hops}.

Previous reasoning: {previous_reasoning}

Query: {query}

Context: {context}

Reasoning step {hop_number}: Analyze the context and identify the key information needed for this hop.
Then provide your reasoning and identify what additional information might be needed for the next hop.

Reasoning:"""
            )

            # Create chain for this hop
            hop_chain = LLMChain(
                llm=self.llm,
                prompt=hop_prompt,
                output_key=f"hop_{hop}_reasoning"
            )

            # Execute
            result = hop_chain.run(
                query=current_query,
                context=current_context,
                hop_number=hop,
                total_hops=hops,
                previous_reasoning=self._format_previous_steps(reasoning_steps) if reasoning_steps else "None"
            )

            reasoning_steps.append({
                'hop': hop,
                'reasoning': result,
                'entities_used': self._extract_entities_from_text(result),
                'relations_used': self._extract_relations_from_text(result)
            })

        # Final conclusion step
        conclusion_prompt = PromptTemplate(
            input_variables=["query", "all_reasoning"],
            template="""Based on all the reasoning steps below, provide a final conclusion to answer the query.

Query: {query}

All Reasoning Steps:
{all_reasoning}

Final Conclusion:"""
        )

        conclusion_chain = LLMChain(llm=self.llm, prompt=conclusion_prompt)
        conclusion = conclusion_chain.run(
            query=query,
            all_reasoning=self._format_previous_steps(reasoning_steps)
        )

        return {
            'reasoning_chain': reasoning_steps,
            'path': self._extract_path(reasoning_steps),
            'conclusion': conclusion
        }

    def validate_reasoning_chain(
        self,
        reasoning_chain: List[Dict[str, Any]],
        query: str
    ) -> Dict[str, Any]:
        """
        Validate a reasoning chain for logical consistency using LLM.

        Args:
            reasoning_chain: List of reasoning step dictionaries.
            query: Original query.

        Returns:
            Dictionary containing:
                - is_valid: Boolean indicating validity
                - issues: List of identified issues
                - suggestions: List of improvement suggestions
        """
        validation_prompt = PromptTemplate(
            input_variables=["query", "reasoning_chain"],
            template="""Validate the following reasoning chain for logical consistency and correctness.

Query: {query}

Reasoning Chain:
{reasoning_chain}

Analyze the reasoning chain and provide:
1. Is the reasoning logically consistent? (yes/no)
2. Are there any logical gaps or issues?
3. Suggestions for improvement

Format your response as:
VALID: [yes/no]
ISSUES: [list of issues, or "None" if no issues]
SUGGESTIONS: [list of suggestions, or "None" if no suggestions needed]"""
        )

        chain = LLMChain(llm=self.llm, prompt=validation_prompt)
        result = chain.run(
            query=query,
            reasoning_chain=self._format_previous_steps(reasoning_chain)
        )

        # Parse validation result
        is_valid = "VALID: yes" in result.lower() or "valid: yes" in result.lower()
        issues = self._extract_section(result, "ISSUES:")
        suggestions = self._extract_section(result, "SUGGESTIONS:")

        return {
            'is_valid': is_valid,
            'issues': issues,
            'suggestions': suggestions,
            'raw_response': result
        }

    def refine_reasoning_chain(
        self,
        reasoning_chain: List[Dict[str, Any]],
        feedback: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Refine a reasoning chain based on feedback or validation using LLM.

        Args:
            reasoning_chain: List of reasoning step dictionaries.
            feedback: Optional feedback dictionary with validation results.

        Returns:
            Refined reasoning chain.
        """
        refinement_prompt = PromptTemplate(
            input_variables=["original_chain", "feedback"],
            template="""Refine the following reasoning chain based on the feedback provided.

Original Reasoning Chain:
{original_chain}

Feedback:
{feedback}

Provide a refined version of the reasoning chain that addresses the issues and incorporates the suggestions.
Maintain the same structure but improve the logic and clarity.

Refined Reasoning Chain:"""
        )

        chain = LLMChain(llm=self.llm, prompt=refinement_prompt)

        feedback_text = self._format_feedback(feedback) if feedback else "No specific feedback provided."

        result = chain.run(
            original_chain=self._format_previous_steps(reasoning_chain),
            feedback=feedback_text
        )

        # Parse refined chain (simplified - in practice, you'd want more sophisticated parsing)
        return self._parse_reasoning_chain(result)

    # Helper methods
    def _create_llm(self):
        """Create LLM instance from config."""
        llm_params = self.config.get_llm_params()
        model_name = llm_params.get('model', 'gpt-4')

        # Get API key
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")

        if 'gpt' in model_name.lower() or 'openai' in model_name.lower():
            return ChatOpenAI(
                model_name=model_name,
                temperature=llm_params.get('temperature', 0.7),
                max_tokens=llm_params.get('max_tokens', 2000),
                openai_api_key=api_key
            )
        elif 'claude' in model_name.lower() or 'anthropic' in model_name.lower():
            return ChatAnthropic(
                model=model_name,
                temperature=llm_params.get('temperature', 0.7),
                max_tokens=llm_params.get('max_tokens', 2000),
                anthropic_api_key=api_key
            )
        else:
            # Default to OpenAI
            return ChatOpenAI(
                model_name="gpt-4",
                temperature=0.7,
                openai_api_key=api_key
            )

    def _build_chains(self):
        """Build LangChain chains for reasoning."""
        # Create reasoning chain prompt
        reasoning_prompt = PromptTemplate(
            input_variables=["step_number", "query", "context", "previous_steps"],
            template="""You are performing step {step_number} of a chain-of-thought reasoning process.

Query: {query}

Context: {context}

Previous Steps:
{previous_steps}

For this step, analyze the context and previous reasoning, then provide:
1. Your reasoning for this step
2. Key entities or relations you're using
3. Whether this step leads to a conclusion (yes/no)

Format your response as:
REASONING: [your reasoning]
ENTITIES: [comma-separated list of entities]
RELATIONS: [comma-separated list of relations]
IS_FINAL: [yes/no]"""
        )

        self.reasoning_chain = LLMChain(llm=self.llm, prompt=reasoning_prompt)

    def _create_cot_prompt(self, query: str, context: str, max_steps: int) -> str:
        """Create chain-of-thought prompt."""
        return f"""Answer the following query using chain-of-thought reasoning.

Query: {query}

Context: {context}

Think step by step. You have up to {max_steps} steps to reason through this problem.
After each step, determine if you have enough information to answer the query."""

    def _create_step_prompt(self, step: int, query: str, context: str, previous_steps: List) -> PromptTemplate:
        """Create prompt for a specific reasoning step."""
        return PromptTemplate(
            input_variables=["step", "query", "context", "previous"],
            template=f"""Step {step} Reasoning:

Query: {query}

Context: {context}

Previous Steps:
{{previous}}

Provide your reasoning for step {step}."""
        )

    def _subgraphs_to_context(self, subgraphs: List[Dict[str, Any]]) -> str:
        """Convert subgraphs to context string."""
        context_parts = []
        for i, subgraph in enumerate(subgraphs, 1):
            context_parts.append(f"Subgraph {i}:")
            if 'text' in subgraph:
                context_parts.append(subgraph['text'])
            elif 'nodes' in subgraph and 'edges' in subgraph:
                context_parts.append(f"Nodes: {subgraph['nodes']}")
                context_parts.append(f"Edges: {subgraph['edges']}")
        return "\n".join(context_parts)

    def _parse_reasoning_step(self, response: str, step_number: int) -> Dict[str, Any]:
        """Parse a reasoning step response."""
        reasoning = self._extract_section(response, "REASONING:")
        entities = self._extract_section(response, "ENTITIES:", as_list=True)
        relations = self._extract_section(response, "RELATIONS:", as_list=True)
        is_final = "IS_FINAL: yes" in response.upper() or "is_final: yes" in response.lower()

        return {
            'step_number': step_number,
            'reasoning': reasoning or response,
            'entities_used': entities,
            'relations_used': relations,
            'is_final': is_final
        }

    def _format_previous_steps(self, steps: List[Dict[str, Any]]) -> str:
        """Format previous reasoning steps as string."""
        if not steps:
            return "None"

        formatted = []
        for step in steps:
            step_num = step.get('step_number', step.get('hop', '?'))
            reasoning = step.get('reasoning', '')
            formatted.append(f"Step {step_num}: {reasoning}")

        return "\n".join(formatted)

    def _extract_section(self, text: str, marker: str, as_list: bool = False) -> Any:
        """Extract a section from text."""
        if marker not in text:
            return [] if as_list else ""

        start = text.find(marker) + len(marker)
        end = text.find("\n", start)
        if end == -1:
            end = len(text)

        content = text[start:end].strip()

        if as_list:
            return [item.strip() for item in content.split(',') if item.strip()]
        return content

    def _extract_entities_from_text(self, text: str) -> List[str]:
        """Extract entity mentions from text (simplified)."""
        # In practice, you'd use NER or entity linking
        # This is a simplified version
        entities = []
        if not text:
            return []

        # Look for capitalized words/phrases (defensive: ensure non-empty)
        words = text.split()
        for word in words:
            if not word:
                continue
            if len(word) > 2 and word[0].isupper():
                entities.append(word)
        return list(set(entities))

    def _extract_relations_from_text(self, text: str) -> List[str]:
        """Extract relation mentions from text (simplified)."""
        # Common relation words
        relation_keywords = ['is', 'has', 'contains', 'relates', 'connects', 'links']
        relations = []
        for keyword in relation_keywords:
            if keyword in text.lower():
                relations.append(keyword)
        return relations

    def _extract_path(self, reasoning_steps: List[Dict[str, Any]]) -> List[str]:
        """Extract reasoning path from steps."""
        path = []
        for step in reasoning_steps:
            entities = step.get('entities_used', [])
            relations = step.get('relations_used', [])
            if entities or relations:
                path.append(f"{entities} -> {relations}")
        return path

    def _format_feedback(self, feedback: Dict[str, Any]) -> str:
        """Format feedback dictionary as string."""
        parts = []
        if 'is_valid' in feedback:
            parts.append(f"Valid: {feedback['is_valid']}")
        if 'issues' in feedback:
            parts.append(f"Issues: {feedback['issues']}")
        if 'suggestions' in feedback:
            parts.append(f"Suggestions: {feedback['suggestions']}")
        return "\n".join(parts)

    def _parse_reasoning_chain(self, text: str) -> List[Dict[str, Any]]:
        """Parse a reasoning chain from text (simplified)."""
        steps = []
        current_step = None

        for line in text.split('\n'):
            if line.strip().startswith('Step') or line.strip().startswith('step'):
                if current_step:
                    steps.append(current_step)
                current_step = {'reasoning': line}
            elif current_step:
                current_step['reasoning'] += '\n' + line

        if current_step:
            steps.append(current_step)

        return steps

