"""
Base prompt templates and prompt-builder utilities.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from string import Template


@dataclass
class PromptTemplate:
    """
    Represents a prompt template.
    """
    template: str
    variables: List[str]
    description: Optional[str] = None
    
    def format(self, **kwargs) -> str:
        """
        Format the template with provided variables.
        
        Args:
            **kwargs: Variables to substitute in the template.
            
        Returns:
            Formatted prompt string.
        """
        template_obj = Template(self.template)
        return template_obj.safe_substitute(**kwargs)


class PromptBuilder:
    """
    Utility class for building prompts from templates.
    
    Provides methods for constructing prompts with context, instructions,
    and examples.
    """
    
    def __init__(self):
        """Initialize the prompt builder."""
        self.templates: Dict[str, PromptTemplate] = {}
        self._register_default_templates()
    
    def _register_default_templates(self) -> None:
        """Register default prompt templates."""
        # TODO: Register default templates
        pass
    
    def register_template(self, name: str, template: PromptTemplate) -> None:
        """
        Register a prompt template.
        
        Args:
            name: Name of the template.
            template: PromptTemplate object.
        """
        self.templates[name] = template
    
    def build(
        self, 
        template_name: str, 
        context: Optional[str] = None,
        query: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Build a prompt from a template.
        
        Args:
            template_name: Name of the template to use.
            context: Optional context to include in the prompt.
            query: Optional user query.
            **kwargs: Additional variables for template substitution.
            
        Returns:
            Formatted prompt string.
        """
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        template = self.templates[template_name]
        
        # Add context and query to kwargs if provided
        if context:
            kwargs['context'] = context
        if query:
            kwargs['query'] = query
        
        return template.format(**kwargs)
    
    def build_with_context(
        self,
        query: str,
        context: str,
        instructions: Optional[str] = None,
        examples: Optional[List[str]] = None
    ) -> str:
        """
        Build a prompt with context, query, instructions, and examples.
        
        Args:
            query: User query.
            context: Context information (e.g., retrieved subgraph).
            instructions: Optional instructions for the LLM.
            examples: Optional list of example interactions.
            
        Returns:
            Complete prompt string.
        """
        prompt_parts = []
        
        if instructions:
            prompt_parts.append(f"Instructions: {instructions}\n")
        
        if context:
            prompt_parts.append(f"Context:\n{context}\n")
        
        if examples:
            prompt_parts.append("Examples:\n")
            for i, example in enumerate(examples, 1):
                prompt_parts.append(f"{i}. {example}\n")
        
        prompt_parts.append(f"Query: {query}\n")
        prompt_parts.append("Response:")
        
        return "\n".join(prompt_parts)
    
    def build_qa_prompt(self, question: str, context: str) -> str:
        """
        Build a question-answering prompt.
        
        Args:
            question: User question.
            context: Context information.
            
        Returns:
            QA prompt string.
        """
        return self.build_with_context(
            query=question,
            context=context,
            instructions="Answer the question based on the provided context. "
                        "If the context does not contain enough information, "
                        "say so explicitly."
        )
    
    def build_reasoning_prompt(
        self, 
        query: str, 
        context: str,
        reasoning_steps: Optional[List[str]] = None
    ) -> str:
        """
        Build a reasoning prompt with chain-of-thought.
        
        Args:
            query: User query.
            context: Context information.
            reasoning_steps: Optional list of reasoning steps to include.
            
        Returns:
            Reasoning prompt string.
        """
        instructions = "Think step by step. Use the provided context to reason about the query."
        
        if reasoning_steps:
            instructions += "\n\nReasoning steps:\n"
            for i, step in enumerate(reasoning_steps, 1):
                instructions += f"{i}. {step}\n"
        
        return self.build_with_context(
            query=query,
            context=context,
            instructions=instructions
        )

