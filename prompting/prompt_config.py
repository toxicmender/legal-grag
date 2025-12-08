"""
Configuration for prompts, LLM settings, temperatures, max tokens, fallback, etc.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    """Configuration for LLM settings."""
    model_name: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: List[str] = field(default_factory=list)
    timeout: int = 60
    retry_attempts: int = 3


@dataclass
class PromptConfig:
    """
    Configuration for prompt engineering and LLM interaction.
    
    Includes LLM settings, fallback strategies, and prompt-specific parameters.
    """
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    fallback_model: Optional[str] = None
    fallback_config: Optional[LLMConfig] = None
    enable_chain_of_thought: bool = True
    max_reasoning_steps: int = 5
    context_window_size: int = 4000
    enable_streaming: bool = False
    cache_responses: bool = True
    prompt_templates: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize fallback config if fallback model is specified."""
        if self.fallback_model and not self.fallback_config:
            self.fallback_config = LLMConfig(
                model_name=self.fallback_model,
                temperature=self.llm_config.temperature,
                max_tokens=self.llm_config.max_tokens
            )
    
    def get_llm_params(self) -> Dict[str, Any]:
        """
        Get LLM parameters as a dictionary.
        
        Returns:
            Dictionary of LLM parameters.
        """
        return {
            'model': self.llm_config.model_name,
            'temperature': self.llm_config.temperature,
            'max_tokens': self.llm_config.max_tokens,
            'top_p': self.llm_config.top_p,
            'frequency_penalty': self.llm_config.frequency_penalty,
            'presence_penalty': self.llm_config.presence_penalty,
            'stop': self.llm_config.stop_sequences if self.llm_config.stop_sequences else None,
        }
    
    def get_fallback_params(self) -> Optional[Dict[str, Any]]:
        """
        Get fallback LLM parameters as a dictionary.
        
        Returns:
            Dictionary of fallback LLM parameters or None.
        """
        if not self.fallback_config:
            return None
        
        return {
            'model': self.fallback_config.model_name,
            'temperature': self.fallback_config.temperature,
            'max_tokens': self.fallback_config.max_tokens,
            'top_p': self.fallback_config.top_p,
            'frequency_penalty': self.fallback_config.frequency_penalty,
            'presence_penalty': self.fallback_config.presence_penalty,
            'stop': self.fallback_config.stop_sequences if self.fallback_config.stop_sequences else None,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PromptConfig':
        """
        Create PromptConfig from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration.
            
        Returns:
            PromptConfig instance.
        """
        llm_config = LLMConfig(**config_dict.get('llm_config', {}))
        fallback_config = None
        if 'fallback_config' in config_dict:
            fallback_config = LLMConfig(**config_dict['fallback_config'])
        
        return cls(
            llm_config=llm_config,
            fallback_model=config_dict.get('fallback_model'),
            fallback_config=fallback_config,
            enable_chain_of_thought=config_dict.get('enable_chain_of_thought', True),
            max_reasoning_steps=config_dict.get('max_reasoning_steps', 5),
            context_window_size=config_dict.get('context_window_size', 4000),
            enable_streaming=config_dict.get('enable_streaming', False),
            cache_responses=config_dict.get('cache_responses', True),
            prompt_templates=config_dict.get('prompt_templates', {})
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert PromptConfig to a dictionary.
        
        Returns:
            Dictionary representation of the configuration.
        """
        result = {
            'llm_config': {
                'model_name': self.llm_config.model_name,
                'temperature': self.llm_config.temperature,
                'max_tokens': self.llm_config.max_tokens,
                'top_p': self.llm_config.top_p,
                'frequency_penalty': self.llm_config.frequency_penalty,
                'presence_penalty': self.llm_config.presence_penalty,
                'stop_sequences': self.llm_config.stop_sequences,
                'timeout': self.llm_config.timeout,
                'retry_attempts': self.llm_config.retry_attempts,
            },
            'fallback_model': self.fallback_model,
            'enable_chain_of_thought': self.enable_chain_of_thought,
            'max_reasoning_steps': self.max_reasoning_steps,
            'context_window_size': self.context_window_size,
            'enable_streaming': self.enable_streaming,
            'cache_responses': self.cache_responses,
            'prompt_templates': self.prompt_templates,
        }
        
        if self.fallback_config:
            result['fallback_config'] = {
                'model_name': self.fallback_config.model_name,
                'temperature': self.fallback_config.temperature,
                'max_tokens': self.fallback_config.max_tokens,
                'top_p': self.fallback_config.top_p,
                'frequency_penalty': self.fallback_config.frequency_penalty,
                'presence_penalty': self.fallback_config.presence_penalty,
                'stop_sequences': self.fallback_config.stop_sequences,
                'timeout': self.fallback_config.timeout,
                'retry_attempts': self.fallback_config.retry_attempts,
            }
        
        return result

