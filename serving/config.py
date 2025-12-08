"""
Configuration: endpoints, database URI, LLM config, storage paths, etc.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass
class DatabaseConfig:
    """Database configuration."""
    uri: str = "neo4j://localhost:7687"
    username: Optional[str] = None
    password: Optional[str] = None
    database: Optional[str] = None


@dataclass
class LLMConfig:
    """LLM configuration."""
    provider: str = "openai"  # 'openai', 'anthropic', 'local', etc.
    model_name: str = "gpt-4"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000


@dataclass
class StorageConfig:
    """Storage configuration."""
    cache_dir: str = ".cache"
    embeddings_dir: str = ".cache/embeddings"
    models_dir: str = ".cache/models"
    data_dir: str = "data"


@dataclass
class ServerConfig:
    """
    Server configuration.
    
    Includes endpoints, database URI, LLM config, storage paths, etc.
    """
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # Database configuration
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    
    # LLM configuration
    llm: LLMConfig = field(default_factory=LLMConfig)
    
    # Storage configuration
    storage: StorageConfig = field(default_factory=StorageConfig)
    
    # API endpoints
    api_prefix: str = "/api/v1"
    
    # Other settings
    max_upload_size: int = 100 * 1024 * 1024  # 100 MB
    enable_cors: bool = True
    cors_origins: list = field(default_factory=lambda: ["*"])
    
    def __post_init__(self):
        """Initialize configuration from environment variables if available."""
        # Database
        if os.getenv("NEO4J_URI"):
            self.database.uri = os.getenv("NEO4J_URI")
        if os.getenv("NEO4J_USERNAME"):
            self.database.username = os.getenv("NEO4J_USERNAME")
        if os.getenv("NEO4J_PASSWORD"):
            self.database.password = os.getenv("NEO4J_PASSWORD")
        
        # LLM
        if os.getenv("LLM_PROVIDER"):
            self.llm.provider = os.getenv("LLM_PROVIDER")
        if os.getenv("LLM_MODEL"):
            self.llm.model_name = os.getenv("LLM_MODEL")
        if os.getenv("LLM_API_KEY"):
            self.llm.api_key = os.getenv("LLM_API_KEY")
        if os.getenv("LLM_BASE_URL"):
            self.llm.base_url = os.getenv("LLM_BASE_URL")
        
        # Storage
        if os.getenv("CACHE_DIR"):
            self.storage.cache_dir = os.getenv("CACHE_DIR")
        if os.getenv("DATA_DIR"):
            self.storage.data_dir = os.getenv("DATA_DIR")
        
        # Server
        if os.getenv("SERVER_HOST"):
            self.host = os.getenv("SERVER_HOST")
        if os.getenv("SERVER_PORT"):
            self.port = int(os.getenv("SERVER_PORT"))
        if os.getenv("DEBUG"):
            self.debug = os.getenv("DEBUG").lower() == "true"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ServerConfig':
        """
        Create ServerConfig from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration.
            
        Returns:
            ServerConfig instance.
        """
        database_config = DatabaseConfig(**config_dict.get('database', {}))
        llm_config = LLMConfig(**config_dict.get('llm', {}))
        storage_config = StorageConfig(**config_dict.get('storage', {}))
        
        return cls(
            host=config_dict.get('host', '0.0.0.0'),
            port=config_dict.get('port', 8000),
            debug=config_dict.get('debug', False),
            database=database_config,
            llm=llm_config,
            storage=storage_config,
            api_prefix=config_dict.get('api_prefix', '/api/v1'),
            max_upload_size=config_dict.get('max_upload_size', 100 * 1024 * 1024),
            enable_cors=config_dict.get('enable_cors', True),
            cors_origins=config_dict.get('cors_origins', ['*'])
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert ServerConfig to a dictionary.
        
        Returns:
            Dictionary representation of the configuration.
        """
        return {
            'host': self.host,
            'port': self.port,
            'debug': self.debug,
            'database': {
                'uri': self.database.uri,
                'username': self.database.username,
                'password': self.database.password,
                'database': self.database.database,
            },
            'llm': {
                'provider': self.llm.provider,
                'model_name': self.llm.model_name,
                'api_key': self.llm.api_key,
                'base_url': self.llm.base_url,
                'temperature': self.llm.temperature,
                'max_tokens': self.llm.max_tokens,
            },
            'storage': {
                'cache_dir': self.storage.cache_dir,
                'embeddings_dir': self.storage.embeddings_dir,
                'models_dir': self.storage.models_dir,
                'data_dir': self.storage.data_dir,
            },
            'api_prefix': self.api_prefix,
            'max_upload_size': self.max_upload_size,
            'enable_cors': self.enable_cors,
            'cors_origins': self.cors_origins,
        }

