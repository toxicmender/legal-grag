"""
Interface for knowledge graph embedding.

Defines the interface for embedding entities and relations into vector spaces.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np


class EmbedderInterface(ABC):
    """
    Abstract interface for knowledge graph embedding.
    
    Defines methods for embedding entities and relations into vector spaces.
    """
    
    @abstractmethod
    def embed_entity(self, entity_id: str, entity_data: Dict[str, Any]) -> np.ndarray:
        """
        Embed a single entity into a vector.
        
        Args:
            entity_id: ID of the entity.
            entity_data: Dictionary containing entity information.
            
        Returns:
            NumPy array representing the entity embedding.
        """
        pass
    
    @abstractmethod
    def embed_entities(self, entities: List[Dict[str, Any]]) -> np.ndarray:
        """
        Embed multiple entities into vectors.
        
        Args:
            entities: List of entity dictionaries.
            
        Returns:
            NumPy array of shape (n_entities, embedding_dim).
        """
        pass
    
    @abstractmethod
    def embed_relation(self, relation_id: str, relation_data: Dict[str, Any]) -> np.ndarray:
        """
        Embed a single relation into a vector.
        
        Args:
            relation_id: ID of the relation.
            relation_data: Dictionary containing relation information.
            
        Returns:
            NumPy array representing the relation embedding.
        """
        pass
    
    @abstractmethod
    def embed_relations(self, relations: List[Dict[str, Any]]) -> np.ndarray:
        """
        Embed multiple relations into vectors.
        
        Args:
            relations: List of relation dictionaries.
            
        Returns:
            NumPy array of shape (n_relations, embedding_dim).
        """
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """
        Get the dimension of embeddings produced by this embedder.
        
        Returns:
            Embedding dimension.
        """
        pass
    
    @abstractmethod
    def fit(self, graph_data: Dict[str, Any]) -> None:
        """
        Fit the embedder to graph data.
        
        Args:
            graph_data: Dictionary containing graph structure and data.
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the embedder model to disk.
        
        Args:
            path: Path to save the model.
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load the embedder model from disk.
        
        Args:
            path: Path to load the model from.
        """
        pass

