"""
Fallback embedder using other KGE libraries (e.g., Pykg2vec) for baseline.
"""

from typing import List, Dict, Any
import numpy as np
from .embedder_interface import EmbedderInterface


class FallbackEmbedder(EmbedderInterface):
    """
    Fallback embedder using other KGE libraries.
    
    Can use libraries like Pykg2vec, DGL-KE, or other knowledge graph
    embedding libraries as a baseline or fallback option.
    """
    
    def __init__(self, library: str = "pykg2vec", model_name: str = "TransE", **kwargs):
        """
        Initialize the fallback embedder.
        
        Args:
            library: Name of the KGE library to use ('pykg2vec', 'dgl-ke', etc.).
            model_name: Name of the model to use (e.g., 'TransE', 'DistMult', 'ComplEx').
            **kwargs: Additional parameters for the model.
        """
        self.library = library
        self.model_name = model_name
        self.model = None
        self.embedding_dim = kwargs.get('embedding_dim', 128)
    
    def embed_entity(self, entity_id: str, entity_data: Dict[str, Any]) -> np.ndarray:
        """
        Embed a single entity using the fallback library.
        
        Args:
            entity_id: ID of the entity.
            entity_data: Dictionary containing entity information.
            
        Returns:
            NumPy array representing the entity embedding.
        """
        # TODO: Implement fallback entity embedding
        raise NotImplementedError("Fallback entity embedding not yet implemented")
    
    def embed_entities(self, entities: List[Dict[str, Any]]) -> np.ndarray:
        """
        Embed multiple entities using the fallback library.
        
        Args:
            entities: List of entity dictionaries.
            
        Returns:
            NumPy array of shape (n_entities, embedding_dim).
        """
        # TODO: Implement fallback batch entity embedding
        raise NotImplementedError("Fallback batch entity embedding not yet implemented")
    
    def embed_relation(self, relation_id: str, relation_data: Dict[str, Any]) -> np.ndarray:
        """
        Embed a single relation using the fallback library.
        
        Args:
            relation_id: ID of the relation.
            relation_data: Dictionary containing relation information.
            
        Returns:
            NumPy array representing the relation embedding.
        """
        # TODO: Implement fallback relation embedding
        raise NotImplementedError("Fallback relation embedding not yet implemented")
    
    def embed_relations(self, relations: List[Dict[str, Any]]) -> np.ndarray:
        """
        Embed multiple relations using the fallback library.
        
        Args:
            relations: List of relation dictionaries.
            
        Returns:
            NumPy array of shape (n_relations, embedding_dim).
        """
        # TODO: Implement fallback batch relation embedding
        raise NotImplementedError("Fallback batch relation embedding not yet implemented")
    
    def get_embedding_dim(self) -> int:
        """
        Get the dimension of embeddings produced by the fallback embedder.
        
        Returns:
            Embedding dimension.
        """
        return self.embedding_dim
    
    def fit(self, graph_data: Dict[str, Any]) -> None:
        """
        Fit the fallback model to graph data.
        
        Args:
            graph_data: Dictionary containing graph structure and data.
        """
        # TODO: Implement fallback model fitting
        raise NotImplementedError("Fallback model fitting not yet implemented")
    
    def save(self, path: str) -> None:
        """
        Save the fallback model to disk.
        
        Args:
            path: Path to save the model.
        """
        # TODO: Implement fallback model saving
        raise NotImplementedError("Fallback model saving not yet implemented")
    
    def load(self, path: str) -> None:
        """
        Load the fallback model from disk.
        
        Args:
            path: Path to load the model from.
        """
        # TODO: Implement fallback model loading
        raise NotImplementedError("Fallback model loading not yet implemented")

