"""
Wrapper for MAYPL (Structure Is All You Need) algorithm for graph representation learning.

MAYPL is a structural representation learning method for hyper-relational knowledge graphs.
Repository: https://github.com/bdi-lab/MAYPL.git
Paper: "Structure Is All You Need: Structural Representation Learning on Hyper-Relational Knowledge Graphs"
ICML 2025
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import os
import sys
import torch
from .embedder_interface import EmbedderInterface

# Try to import MAYPL modules
MAYPL_AVAILABLE = False
try:
    # MAYPL code structure - adjust paths as needed
    # Assuming MAYPL is cloned to a maypl directory or installed
    maypl_path = os.getenv("MAYPL_PATH", None)
    if maypl_path:
        sys.path.insert(0, maypl_path)
    
    # Try importing MAYPL modules (adjust based on actual structure)
    # These imports may need adjustment based on MAYPL's actual code structure
    try:
        from code.model import MAYPL  # Adjust import path as needed
        from code.data_loader import DataLoader  # Adjust import path as needed
        MAYPL_AVAILABLE = True
    except ImportError:
        # Alternative import paths
        try:
            import maypl
            from maypl.model import MAYPL
            from maypl.data_loader import DataLoader
            MAYPL_AVAILABLE = True
        except ImportError:
            pass
except Exception:
    pass


class MAYPLWrapper(EmbedderInterface):
    """
    Wrapper around MAYPL algorithm for graph representation learning.
    
    MAYPL (Structure Is All You Need) is a structural representation learning method
    for hyper-relational knowledge graphs. This wrapper integrates MAYPL with our
    knowledge graph format.
    
    Reference: https://github.com/bdi-lab/MAYPL.git
    """
    
    def __init__(
        self, 
        embedding_dim: int = 128,
        maypl_path: Optional[str] = None,
        device: str = "cpu",
        **kwargs
    ):
        """
        Initialize the MAYPL embedder.
        
        Args:
            embedding_dim: Dimension of embeddings to produce.
            maypl_path: Optional path to MAYPL repository/code.
            device: Device to run on ('cpu' or 'cuda').
            **kwargs: Additional parameters for MAYPL algorithm.
        """
        if not MAYPL_AVAILABLE:
            raise ImportError(
                "MAYPL is not available. Please:\n"
                "1. Clone MAYPL repository: git clone https://github.com/bdi-lab/MAYPL.git\n"
                "2. Set MAYPL_PATH environment variable to the MAYPL code directory\n"
                "3. Install MAYPL requirements: pip install -r requirements.txt"
            )
        
        self.embedding_dim = embedding_dim
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = None
        self.entity_embeddings: Dict[str, np.ndarray] = {}
        self.relation_embeddings: Dict[str, np.ndarray] = {}
        self.entity_id_to_idx: Dict[str, int] = {}
        self.relation_id_to_idx: Dict[str, int] = {}
        self.idx_to_entity_id: Dict[int, str] = {}
        self.idx_to_relation_id: Dict[int, str] = {}
        self.is_trained = False
        self.maypl_path = maypl_path
        self.kwargs = kwargs
    
    def embed_entity(self, entity_id: str, entity_data: Dict[str, Any]) -> np.ndarray:
        """
        Embed a single entity using MAYPL.
        
        Args:
            entity_id: ID of the entity.
            entity_data: Dictionary containing entity information.
            
        Returns:
            NumPy array representing the entity embedding.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before embedding entities. Call fit() first.")
        
        if entity_id in self.entity_embeddings:
            return self.entity_embeddings[entity_id]
        
        # Get entity index
        entity_idx = self.entity_id_to_idx.get(entity_id)
        if entity_idx is None:
            raise ValueError(f"Entity {entity_id} not found in trained model")
        
        # Extract embedding from model
        if self.model is not None:
            with torch.no_grad():
                # MAYPL typically stores entity embeddings in model.ent_emb or similar
                # Adjust based on actual MAYPL model structure
                if hasattr(self.model, 'ent_emb'):
                    embedding = self.model.ent_emb[entity_idx].cpu().numpy()
                elif hasattr(self.model, 'entity_embedding'):
                    embedding = self.model.entity_embedding[entity_idx].cpu().numpy()
                else:
                    # Fallback: try to get from model parameters
                    embeddings = list(self.model.parameters())[0]
                    embedding = embeddings[entity_idx].cpu().detach().numpy()
                
                self.entity_embeddings[entity_id] = embedding
                return embedding
        else:
            raise RuntimeError("Model not loaded")
    
    def embed_entities(self, entities: List[Dict[str, Any]]) -> np.ndarray:
        """
        Embed multiple entities using MAYPL.
        
        Args:
            entities: List of entity dictionaries with 'id' field.
            
        Returns:
            NumPy array of shape (n_entities, embedding_dim).
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before embedding entities. Call fit() first.")
        
        embeddings = []
        for entity in entities:
            entity_id = entity.get('id', entity.get('entity_id'))
            if entity_id:
                embedding = self.embed_entity(entity_id, entity)
                embeddings.append(embedding)
            else:
                raise ValueError("Entity dictionary must contain 'id' or 'entity_id' field")
        
        return np.array(embeddings)
    
    def embed_relation(self, relation_id: str, relation_data: Dict[str, Any]) -> np.ndarray:
        """
        Embed a single relation using MAYPL.
        
        Args:
            relation_id: ID of the relation.
            relation_data: Dictionary containing relation information.
            
        Returns:
            NumPy array representing the relation embedding.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before embedding relations. Call fit() first.")
        
        if relation_id in self.relation_embeddings:
            return self.relation_embeddings[relation_id]
        
        # Get relation index
        relation_idx = self.relation_id_to_idx.get(relation_id)
        if relation_idx is None:
            raise ValueError(f"Relation {relation_id} not found in trained model")
        
        # Extract embedding from model
        if self.model is not None:
            with torch.no_grad():
                # MAYPL typically stores relation embeddings in model.rel_emb or similar
                # Adjust based on actual MAYPL model structure
                if hasattr(self.model, 'rel_emb'):
                    embedding = self.model.rel_emb[relation_idx].cpu().numpy()
                elif hasattr(self.model, 'relation_embedding'):
                    embedding = self.model.relation_embedding[relation_idx].cpu().numpy()
                else:
                    # Fallback: try to get from model parameters
                    params = list(self.model.parameters())
                    if len(params) > 1:
                        embedding = params[1][relation_idx].cpu().detach().numpy()
                    else:
                        raise RuntimeError("Could not find relation embeddings in model")
                
                self.relation_embeddings[relation_id] = embedding
                return embedding
        else:
            raise RuntimeError("Model not loaded")
    
    def embed_relations(self, relations: List[Dict[str, Any]]) -> np.ndarray:
        """
        Embed multiple relations using MAYPL.
        
        Args:
            relations: List of relation dictionaries with 'id' field.
            
        Returns:
            NumPy array of shape (n_relations, embedding_dim).
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before embedding relations. Call fit() first.")
        
        embeddings = []
        for relation in relations:
            relation_id = relation.get('id', relation.get('relation_id'))
            if relation_id:
                embedding = self.embed_relation(relation_id, relation)
                embeddings.append(embedding)
            else:
                raise ValueError("Relation dictionary must contain 'id' or 'relation_id' field")
        
        return np.array(embeddings)
    
    def get_embedding_dim(self) -> int:
        """
        Get the dimension of embeddings produced by MAYPL.
        
        Returns:
            Embedding dimension.
        """
        return self.embedding_dim
    
    def fit(
        self, 
        graph_data: Dict[str, Any],
        train_epochs: int = 100,
        batch_size: int = 512,
        learning_rate: float = 0.001,
        **training_kwargs
    ) -> None:
        """
        Fit MAYPL model to graph data.
        
        Args:
            graph_data: Dictionary containing:
                - 'entities': List of Entity objects or dicts
                - 'relations': List of Relation objects or dicts
                - 'triples': Optional list of (head, relation, tail) tuples
            train_epochs: Number of training epochs.
            batch_size: Batch size for training.
            learning_rate: Learning rate for optimizer.
            **training_kwargs: Additional training parameters.
        """
        # Convert graph data to MAYPL format
        triples, qualifiers = self._convert_to_maypl_format(graph_data)
        
        # Build entity and relation mappings
        self._build_mappings(graph_data)
        
        # Initialize MAYPL model
        num_entities = len(self.entity_id_to_idx)
        num_relations = len(self.relation_id_to_idx)
        
        # Initialize MAYPL model (adjust parameters based on actual MAYPL API)
        try:
            self.model = MAYPL(
                num_entities=num_entities,
                num_relations=num_relations,
                embedding_dim=self.embedding_dim,
                **self.kwargs
            ).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MAYPL model: {e}")
        
        # Prepare data loader
        # MAYPL expects data in specific format - adjust based on actual implementation
        train_data = self._prepare_training_data(triples, qualifiers)
        
        # Train the model
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        print(f"Training MAYPL model on {len(triples)} triples...")
        for epoch in range(train_epochs):
            total_loss = 0.0
            for batch in self._create_batches(train_data, batch_size):
                optimizer.zero_grad()
                
                # Forward pass - adjust based on actual MAYPL model interface
                # MAYPL model interface may vary - this is a generic implementation
                try:
                    # Try standard forward pass
                    loss = self.model(batch['head'], batch['relation'], batch['tail'])
                except TypeError:
                    # Try with dictionary input
                    try:
                        loss = self.model(batch)
                    except Exception:
                        # Try with separate arguments
                        loss = self.model(
                            batch['head'], 
                            batch['relation'], 
                            batch['tail'],
                            qualifiers=batch.get('qualifiers')
                        )
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(train_data) * batch_size
                print(f"Epoch {epoch + 1}/{train_epochs}, Average Loss: {avg_loss:.4f}")
        
        self.is_trained = True
        print("MAYPL model training completed!")
    
    def save(self, path: str) -> None:
        """
        Save the MAYPL model to disk.
        
        Args:
            path: Path to save the model.
        """
        if self.model is None:
            raise RuntimeError("No model to save. Train the model first.")
        
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'embedding_dim': self.embedding_dim,
            'entity_id_to_idx': self.entity_id_to_idx,
            'relation_id_to_idx': self.relation_id_to_idx,
            'idx_to_entity_id': self.idx_to_entity_id,
            'idx_to_relation_id': self.idx_to_relation_id,
            'entity_embeddings': self.entity_embeddings,
            'relation_embeddings': self.relation_embeddings,
            'is_trained': self.is_trained,
            'kwargs': self.kwargs
        }, str(save_path))
        
        print(f"Model saved to {save_path}")
    
    def load(self, path: str) -> None:
        """
        Load the MAYPL model from disk.
        
        Args:
            path: Path to load the model from.
        """
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        checkpoint = torch.load(str(load_path), map_location=self.device)
        
        # Restore mappings
        self.embedding_dim = checkpoint['embedding_dim']
        self.entity_id_to_idx = checkpoint['entity_id_to_idx']
        self.relation_id_to_idx = checkpoint['relation_id_to_idx']
        self.idx_to_entity_id = checkpoint['idx_to_entity_id']
        self.idx_to_relation_id = checkpoint['idx_to_relation_id']
        self.entity_embeddings = checkpoint.get('entity_embeddings', {})
        self.relation_embeddings = checkpoint.get('relation_embeddings', {})
        self.is_trained = checkpoint.get('is_trained', False)
        self.kwargs = checkpoint.get('kwargs', {})
        
        # Reinitialize and load model
        num_entities = len(self.entity_id_to_idx)
        num_relations = len(self.relation_id_to_idx)
        
        try:
            self.model = MAYPL(
                num_entities=num_entities,
                num_relations=num_relations,
                embedding_dim=self.embedding_dim,
                **self.kwargs
            ).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            print(f"Model loaded from {load_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load MAYPL model: {e}")
    
    def _convert_to_maypl_format(self, graph_data: Dict[str, Any]) -> Tuple[List[Tuple], List]:
        """
        Convert our graph format to MAYPL's expected format.
        
        MAYPL expects triples in format: (head_idx, relation_idx, tail_idx)
        and optional qualifiers for hyper-relations.
        
        Args:
            graph_data: Dictionary with 'entities' and 'relations'.
            
        Returns:
            Tuple of (triples, qualifiers) in MAYPL format.
        """
        entities = graph_data.get('entities', [])
        relations = graph_data.get('relations', [])
        
        # Build mappings first
        self._build_mappings(graph_data)
        
        # Convert relations to triples
        triples = []
        qualifiers = []  # MAYPL supports hyper-relations with qualifiers
        
        for relation in relations:
            # Handle both dict and Relation object
            if hasattr(relation, 'source_entity_id'):
                head_id = relation.source_entity_id
                tail_id = relation.target_entity_id
                relation_id = relation.id
            else:
                head_id = relation.get('source_entity_id', relation.get('head'))
                tail_id = relation.get('target_entity_id', relation.get('tail'))
                relation_id = relation.get('id', relation.get('relation_id'))
            
            # Get indices
            head_idx = self.entity_id_to_idx.get(head_id)
            tail_idx = self.entity_id_to_idx.get(tail_id)
            rel_idx = self.relation_id_to_idx.get(relation_id)
            
            if head_idx is not None and tail_idx is not None and rel_idx is not None:
                triple = (head_idx, rel_idx, tail_idx)
                triples.append(triple)
                
                # Extract qualifiers if present (for hyper-relations)
                if hasattr(relation, 'properties') and relation.properties:
                    qualifier = relation.properties
                    qualifiers.append(qualifier)
                elif isinstance(relation, dict) and relation.get('properties'):
                    qualifiers.append(relation['properties'])
                else:
                    qualifiers.append({})
        
        return triples, qualifiers
    
    def _build_mappings(self, graph_data: Dict[str, Any]) -> None:
        """
        Build mappings between entity/relation IDs and indices.
        
        Args:
            graph_data: Dictionary with 'entities' and 'relations'.
        """
        entities = graph_data.get('entities', [])
        relations = graph_data.get('relations', [])
        
        # Build entity mappings
        self.entity_id_to_idx = {}
        self.idx_to_entity_id = {}
        for idx, entity in enumerate(entities):
            if hasattr(entity, 'id'):
                entity_id = entity.id
            else:
                entity_id = entity.get('id', entity.get('entity_id'))
            
            if entity_id:
                self.entity_id_to_idx[entity_id] = idx
                self.idx_to_entity_id[idx] = entity_id
        
        # Build relation mappings
        self.relation_id_to_idx = {}
        self.idx_to_relation_id = {}
        for idx, relation in enumerate(relations):
            if hasattr(relation, 'id'):
                relation_id = relation.id
            else:
                relation_id = relation.get('id', relation.get('relation_id'))
            
            if relation_id:
                self.relation_id_to_idx[relation_id] = idx
                self.idx_to_relation_id[idx] = relation_id
    
    def _prepare_training_data(self, triples: List[Tuple], qualifiers: List) -> List:
        """
        Prepare training data in format expected by MAYPL.
        
        Args:
            triples: List of (head_idx, relation_idx, tail_idx) tuples.
            qualifiers: List of qualifier dictionaries.
            
        Returns:
            Prepared training data.
        """
        # Convert to tensors
        train_data = []
        for i, triple in enumerate(triples):
            head_idx, rel_idx, tail_idx = triple
            data_point = {
                'head': torch.tensor([head_idx], dtype=torch.long).to(self.device),
                'relation': torch.tensor([rel_idx], dtype=torch.long).to(self.device),
                'tail': torch.tensor([tail_idx], dtype=torch.long).to(self.device),
            }
            
            # Add qualifiers if present
            if i < len(qualifiers) and qualifiers[i]:
                data_point['qualifiers'] = qualifiers[i]
            
            train_data.append(data_point)
        
        return train_data
    
    def _create_batches(self, data: List, batch_size: int):
        """
        Create batches from training data.
        
        Args:
            data: List of training data points.
            batch_size: Size of each batch.
            
        Yields:
            Batches of data.
        """
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            
            # Stack tensors
            heads = torch.cat([item['head'] for item in batch])
            relations = torch.cat([item['relation'] for item in batch])
            tails = torch.cat([item['tail'] for item in batch])
            
            batch_dict = {
                'head': heads,
                'relation': relations,
                'tail': tails
            }
            
            # Add qualifiers if present
            if 'qualifiers' in batch[0]:
                batch_dict['qualifiers'] = [item.get('qualifiers', {}) for item in batch]
            
            yield batch_dict

