"""
Data models for Knowledge Graph construction.

Defines Entity, Relation, Statement/Triple, and KnowledgeGraph classes.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import uuid


@dataclass
class Entity:
    """
    Represents an entity in the knowledge graph.
    
    Entities can be people, organizations, concepts, documents, etc.
    """
    id: str
    name: str
    entity_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize created_at if not provided."""
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class Relation:
    """
    Represents a relation between entities.
    
    Relations connect two entities with a specific relationship type.
    """
    id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: Optional[float] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize created_at if not provided."""
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class Statement:
    """
    Represents a statement (triple) in the knowledge graph.
    
    A statement is a subject-predicate-object triple.
    """
    subject: str
    predicate: str
    object: str
    confidence: Optional[float] = None
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_relation(self, entity_map: Dict[str, str]) -> Relation:
        """
        Convert statement to a Relation object.
        
        Args:
            entity_map: Mapping from entity names to entity IDs.
            
        Returns:
            Relation object.
            
        Raises:
            ValueError: If subject or object entity IDs are not found in entity_map.
        """
        source_id = entity_map.get(self.subject)
        target_id = entity_map.get(self.object)
        
        if not source_id:
            raise ValueError(f"Subject entity '{self.subject}' not found in entity_map")
        if not target_id:
            raise ValueError(f"Object entity '{self.object}' not found in entity_map")
        
        return Relation(
            id=str(uuid.uuid4()),
            source_entity_id=source_id,
            target_entity_id=target_id,
            relation_type=self.predicate,
            confidence=self.confidence,
            metadata={
                'source': self.source,
                'original_statement': {
                    'subject': self.subject,
                    'predicate': self.predicate,
                    'object': self.object
                }
            }
        )


class KnowledgeGraph:
    """
    Represents a knowledge graph containing entities and relations.
    """
    
    def __init__(self):
        """Initialize an empty knowledge graph."""
        self.entities: Dict[str, Entity] = {}
        self.relations: Dict[str, Relation] = {}
        self.metadata: Dict[str, Any] = {}
    
    def add_entity(self, entity: Entity) -> None:
        """
        Add an entity to the graph.
        
        Args:
            entity: Entity object to add.
        """
        self.entities[entity.id] = entity
    
    def add_relation(self, relation: Relation) -> None:
        """
        Add a relation to the graph.
        
        Args:
            relation: Relation object to add.
        """
        self.relations[relation.id] = relation
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """
        Get an entity by ID.
        
        Args:
            entity_id: Entity ID.
            
        Returns:
            Entity object or None if not found.
        """
        return self.entities.get(entity_id)
    
    def get_relation(self, relation_id: str) -> Optional[Relation]:
        """
        Get a relation by ID.
        
        Args:
            relation_id: Relation ID.
            
        Returns:
            Relation object or None if not found.
        """
        return self.relations.get(relation_id)
    
    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """
        Get all entities of a specific type.
        
        Args:
            entity_type: Type of entity to filter by.
            
        Returns:
            List of Entity objects.
        """
        return [e for e in self.entities.values() if e.entity_type == entity_type]
    
    def get_relations_by_type(self, relation_type: str) -> List[Relation]:
        """
        Get all relations of a specific type.
        
        Args:
            relation_type: Type of relation to filter by.
            
        Returns:
            List of Relation objects.
        """
        return [r for r in self.relations.values() if r.relation_type == relation_type]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph.
        
        Returns:
            Dictionary with statistics (entity count, relation count, etc.).
        """
        return {
            'entity_count': len(self.entities),
            'relation_count': len(self.relations),
            'entity_types': len(set(e.entity_type for e in self.entities.values())),
            'relation_types': len(set(r.relation_type for r in self.relations.values())),
        }

