"""
Interface to build or update Knowledge Graph (entities + relations).
"""

from typing import List, Optional
from .models import Entity, Relation, Statement, KnowledgeGraph
from .storage import KGStorage


class GraphBuilder:
    """
    Interface for building and updating knowledge graphs.
    
    Handles the construction of knowledge graphs from entities and relations,
    and provides methods to update existing graphs.
    """
    
    def __init__(self, storage: Optional[KGStorage] = None):
        """
        Initialize the graph builder.
        
        Args:
            storage: Optional KG storage backend instance.
        """
        self.storage = storage
        self.graph = KnowledgeGraph()
    
    def add_entity(self, entity: Entity) -> None:
        """
        Add an entity to the knowledge graph.
        
        Args:
            entity: Entity object to add.
        """
        self.graph.add_entity(entity)
        
        # Optionally save to storage
        if self.storage:
            try:
                self.storage.save_entity(entity)
            except Exception as e:
                # Log error but continue
                print(f"Warning: Failed to save entity to storage: {e}")
    
    def add_relation(self, relation: Relation) -> None:
        """
        Add a relation to the knowledge graph.
        
        Args:
            relation: Relation object to add.
        """
        # Verify that both entities exist
        if relation.source_entity_id not in self.graph.entities:
            raise ValueError(f"Source entity {relation.source_entity_id} not found in graph")
        if relation.target_entity_id not in self.graph.entities:
            raise ValueError(f"Target entity {relation.target_entity_id} not found in graph")
        
        self.graph.add_relation(relation)
        
        # Optionally save to storage
        if self.storage:
            try:
                self.storage.save_relation(relation)
            except Exception as e:
                # Log error but continue
                print(f"Warning: Failed to save relation to storage: {e}")
    
    def add_statement(self, statement: Statement) -> None:
        """
        Add a statement (triple) to the knowledge graph.
        
        Args:
            statement: Statement object to add.
        """
        # TODO: Implement statement addition
        raise NotImplementedError("Statement addition not yet implemented")
    
    def build_from_entities_relations(
        self, 
        entities: List[Entity], 
        relations: List[Relation]
    ) -> KnowledgeGraph:
        """
        Build a knowledge graph from lists of entities and relations.
        
        Args:
            entities: List of Entity objects.
            relations: List of Relation objects.
            
        Returns:
            Constructed KnowledgeGraph object.
        """
        # Create a new graph
        self.graph = KnowledgeGraph()
        
        # Add all entities first
        for entity in entities:
            self.add_entity(entity)
        
        # Then add all relations
        for relation in relations:
            try:
                self.add_relation(relation)
            except ValueError as e:
                # Skip relations with missing entities
                print(f"Warning: Skipping relation {relation.id}: {e}")
                continue
        
        return self.graph
    
    def update_graph(self, entities: List[Entity], relations: List[Relation]) -> None:
        """
        Update existing knowledge graph with new entities and relations.
        
        Args:
            entities: List of new Entity objects.
            relations: List of new Relation objects.
        """
        # Add new entities (will update if ID already exists)
        for entity in entities:
            if entity.id in self.graph.entities:
                # Update existing entity
                existing = self.graph.entities[entity.id]
                existing.properties.update(entity.properties)
                existing.metadata.update(entity.metadata)
            else:
                self.add_entity(entity)
        
        # Add new relations
        for relation in relations:
            try:
                self.add_relation(relation)
            except ValueError as e:
                print(f"Warning: Skipping relation {relation.id}: {e}")
                continue
    
    def merge_graphs(self, other_graph: KnowledgeGraph) -> None:
        """
        Merge another knowledge graph into this one.
        
        Args:
            other_graph: KnowledgeGraph object to merge.
        """
        # TODO: Implement graph merging
        raise NotImplementedError("Graph merging not yet implemented")
    
    def get_graph(self) -> KnowledgeGraph:
        """
        Get the current knowledge graph.
        
        Returns:
            Current KnowledgeGraph object.
        """
        return self.graph

