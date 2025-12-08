"""
Unit tests for knowledge graph construction module.
"""

import unittest
from kg_construction.distiller import DocumentDistiller
from kg_construction.extractor import EntityRelationExtractor
from kg_construction.graph_builder import GraphBuilder
from kg_construction.models import Entity, Relation, KnowledgeGraph


class TestDocumentDistiller(unittest.TestCase):
    """Tests for DocumentDistiller."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.distiller = DocumentDistiller()
    
    def test_distiller_initialization(self):
        """Test distiller initialization."""
        self.assertIsNotNone(self.distiller)
    
    # TODO: Add more tests for document distillation


class TestEntityRelationExtractor(unittest.TestCase):
    """Tests for EntityRelationExtractor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = EntityRelationExtractor()
    
    def test_extractor_initialization(self):
        """Test extractor initialization."""
        self.assertIsNotNone(self.extractor)
    
    # TODO: Add more tests for entity/relation extraction


class TestGraphBuilder(unittest.TestCase):
    """Tests for GraphBuilder."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.builder = GraphBuilder()
    
    def test_builder_initialization(self):
        """Test builder initialization."""
        self.assertIsNotNone(self.builder)
        self.assertIsNotNone(self.builder.graph)
    
    # TODO: Add more tests for graph building


class TestKnowledgeGraphModels(unittest.TestCase):
    """Tests for KnowledgeGraph data models."""
    
    def test_entity_creation(self):
        """Test Entity creation."""
        entity = Entity(
            id="e1",
            name="Test Entity",
            entity_type="PERSON"
        )
        self.assertEqual(entity.id, "e1")
        self.assertEqual(entity.name, "Test Entity")
        self.assertEqual(entity.entity_type, "PERSON")
    
    def test_relation_creation(self):
        """Test Relation creation."""
        relation = Relation(
            id="r1",
            source_entity_id="e1",
            target_entity_id="e2",
            relation_type="KNOWS"
        )
        self.assertEqual(relation.id, "r1")
        self.assertEqual(relation.source_entity_id, "e1")
        self.assertEqual(relation.target_entity_id, "e2")
        self.assertEqual(relation.relation_type, "KNOWS")
    
    def test_knowledge_graph_operations(self):
        """Test KnowledgeGraph operations."""
        graph = KnowledgeGraph()
        entity = Entity(id="e1", name="Entity 1", entity_type="PERSON")
        relation = Relation(
            id="r1",
            source_entity_id="e1",
            target_entity_id="e2",
            relation_type="KNOWS"
        )
        
        graph.add_entity(entity)
        graph.add_relation(relation)
        
        self.assertEqual(len(graph.entities), 1)
        self.assertEqual(len(graph.relations), 1)
        self.assertEqual(graph.get_entity("e1"), entity)
        self.assertEqual(graph.get_relation("r1"), relation)


if __name__ == '__main__':
    unittest.main()

