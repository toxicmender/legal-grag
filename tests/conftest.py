"""
Shared pytest fixtures and test utilities.
"""

import pytest
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from kg_construction.models import Entity, Relation, KnowledgeGraph


@pytest.fixture
def sample_text():
    """Sample text for lexical analysis tests."""
    return """
    The Bharatiya Nyaya Sanhita Act, 2023 is an important piece of legislation.
    It was enacted by the Parliament of India. The Act contains various sections
    that define criminal offenses and their punishments. Section 1 deals with
    the title and extent of the Act. The Act came into force on January 1, 2024.
    """


@pytest.fixture
def sample_entities():
    """Sample entity data for testing."""
    return [
        Entity(
            id="e1",
            name="Bharatiya Nyaya Sanhita Act",
            entity_type="LAW",
            properties={"year": 2023},
            metadata={"source": "test"}
        ),
        Entity(
            id="e2",
            name="Parliament of India",
            entity_type="ORGANIZATION",
            metadata={"source": "test"}
        ),
        Entity(
            id="e3",
            name="Section 1",
            entity_type="SECTION",
            metadata={"source": "test"}
        ),
        Entity(
            id="e4",
            name="India",
            entity_type="LOCATION",
            metadata={"source": "test"}
        ),
    ]


@pytest.fixture
def sample_relations(sample_entities):
    """Sample relation data for testing."""
    return [
        Relation(
            id="r1",
            source_entity_id="e2",
            target_entity_id="e1",
            relation_type="ENACTED",
            confidence=0.95,
            metadata={"source": "test"}
        ),
        Relation(
            id="r2",
            source_entity_id="e1",
            target_entity_id="e3",
            relation_type="CONTAINS",
            confidence=0.90,
            metadata={"source": "test"}
        ),
        Relation(
            id="r3",
            source_entity_id="e1",
            target_entity_id="e4",
            relation_type="APPLIES_TO",
            confidence=0.85,
            metadata={"source": "test"}
        ),
    ]


@pytest.fixture
def sample_knowledge_graph(sample_entities, sample_relations):
    """Complete knowledge graph for testing."""
    graph = KnowledgeGraph()
    
    for entity in sample_entities:
        graph.add_entity(entity)
    
    for relation in sample_relations:
        graph.add_relation(relation)
    
    return graph


@pytest.fixture
def temp_dir():
    """Temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_pdf_path(temp_dir):
    """
    Create a sample PDF file for testing.
    
    Note: This creates a minimal PDF file. For real PDF tests,
    use actual PDF files from the dataset.
    """
    # Create a simple text file that can be used as a mock PDF path
    # In real tests, you would use actual PDF files
    pdf_path = temp_dir / "sample.pdf"
    pdf_path.touch()  # Create empty file as placeholder
    return pdf_path


@pytest.fixture
def mock_openai_api(mocker):
    """Mock OpenAI API calls for itext2kg."""
    # Mock the itext2kg build_graph method
    mock_itext2kg = mocker.MagicMock()
    mock_entities = [
        {"name": "Test Entity 1", "type": "PERSON"},
        {"name": "Test Entity 2", "type": "ORGANIZATION"},
    ]
    mock_relations = [
        {"head": "Test Entity 1", "relation": "WORKS_FOR", "tail": "Test Entity 2"},
    ]
    mock_itext2kg.build_graph.return_value = (mock_entities, mock_relations)
    
    return mock_itext2kg


@pytest.fixture
def empty_knowledge_graph():
    """Empty knowledge graph for testing edge cases."""
    return KnowledgeGraph()


@pytest.fixture
def large_knowledge_graph():
    """Large knowledge graph for testing performance."""
    graph = KnowledgeGraph()
    
    # Create 100 entities
    for i in range(100):
        entity = Entity(
            id=f"e{i}",
            name=f"Entity {i}",
            entity_type="CONCEPT" if i % 2 == 0 else "PERSON",
            metadata={"index": i}
        )
        graph.add_entity(entity)
    
    # Create 150 relations
    for i in range(150):
        source_id = f"e{i % 100}"
        target_id = f"e{(i + 1) % 100}"
        relation = Relation(
            id=f"r{i}",
            source_entity_id=source_id,
            target_entity_id=target_id,
            relation_type="RELATED_TO",
            metadata={"index": i}
        )
        graph.add_relation(relation)
    
    return graph


@pytest.fixture(autouse=True)
def setup_nltk_data():
    """Ensure NLTK data is downloaded before tests."""
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('stopwords', quiet=True)
    except ImportError:
        pass

