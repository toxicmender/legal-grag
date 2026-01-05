"""
Integration tests for the complete PDF to Knowledge Graph pipeline.
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from ingestion.pdf_reader import PDFReader
from ingestion.lexical_analyzer import LexicalAnalyzer
from kg_construction.extractor import EntityRelationExtractor
from kg_construction.graph_builder import GraphBuilder
from kg_construction.visualization import GraphVisualizer
from kg_construction.models import Entity, Relation


@pytest.mark.integration
class TestPDFKGPipelineIntegration:
    """Integration tests for the complete pipeline."""
    
    @pytest.fixture
    def sample_pdf_text(self):
        """Sample PDF text content."""
        return """
        The Information Technology Act, 2000 is an Act of the Indian Parliament.
        It was enacted on 9 June 2000. The Act provides legal recognition for
        electronic commerce and digital signatures. Section 43 deals with
        penalties for damage to computer systems. The Act applies to the whole
        of India.
        """
    
    @pytest.fixture
    def mock_pdf_reader(self, sample_pdf_text, tmp_path):
        """Mock PDF reader that returns sample text."""
        reader = PDFReader()
        reader.base_dir = tmp_path
        
        # Create a mock PDF file
        pdf_file = tmp_path / "test_act.pdf"
        pdf_file.touch()
        
        # Mock the parser
        reader.parser.parse_pdf = Mock(return_value={
            'text': sample_pdf_text,
            'metadata': {'title': 'Test Act'},
            'pages': [{'page_number': 1, 'text': sample_pdf_text}],
            'page_count': 1
        })
        
        return reader
    
    @pytest.fixture
    def mock_itext2kg_extractor(self):
        """Mock itext2kg extractor."""
        extractor = Mock(spec=EntityRelationExtractor)
        
        # Create mock entities and relations
        mock_entities = [
            Entity(id="e1", name="Information Technology Act", entity_type="LAW"),
            Entity(id="e2", name="Indian Parliament", entity_type="ORGANIZATION"),
            Entity(id="e3", name="Section 43", entity_type="SECTION"),
            Entity(id="e4", name="India", entity_type="LOCATION"),
        ]
        
        mock_relations = [
            Relation(
                id="r1",
                source_entity_id="e2",
                target_entity_id="e1",
                relation_type="ENACTED"
            ),
            Relation(
                id="r2",
                source_entity_id="e1",
                target_entity_id="e3",
                relation_type="CONTAINS"
            ),
        ]
        
        extractor.extract_from_text.return_value = {
            'entities': mock_entities,
            'relations': mock_relations
        }
        
        return extractor
    
    def test_complete_pipeline(
        self,
        mock_pdf_reader,
        mock_itext2kg_extractor,
        temp_dir
    ):
        """Test the complete pipeline from PDF to visualization."""
        # Step 1: Read PDF
        pdf_data = mock_pdf_reader.read_pdf_with_metadata("test_act.pdf")
        text = pdf_data['text']
        assert len(text) > 0
        
        # Step 2: Lexical Analysis
        analyzer = LexicalAnalyzer()
        analysis = analyzer.analyze(text)
        assert 'tokens' in analysis
        assert 'statistics' in analysis
        assert analysis['statistics']['word_count'] > 0
        
        # Step 3: Knowledge Graph Creation
        result = mock_itext2kg_extractor.extract_from_text(analysis['processed_text'])
        entities = result['entities']
        relations = result['relations']
        assert len(entities) > 0
        
        # Step 4: Build Graph
        builder = GraphBuilder()
        graph = builder.build_from_entities_relations(entities, relations)
        assert len(graph.entities) > 0
        
        # Step 5: Visualization
        visualizer = GraphVisualizer()
        output_path = temp_dir / "pipeline_test.png"
        visualizer.visualize(graph, output_path=str(output_path))
        assert output_path.exists()
    
    def test_pipeline_error_handling_missing_pdf(self, tmp_path):
        """Test pipeline handles missing PDF gracefully."""
        reader = PDFReader()
        reader.base_dir = tmp_path
        
        with pytest.raises((FileNotFoundError, RuntimeError)):
            reader.read_pdf("nonexistent.pdf")
    
    def test_pipeline_error_handling_empty_text(self, mock_pdf_reader):
        """Test pipeline handles empty PDF text."""
        # Mock empty text
        mock_pdf_reader.parser.parse_pdf.return_value = {
            'text': '',
            'metadata': {},
            'pages': [],
            'page_count': 0
        }
        
        pdf_data = mock_pdf_reader.read_pdf_with_metadata("test_act.pdf")
        text = pdf_data['text']
        
        analyzer = LexicalAnalyzer()
        analysis = analyzer.analyze(text)
        
        # Should return empty analysis without crashing
        assert analysis['tokens'] == []
        assert analysis['statistics']['word_count'] == 0
    
    def test_pipeline_data_flow(
        self,
        mock_pdf_reader,
        mock_itext2kg_extractor,
        temp_dir
    ):
        """Test data flows correctly through pipeline stages."""
        # Read PDF
        pdf_data = mock_pdf_reader.read_pdf_with_metadata("test_act.pdf")
        original_text = pdf_data['text']
        
        # Lexical analysis
        analyzer = LexicalAnalyzer()
        analysis = analyzer.analyze(original_text)
        processed_text = analysis['processed_text']
        
        # Verify processed text is derived from original
        assert len(processed_text) <= len(original_text)
        assert len(processed_text) > 0
        
        # Extract entities/relations
        result = mock_itext2kg_extractor.extract_from_text(processed_text)
        
        # Verify extractor was called with processed text
        mock_itext2kg_extractor.extract_from_text.assert_called_once()
        call_args = mock_itext2kg_extractor.extract_from_text.call_args[0][0]
        assert call_args == processed_text
        
        # Build graph
        builder = GraphBuilder()
        graph = builder.build_from_entities_relations(
            result['entities'],
            result['relations']
        )
        
        # Verify graph contains extracted entities
        assert len(graph.entities) == len(result['entities'])
        assert len(graph.relations) == len(result['relations'])
    
    def test_pipeline_with_multiple_pdfs(
        self,
        mock_pdf_reader,
        mock_itext2kg_extractor,
        temp_dir
    ):
        """Test pipeline processing multiple PDFs."""
        # Create multiple mock PDFs
        pdf_files = []
        for i in range(3):
            pdf_file = mock_pdf_reader.base_dir / f"act_{i}.pdf"
            pdf_file.touch()
            pdf_files.append(pdf_file)
        
        # Process each PDF
        analyzer = LexicalAnalyzer()
        visualizer = GraphVisualizer()
        builder = GraphBuilder()
        
        results = []
        for pdf_file in pdf_files:
            pdf_data = mock_pdf_reader.read_pdf_with_metadata(pdf_file.name)
            analysis = analyzer.analyze(pdf_data['text'])
            result = mock_itext2kg_extractor.extract_from_text(analysis['processed_text'])
            graph = builder.build_from_entities_relations(
                result['entities'],
                result['relations']
            )
            
            output_path = temp_dir / f"{pdf_file.stem}_viz.png"
            visualizer.visualize(graph, output_path=str(output_path))
            
            results.append({
                'pdf': pdf_file.name,
                'graph': graph,
                'viz_path': output_path
            })
        
        assert len(results) == 3
        assert all(r['viz_path'].exists() for r in results)
    
    @patch('kg_construction.extractor.iText2KG')
    def test_pipeline_with_mocked_api(self, mock_itext2kg_class, mock_pdf_reader, temp_dir):
        """Test pipeline with mocked itext2kg API to avoid API costs."""
        # Mock itext2kg
        mock_itext2kg_instance = Mock()
        mock_entities = [
            {"name": "Test Entity", "type": "PERSON"}
        ]
        mock_relations = [
            {"head": "Test Entity", "relation": "TEST_REL", "tail": "Other Entity"}
        ]
        mock_itext2kg_instance.build_graph.return_value = (mock_entities, mock_relations)
        mock_itext2kg_class.return_value = mock_itext2kg_instance
        
        # Create extractor (will use mocked itext2kg)
        extractor = EntityRelationExtractor(openai_api_key="test_key")
        
        # Run pipeline
        pdf_data = mock_pdf_reader.read_pdf_with_metadata("test_act.pdf")
        analyzer = LexicalAnalyzer()
        analysis = analyzer.analyze(pdf_data['text'])
        
        result = extractor.extract_from_text(analysis['processed_text'])
        
        # Verify API was called
        assert mock_itext2kg_instance.build_graph.called
        
        # Verify results
        assert len(result['entities']) > 0
    
    def test_pipeline_visualization_output(self, sample_knowledge_graph, temp_dir):
        """Test that visualization generates correct output files."""
        visualizer = GraphVisualizer()
        
        # Test different output formats
        output_paths = [
            temp_dir / "test.png",
            temp_dir / "test.jpg",
        ]
        
        for output_path in output_paths:
            visualizer.visualize(
                sample_knowledge_graph,
                output_path=str(output_path)
            )
            assert output_path.exists()
            assert output_path.stat().st_size > 0  # File is not empty

