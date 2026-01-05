"""
Unit tests for GraphVisualizer module.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from kg_construction.visualization import GraphVisualizer
from kg_construction.models import KnowledgeGraph, Entity, Relation


class TestGraphVisualizer:
    """Tests for GraphVisualizer class."""
    
    def test_initialization(self):
        """Test GraphVisualizer initialization."""
        visualizer = GraphVisualizer()
        assert visualizer is not None
        assert visualizer.figsize == (12, 8)
        assert visualizer.dpi == 100
    
    def test_initialization_custom_params(self):
        """Test GraphVisualizer initialization with custom parameters."""
        visualizer = GraphVisualizer(figsize=(10, 6), dpi=150)
        assert visualizer.figsize == (10, 6)
        assert visualizer.dpi == 150
    
    def test_visualize_empty_graph(self, empty_knowledge_graph, capsys):
        """Test visualization of empty graph."""
        visualizer = GraphVisualizer()
        visualizer.visualize(empty_knowledge_graph)
        
        captured = capsys.readouterr()
        assert "Warning" in captured.out or "empty" in captured.out.lower()
    
    def test_visualize_basic(self, sample_knowledge_graph, temp_dir):
        """Test basic graph visualization."""
        visualizer = GraphVisualizer()
        output_path = temp_dir / "test_viz.png"
        
        visualizer.visualize(
            sample_knowledge_graph,
            output_path=str(output_path),
            show_labels=True
        )
        
        assert output_path.exists()
    
    def test_visualize_with_filtering(self, sample_knowledge_graph, temp_dir):
        """Test visualization with entity type filtering."""
        visualizer = GraphVisualizer()
        output_path = temp_dir / "test_viz_filtered.png"
        
        visualizer.visualize(
            sample_knowledge_graph,
            output_path=str(output_path),
            filter_entity_types=["LAW", "ORGANIZATION"]
        )
        
        assert output_path.exists()
    
    def test_visualize_with_relation_filtering(self, sample_knowledge_graph, temp_dir):
        """Test visualization with relation type filtering."""
        visualizer = GraphVisualizer()
        output_path = temp_dir / "test_viz_rel_filtered.png"
        
        visualizer.visualize(
            sample_knowledge_graph,
            output_path=str(output_path),
            filter_relation_types=["ENACTED"]
        )
        
        assert output_path.exists()
    
    def test_visualize_max_nodes(self, large_knowledge_graph, temp_dir):
        """Test visualization with max_nodes limit."""
        visualizer = GraphVisualizer()
        output_path = temp_dir / "test_viz_max_nodes.png"
        
        visualizer.visualize(
            large_knowledge_graph,
            output_path=str(output_path),
            max_nodes=20
        )
        
        assert output_path.exists()
    
    def test_visualize_different_layouts(self, sample_knowledge_graph, temp_dir):
        """Test visualization with different layout algorithms."""
        visualizer = GraphVisualizer()
        layouts = ['spring', 'circular']
        
        for layout in layouts:
            output_path = temp_dir / f"test_viz_{layout}.png"
            visualizer.visualize(
                sample_knowledge_graph,
                output_path=str(output_path),
                layout=layout
            )
            assert output_path.exists()
    
    def test_build_networkx_graph(self, sample_knowledge_graph):
        """Test networkx graph conversion."""
        visualizer = GraphVisualizer()
        nx_graph = visualizer._build_networkx_graph(sample_knowledge_graph)
        
        assert nx_graph is not None
        assert len(nx_graph.nodes()) > 0
        assert len(nx_graph.edges()) > 0
    
    def test_build_networkx_graph_empty(self, empty_knowledge_graph):
        """Test networkx graph conversion with empty graph."""
        visualizer = GraphVisualizer()
        nx_graph = visualizer._build_networkx_graph(empty_knowledge_graph)
        
        assert nx_graph is not None
        assert len(nx_graph.nodes()) == 0
        assert len(nx_graph.edges()) == 0
    
    def test_build_networkx_graph_with_filtering(self, sample_knowledge_graph):
        """Test networkx graph conversion with entity type filtering."""
        visualizer = GraphVisualizer()
        nx_graph = visualizer._build_networkx_graph(
            sample_knowledge_graph,
            filter_entity_types=["LAW"]
        )
        
        # Should only contain LAW entities
        assert len(nx_graph.nodes()) <= len(sample_knowledge_graph.entities)
    
    def test_apply_layout_spring(self, sample_knowledge_graph):
        """Test spring layout application."""
        visualizer = GraphVisualizer()
        nx_graph = visualizer._build_networkx_graph(sample_knowledge_graph)
        pos = visualizer._apply_layout(nx_graph, 'spring')
        
        assert isinstance(pos, dict)
        assert len(pos) == len(nx_graph.nodes())
        assert all(isinstance(v, tuple) and len(v) == 2 for v in pos.values())
    
    def test_apply_layout_circular(self, sample_knowledge_graph):
        """Test circular layout application."""
        visualizer = GraphVisualizer()
        nx_graph = visualizer._build_networkx_graph(sample_knowledge_graph)
        pos = visualizer._apply_layout(nx_graph, 'circular')
        
        assert isinstance(pos, dict)
        assert len(pos) == len(nx_graph.nodes())
    
    def test_get_node_colors(self, sample_knowledge_graph):
        """Test node color assignment."""
        visualizer = GraphVisualizer()
        nx_graph = visualizer._build_networkx_graph(sample_knowledge_graph)
        colors = visualizer._get_node_colors(nx_graph, sample_knowledge_graph)
        
        assert isinstance(colors, list)
        assert len(colors) == len(nx_graph.nodes())
        assert all(isinstance(c, str) for c in colors)
    
    def test_get_node_labels(self, sample_knowledge_graph):
        """Test node label extraction."""
        visualizer = GraphVisualizer()
        nx_graph = visualizer._build_networkx_graph(sample_knowledge_graph)
        labels = visualizer._get_node_labels(nx_graph, sample_knowledge_graph)
        
        assert isinstance(labels, dict)
        assert len(labels) <= len(nx_graph.nodes())
    
    def test_get_edge_labels(self, sample_knowledge_graph):
        """Test edge label extraction."""
        visualizer = GraphVisualizer()
        nx_graph = visualizer._build_networkx_graph(sample_knowledge_graph)
        labels = visualizer._get_edge_labels(nx_graph, sample_knowledge_graph)
        
        assert isinstance(labels, dict)
        assert len(labels) <= len(nx_graph.edges())
    
    def test_create_legend(self, sample_knowledge_graph):
        """Test legend creation."""
        visualizer = GraphVisualizer()
        legend = visualizer._create_legend(sample_knowledge_graph)
        
        assert isinstance(legend, list)
        assert len(legend) > 0
    
    def test_create_legend_with_filter(self, sample_knowledge_graph):
        """Test legend creation with entity type filter."""
        visualizer = GraphVisualizer()
        legend = visualizer._create_legend(
            sample_knowledge_graph,
            filter_entity_types=["LAW", "ORGANIZATION"]
        )
        
        assert isinstance(legend, list)
        assert len(legend) <= 2
    
    def test_visualize_no_output_path(self, sample_knowledge_graph):
        """Test visualization without output path (should show interactively)."""
        visualizer = GraphVisualizer()
        # This should not raise an error, but we can't easily test interactive display
        # So we'll just verify it doesn't crash
        try:
            with patch('matplotlib.pyplot.show'):
                visualizer.visualize(sample_knowledge_graph, output_path=None)
        except Exception as e:
            # If it fails, it should be a display-related error, not a code error
            assert "display" in str(e).lower() or "gui" in str(e).lower()

