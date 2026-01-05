"""
Knowledge Graph Visualization module using matplotlib and networkx.

Provides functionality to visualize knowledge graphs with customizable layouts,
colors, and filtering options.
"""

from typing import Dict, Any, Optional, List, Set
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from .models import KnowledgeGraph, Entity, Relation


class GraphVisualizer:
    """
    Visualizes knowledge graphs using matplotlib and networkx.
    
    Supports multiple layout algorithms, node/edge coloring, and filtering.
    """
    
    def __init__(self, figsize: tuple = (12, 8), dpi: int = 100):
        """
        Initialize the graph visualizer.
        
        Args:
            figsize: Figure size (width, height) in inches.
            dpi: Resolution in dots per inch.
        """
        self.figsize = figsize
        self.dpi = dpi
        self.default_node_color = '#3498db'
        self.default_edge_color = '#95a5a6'
        
        # Color palette for entity types
        self.entity_type_colors = {
            'PERSON': '#e74c3c',
            'ORGANIZATION': '#9b59b6',
            'LOCATION': '#1abc9c',
            'CONCEPT': '#f39c12',
            'EVENT': '#e67e22',
            'DATE': '#34495e',
            'LAW': '#2ecc71',
            'SECTION': '#16a085',
        }
    
    def visualize(
        self,
        graph: KnowledgeGraph,
        output_path: Optional[str] = None,
        layout: str = 'spring',
        show_labels: bool = True,
        node_size: int = 1000,
        font_size: int = 10,
        filter_entity_types: Optional[List[str]] = None,
        filter_relation_types: Optional[List[str]] = None,
        max_nodes: Optional[int] = None
    ) -> None:
        """
        Visualize a knowledge graph.
        
        Args:
            graph: KnowledgeGraph object to visualize.
            output_path: Optional path to save the visualization. If None, displays interactively.
            layout: Layout algorithm ('spring', 'circular', 'hierarchical', 'kamada_kawai').
            show_labels: Whether to show node and edge labels.
            node_size: Size of nodes.
            font_size: Font size for labels.
            filter_entity_types: Optional list of entity types to include.
            filter_relation_types: Optional list of relation types to include.
            max_nodes: Optional maximum number of nodes to display (for large graphs).
        """
        if not graph.entities:
            print("Warning: Graph is empty. Nothing to visualize.")
            return
        
        # Build networkx graph
        nx_graph = self._build_networkx_graph(
            graph,
            filter_entity_types=filter_entity_types,
            filter_relation_types=filter_relation_types,
            max_nodes=max_nodes
        )
        
        if len(nx_graph.nodes()) == 0:
            print("Warning: No nodes after filtering. Nothing to visualize.")
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Apply layout
        pos = self._apply_layout(nx_graph, layout)
        
        # Get node colors and labels
        node_colors = self._get_node_colors(nx_graph, graph)
        node_labels = self._get_node_labels(nx_graph, graph) if show_labels else {}
        edge_labels = self._get_edge_labels(nx_graph, graph) if show_labels else {}
        
        # Draw graph
        nx.draw_networkx_nodes(
            nx_graph,
            pos,
            node_color=node_colors,
            node_size=node_size,
            ax=ax,
            alpha=0.9
        )
        
        nx.draw_networkx_edges(
            nx_graph,
            pos,
            edge_color=self.default_edge_color,
            width=1.5,
            alpha=0.6,
            ax=ax,
            arrows=True,
            arrowsize=20,
            arrowstyle='->'
        )
        
        if node_labels:
            nx.draw_networkx_labels(
                nx_graph,
                pos,
                labels=node_labels,
                font_size=font_size,
                ax=ax
            )
        
        if edge_labels:
            nx.draw_networkx_edge_labels(
                nx_graph,
                pos,
                edge_labels=edge_labels,
                font_size=font_size - 2,
                ax=ax
            )
        
        # Add legend for entity types
        legend_elements = self._create_legend(graph, filter_entity_types)
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper left', fontsize=8)
        
        ax.set_title("Knowledge Graph Visualization", fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        # Save or show
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Visualization saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def _build_networkx_graph(
        self,
        graph: KnowledgeGraph,
        filter_entity_types: Optional[List[str]] = None,
        filter_relation_types: Optional[List[str]] = None,
        max_nodes: Optional[int] = None
    ) -> nx.DiGraph:
        """
        Convert KnowledgeGraph to networkx DiGraph.
        
        Args:
            graph: KnowledgeGraph to convert.
            filter_entity_types: Optional list of entity types to include.
            filter_relation_types: Optional list of relation types to include.
            max_nodes: Optional maximum number of nodes.
            
        Returns:
            networkx DiGraph representation.
        """
        nx_graph = nx.DiGraph()
        
        # Filter entities
        entities_to_include = {}
        for entity_id, entity in graph.entities.items():
            if filter_entity_types and entity.entity_type not in filter_entity_types:
                continue
            entities_to_include[entity_id] = entity
        
        # Limit nodes if specified
        if max_nodes and len(entities_to_include) > max_nodes:
            # Select entities with most relations
            entity_relation_counts = {}
            for relation in graph.relations.values():
                entity_relation_counts[relation.source_entity_id] = \
                    entity_relation_counts.get(relation.source_entity_id, 0) + 1
                entity_relation_counts[relation.target_entity_id] = \
                    entity_relation_counts.get(relation.target_entity_id, 0) + 1
            
            sorted_entities = sorted(
                entities_to_include.items(),
                key=lambda x: entity_relation_counts.get(x[0], 0),
                reverse=True
            )
            entities_to_include = dict(sorted_entities[:max_nodes])
        
        # Add nodes
        for entity_id, entity in entities_to_include.items():
            nx_graph.add_node(entity_id, name=entity.name, entity_type=entity.entity_type)
        
        # Add edges
        for relation in graph.relations.values():
            if relation.source_entity_id not in entities_to_include:
                continue
            if relation.target_entity_id not in entities_to_include:
                continue
            if filter_relation_types and relation.relation_type not in filter_relation_types:
                continue
            
            nx_graph.add_edge(
                relation.source_entity_id,
                relation.target_entity_id,
                relation_type=relation.relation_type,
                confidence=relation.confidence
            )
        
        return nx_graph
    
    def _apply_layout(self, graph: nx.DiGraph, layout: str) -> Dict[str, tuple]:
        """
        Apply layout algorithm to graph.
        
        Args:
            graph: networkx graph.
            layout: Layout algorithm name.
            
        Returns:
            Dictionary mapping node IDs to (x, y) positions.
        """
        if layout == 'spring':
            return nx.spring_layout(graph, k=1, iterations=50)
        elif layout == 'circular':
            return nx.circular_layout(graph)
        elif layout == 'hierarchical':
            try:
                return nx.nx_agraph.graphviz_layout(graph, prog='dot')
            except:
                # Fallback to spring if graphviz not available
                return nx.spring_layout(graph, k=1, iterations=50)
        elif layout == 'kamada_kawai':
            try:
                return nx.kamada_kawai_layout(graph)
            except:
                return nx.spring_layout(graph, k=1, iterations=50)
        else:
            return nx.spring_layout(graph, k=1, iterations=50)
    
    def _get_node_colors(self, nx_graph: nx.DiGraph, graph: KnowledgeGraph) -> List[str]:
        """
        Get colors for nodes based on entity types.
        
        Args:
            nx_graph: networkx graph.
            graph: Original KnowledgeGraph.
            
        Returns:
            List of colors for each node.
        """
        colors = []
        for node_id in nx_graph.nodes():
            entity = graph.get_entity(node_id)
            if entity:
                entity_type = entity.entity_type
                color = self.entity_type_colors.get(entity_type, self.default_node_color)
                colors.append(color)
            else:
                colors.append(self.default_node_color)
        return colors
    
    def _get_node_labels(self, nx_graph: nx.DiGraph, graph: KnowledgeGraph) -> Dict[str, str]:
        """
        Get labels for nodes (entity names).
        
        Args:
            nx_graph: networkx graph.
            graph: Original KnowledgeGraph.
            
        Returns:
            Dictionary mapping node IDs to labels.
        """
        labels = {}
        for node_id in nx_graph.nodes():
            entity = graph.get_entity(node_id)
            if entity:
                # Truncate long names
                name = entity.name
                if len(name) > 20:
                    name = name[:17] + "..."
                labels[node_id] = name
        return labels
    
    def _get_edge_labels(self, nx_graph: nx.DiGraph, graph: KnowledgeGraph) -> Dict[tuple, str]:
        """
        Get labels for edges (relation types).
        
        Args:
            nx_graph: networkx graph.
            graph: Original KnowledgeGraph.
            
        Returns:
            Dictionary mapping (source, target) tuples to relation type labels.
        """
        labels = {}
        for source, target in nx_graph.edges():
            # Find relation
            for relation in graph.relations.values():
                if (relation.source_entity_id == source and 
                    relation.target_entity_id == target):
                    # Truncate long relation types
                    rel_type = relation.relation_type
                    if len(rel_type) > 15:
                        rel_type = rel_type[:12] + "..."
                    labels[(source, target)] = rel_type
                    break
        return labels
    
    def _create_legend(
        self,
        graph: KnowledgeGraph,
        filter_entity_types: Optional[List[str]] = None
    ) -> List[mpatches.Patch]:
        """
        Create legend patches for entity types.
        
        Args:
            graph: KnowledgeGraph.
            filter_entity_types: Optional filter for entity types.
            
        Returns:
            List of legend patch objects.
        """
        entity_types = set(e.entity_type for e in graph.entities.values())
        
        if filter_entity_types:
            entity_types = entity_types.intersection(set(filter_entity_types))
        
        legend_elements = []
        for entity_type in sorted(entity_types):
            color = self.entity_type_colors.get(entity_type, self.default_node_color)
            patch = mpatches.Patch(color=color, label=entity_type)
            legend_elements.append(patch)
        
        return legend_elements

