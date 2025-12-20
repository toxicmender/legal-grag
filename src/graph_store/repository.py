"""Simplified repository API for the graph store.

Defines an abstract interface and a Neo4j implementation.
"""

# Existing simple implementation kept for now
# src/graph_store/repository.py

from abc import ABC, abstractmethod
from typing import Any, Dict
from kg.storage import KGStorageInterface
from kg.construction import KnowledgeGraph, Entity, Relation

class GraphRepository(ABC):
    """Repository interface for storing and retrieving graph data."""
    @abstractmethod
    def connect(self) -> None:
        pass

    @abstractmethod
    def add_graph(self, graph_data: Any) -> None:
        """Add nodes & edges to the store."""
        pass

    @abstractmethod
    def query_subgraph(self, criteria: Dict) -> Any:
        """Retrieve subgraph matching some criteria (e.g. by entities, types)."""
        pass

    @abstractmethod
    def close(self) -> None:
        pass

class Neo4jRepository(GraphRepository, KGStorageInterface):
    def __init__(self, uri: str, user: str, password: str):
        # store config
        self._uri = uri
        self._user = user
        self._password = password
        self._driver = None

    def connect(self) -> None:
        from neo4j import GraphDatabase
        self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))

    def add_graph(self, graph_data: Any) -> None:
        # Convert either a KnowledgeGraph or a dict/GraphData-like object into
        # nodes and relationships and persist them to Neo4j.
        if self._driver is None:
            self.connect()

        # Normalize input to nodes/relations lists
        nodes = []
        rels = []

        if isinstance(graph_data, KnowledgeGraph):
            for eid, ent in graph_data.entities.items():
                nodes.append({
                    "id": ent.id,
                    "label": ent.label,
                    "type": ent.type,
                    "metadata": ent.metadata,
                })
            # relations list
            for rel in graph_data.relations:
                rels.append({
                    "id": getattr(rel, "id", None),
                    "type": rel.type,
                    "source_id": rel.source.id,
                    "target_id": rel.target.id,
                    "metadata": rel.metadata,
                })
            # triples may contain additional relation objects
            for tri in getattr(graph_data, "triples", []):
                rel = tri.relation
                rels.append({
                    "id": getattr(rel, "id", None),
                    "type": rel.type,
                    "source_id": tri.head.id,
                    "target_id": tri.tail.id,
                    "metadata": rel.metadata,
                })
        elif isinstance(graph_data, dict):
            # expect {'nodes': [...], 'edges': [...]} format
            nodes = graph_data.get("nodes", [])
            rels = graph_data.get("edges", [])
        else:
            # try to treat as generic iterable
            raise TypeError("Unsupported graph_data type for add_graph")

        # Write to Neo4j using UNWIND for batch writes
        with self._driver.session() as session:
            if nodes:
                session.run(
                    """
                    UNWIND $nodes AS node
                    MERGE (n:Entity {id: node.id})
                    SET n.label = node.label, n.type = node.type, n.metadata = node.metadata
                    """,
                    nodes=nodes,
                )

            if rels:
                session.run(
                    """
                    UNWIND $rels AS rel
                    MATCH (s:Entity {id: rel.source_id}), (t:Entity {id: rel.target_id})
                    MERGE (s)-[r:RELATED {id: rel.id}]->(t)
                    SET r.type = rel.type, r.metadata = rel.metadata
                    """,
                    rels=rels,
                )

    # KGStorageInterface compatibility
    def save_graph(self, kg) -> None:
        """Adapter method to save a KnowledgeGraph via repository.

        This is a simple shim that reuses `add_graph`. Real implementations
        would map the `kg` object to the repository's expected `graph_data`.
        """
        # For now, forward to add_graph (expects GraphData-like structure)
        self.add_graph(kg)

    def load_graph(self, query: str):
        """Adapter method to query and return a KnowledgeGraph-like object.

        Real implementations should translate the `query`/criteria and
        reconstruct a `KnowledgeGraph` object.
        """
        return self.query_subgraph({"query": query})

    def query_subgraph(self, criteria: Dict) -> Any:
        # Accept either a simple text 'query' (search by label) or arbitrary
        # criteria dict. Return a KnowledgeGraph constructed from results.
        if self._driver is None:
            self.connect()

        q = None
        if isinstance(criteria, dict):
            q = criteria.get("query")
        else:
            q = str(criteria)

        kg = KnowledgeGraph()

        with self._driver.session() as session:
            if q:
                # search nodes by label containing query (case-insensitive)
                result = session.run(
                    """
                    MATCH (n:Entity)
                    WHERE toLower(n.label) CONTAINS toLower($q)
                    OPTIONAL MATCH (n)-[r]->(m)
                    RETURN n, r, m LIMIT 250
                    """,
                    q=q,
                )
            else:
                # fallback: return all nodes and their relationships (capped)
                result = session.run(
                    """
                    MATCH (n:Entity)
                    OPTIONAL MATCH (n)-[r]->(m)
                    RETURN n, r, m LIMIT 250
                    """,
                )

            for record in result:
                node = record.get("n")
                if node is not None:
                    props = dict(node)
                    ent = Entity(id=props.get("id"), label=props.get("label"), type=props.get("type"), metadata=props.get("metadata"))
                    kg.add_entity(ent)

                rel = record.get("r")
                if rel is not None:
                    rel_props = dict(rel)
                    source = record.get("n")
                    target = record.get("m")
                    if source is None or target is None:
                        continue
                    sprops = dict(source)
                    tprops = dict(target)
                    s_ent = Entity(id=sprops.get("id"), label=sprops.get("label"), type=sprops.get("type"), metadata=sprops.get("metadata"))
                    t_ent = Entity(id=tprops.get("id"), label=tprops.get("label"), type=tprops.get("type"), metadata=tprops.get("metadata"))
                    kg.add_entity(s_ent)
                    kg.add_entity(t_ent)
                    relation = Relation(id=rel_props.get("id"), type=rel_props.get("type"), source=s_ent, target=t_ent, metadata=rel_props.get("metadata"))
                    kg.add_relation(relation)

        return kg

    def close(self) -> None:
        if self._driver:
            self._driver.close()
