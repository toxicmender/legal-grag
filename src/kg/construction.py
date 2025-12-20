"""KG construction utilities: distillation, extraction and building.

This module consolidates `kg_construction` helpers: `distill`, `extract` and a
convenience `build_kg_from_text` that composes distillation + conversion.
"""
from typing import Dict, List, Optional, Any
from .builder import convert_text_to_kg


def distill(text: str) -> str:
    """Condense long text to salient facts for extraction.

    Placeholder implementation: truncate. Swap for a summarizer later.
    """
    return text[:200]


def extract(text: str) -> Dict[str, Any]:
    """Extract entities/relations from text (placeholder)."""
    return {"entities": [], "relations": []}


def build_kg_from_text(text: str) -> Dict[str, Any]:
    """High-level convenience: distill text then convert to KG dict."""
    distilled = distill(text)
    kg = convert_text_to_kg(distilled)
    return {"nodes": kg.get("nodes", []), "edges": kg.get("edges", [])}

class Entity:
    def __init__(self, id: str, label: str, type: Optional[str] = None, metadata: Optional[dict] = None):
        self.id = id
        self.label = label
        self.type = type
        self.metadata = metadata or {}

    def __repr__(self):
        return f"Entity(id={self.id}, label={self.label}, type={self.type})"

class Relation:
    def __init__(self, id: str, type: str, source: Entity, target: Entity, metadata: Optional[dict] = None):
        self.id = id
        self.type = type
        self.source = source
        self.target = target
        self.metadata = metadata or {}

    def __repr__(self):
        return (f"Relation(id={self.id}, type={self.type}, "
                f"source={self.source.id}, target={self.target.id})")

class Triple:
    def __init__(self, head: Entity, relation: Relation, tail: Entity):
        self.head = head
        self.relation = relation
        self.tail = tail

    def __repr__(self):
        return (f"Triple(head={self.head.id}, "
                f"relation={self.relation.type}, tail={self.tail.id})")

class KnowledgeGraph:
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self.triples: List[Triple] = []

    def add_entity(self, entity: Entity):
        if entity.id not in self.entities:
            self.entities[entity.id] = entity

    def add_relation(self, relation: Relation):
        self.relations.append(relation)

    def add_triple(self, head: Entity, relation: Relation, tail: Entity):
        self.add_entity(head)
        self.add_entity(tail)
        self.add_relation(relation)
        triple = Triple(head=head, relation=relation, tail=tail)
        self.triples.append(triple)

    def merge(self, other: "KnowledgeGraph", resolve_duplicates: bool = True):
        # simplistic merge: just add entities/relations/triples that don't exist yet
        for eid, ent in other.entities.items():
            if eid not in self.entities:
                self.entities[eid] = ent
        for rel in other.relations:
            self.relations.append(rel)
        for tri in other.triples:
            self.triples.append(tri)

    def __repr__(self):
        return (f"KnowledgeGraph(num_entities={len(self.entities)}, "
                f"num_relations={len(self.relations)}, num_triples={len(self.triples)})")

class KGExtractor:
    def __init__(self, llm_model, embedding_model=None, config: dict = None):
        """
        llm_model: an object or wrapper exposing a call() method to run LLM inference
        embedding_model: optional, for embedding-assisted extraction / disambiguation
        """
        self.llm = llm_model
        self.embedding_model = embedding_model
        self.config = config or {}

    def extract_entities(self, text_blocks: List[str]) -> List[Entity]:
        """
        Extract entities from list of text blocks.
        """
        entities: List[Entity] = []
        # TODO: implement call to LLM or other extractor
        return entities

    def extract_relations(self, text_blocks: List[str], entities: List[Entity]) -> List[Relation]:
        """
        Given the text blocks and extracted entities, extract relations among entities.
        """
        relations: List[Relation] = []
        # TODO: implement relation extraction via LLM + post-processing
        return relations

    def extract(self, text_blocks: List[str]) -> KnowledgeGraph:
        entities = self.extract_entities(text_blocks)
        relations = self.extract_relations(text_blocks, entities)
        kg = KnowledgeGraph()
        for ent in entities:
            kg.add_entity(ent)
        for rel in relations:
            kg.add_relation(rel)
        # Optionally build triples if your pipeline includes relations heads/tails
        # ...
        return kg

# Storage implementations are provided in a separate module to keep the
# construction utilities independent from any particular backend. Import
# the interface/adapter from `kg.storage` when you need a concrete storage.
