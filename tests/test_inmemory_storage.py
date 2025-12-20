from kg.storage import InMemoryKGStorage
from kg.construction import KnowledgeGraph, Entity, Relation


def test_inmemory_save_load():
    store = InMemoryKGStorage()
    kg = KnowledgeGraph()

    e1 = Entity(id="e1", label="Alice", type="Person", metadata={"age": 30})
    e2 = Entity(id="e2", label="Bob", type="Person", metadata={"age": 40})
    kg.add_entity(e1)
    kg.add_entity(e2)

    rel = Relation(id="r1", type="knows", source=e1, target=e2, metadata={"since": 2020})
    kg.add_relation(rel)

    store.save_graph(kg, key="testkg")

    loaded = store.load_graph(key="testkg")
    assert isinstance(loaded, KnowledgeGraph)
    assert "e1" in loaded.entities
    assert "e2" in loaded.entities
    assert any(r.type == "knows" for r in loaded.relations)
