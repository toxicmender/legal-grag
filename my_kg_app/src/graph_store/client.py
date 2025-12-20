"""Neo4j client wrapper (placeholder)."""

from neo4j import GraphDatabase


def get_driver(uri: str, user: str, password: str):
    """Return a neo4j driver instance (do not call in tests)."""
    return GraphDatabase.driver(uri, auth=(user, password))
