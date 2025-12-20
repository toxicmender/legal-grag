"""Convert graph subgraphs into text context for prompting."""

def graph_to_context(subgraph: dict) -> str:
    # Flatten simple subgraph to text (placeholder)
    nodes = subgraph.get("nodes", [])
    return "\n".join(str(n) for n in nodes)
