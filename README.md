# Legal Knowledge Graph System

A comprehensive system for building, querying, and reasoning over legal knowledge graphs using large language models.

## Overview

This system provides an end-to-end pipeline for:
- **Document Ingestion**: Parse and process legal documents (PDF, DOC, text)
- **Knowledge Graph Construction**: Extract entities and relations to build structured knowledge graphs
- **Graph Embedding**: Learn vector representations of entities and relations
- **Retrieval**: Retrieve relevant subgraphs for user queries
- **Reasoning**: Generate responses using chain-of-thought reasoning
- **Explainability**: Provide explanations for model decisions

## Project Structure

```
project_root/
├── ingestion/              # Document ingestion + parsing
├── kg_construction/        # Build KG from ingested text
├── kg_embedding/          # Graph representation learning
├── retrieval/            # Subgraph retrieval + generation
├── prompting/            # Prompt engineering / chain-of-thought
├── explainability/       # Explainability & interpretability
├── serving/             # Backend + Frontend serving
├── tests/               # Unit / integration tests
├── scripts/             # Helper scripts
└── examples/            # Example usage and demos
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd legal-grag
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up MAYPL (for knowledge graph embedding):
```bash
# Option 1: Use setup script
python scripts/setup_maypl.py

# Option 2: Manual setup
git clone https://github.com/bdi-lab/MAYPL.git
export MAYPL_PATH=$(pwd)/MAYPL
cd MAYPL && pip install -r requirements.txt
```

5. Set up environment variables (create a `.env` file):
```env
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
LLM_API_KEY=your_api_key
MAYPL_PATH=/path/to/MAYPL  # Path to MAYPL repository
```

## Quick Start

### 1. Ingest Documents

```python
from ingestion.loader import DocumentLoader
from ingestion.parser import DocumentParser

loader = DocumentLoader()
parser = DocumentParser()

# Load a document
text = loader.load("path/to/document.pdf")
parsed = parser.parse("path/to/document.pdf")
```

### 2. Build Knowledge Graph

```python
from kg_construction.extractor import EntityRelationExtractor
from kg_construction.graph_builder import GraphBuilder

extractor = EntityRelationExtractor()
entities = extractor.extract_entities(text)
relations = extractor.extract_relations(text, entities=entities)

graph_builder = GraphBuilder()
graph = graph_builder.build_from_entities_relations(entities, relations)
```

### 3. Create Graph Embeddings

```python
from kg_embedding.maypl_wrapper import MAYPLWrapper

# Initialize MAYPL embedder
embedder = MAYPLWrapper(embedding_dim=128)

# Prepare graph data
graph_data = {
    'entities': list(graph.entities.values()),
    'relations': list(graph.relations.values())
}

# Train the model
embedder.fit(graph_data, train_epochs=100)

# Extract embeddings
entity_embedding = embedder.embed_entity("entity_id", {"id": "entity_id"})
relation_embedding = embedder.embed_relation("relation_id", {"id": "relation_id"})
```

### 4. Query the Knowledge Graph

```python
from retrieval.retriever import SubgraphRetriever
from retrieval.integration import RetrievalIntegration

retriever = SubgraphRetriever(graph=graph)
integration = RetrievalIntegration(retriever=retriever)

result = integration.retrieve_context("What are the key legal concepts?", top_k=5)
print(result['context'])
```

### 5. Run the API Server

```bash
python -m serving.api.main
```

The API will be available at `http://localhost:8000`.

## Usage Examples

See the `examples/` directory for:
- `demo_pipeline.ipynb`: End-to-end example notebook
- Additional usage examples

## Testing

Run tests with pytest:

```bash
pytest tests/
```

Run specific test modules:

```bash
pytest tests/ingestion_tests.py
pytest tests/kg_construction_tests.py
```

## Configuration

Configuration can be set via:
1. Environment variables (see `.env` file)
2. Configuration files (YAML/JSON)
3. Programmatic configuration in code

See `serving/config.py` for configuration options.

## Development

### Code Formatting

```bash
black .
```

### Linting

```bash
flake8 .
```

### Type Checking

```bash
mypy .
```

## Architecture

### Ingestion Module
- **loader.py**: High-level document loading
- **parser.py**: Text extraction from PDFs/DOCs
- **chunker.py**: Document chunking strategies
- **metadata.py**: Metadata extraction

### Knowledge Graph Construction
- **distiller.py**: Document distillation into semantic blocks
- **extractor.py**: Entity and relation extraction using LLM
- **graph_builder.py**: Graph construction interface
- **models.py**: Data models (Entity, Relation, KnowledgeGraph)
- **storage.py**: Storage backend abstraction (Neo4j, etc.)

### Graph Embedding
- **embedder_interface.py**: Embedding interface
- **maypl_wrapper.py**: MAYPL algorithm wrapper ([MAYPL repository](https://github.com/bdi-lab/MAYPL.git))
  - Structural representation learning for hyper-relational knowledge graphs
  - ICML 2025: "Structure Is All You Need"
- **fallback_embedder.py**: Fallback to other KGE libraries
- **cache.py**: Embedding caching and persistence
- **utils.py**: Utility functions

### Retrieval
- **retriever.py**: Subgraph retrieval algorithms
- **ranking.py**: Ranking candidate subgraphs
- **graph_to_context.py**: Convert subgraphs to text
- **integration.py**: Integration with LLM and prompts

### Prompting
- **base_prompt.py**: Prompt templates and builders
- **chain_of_thought.py**: Chain-of-thought reasoning
- **prompt_config.py**: LLM configuration

### Explainability
- **circuit_tracer.py**: Trace reasoning through LLM
- **shap_wrapper.py**: SHAP integration for attribution
- **attribution.py**: High-level attribution API
- **visualization.py**: Visualization tools

### Serving
- **api/main.py**: FastAPI application
- **api/routes.py**: API route definitions
- **session_manager.py**: Conversation session management
- **config.py**: Server configuration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

[Specify your license here]

## Acknowledgments

[Add acknowledgments if applicable]

