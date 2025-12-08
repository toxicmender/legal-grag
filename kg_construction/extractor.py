"""
Entity and relation extraction using LLM via itext2kg module.
"""

from typing import List, Dict, Any, Optional
import uuid
from .models import Entity, Relation, Statement

try:
    from itext2kg import iText2KG
    ITEXT2KG_AVAILABLE = True
except ImportError:
    ITEXT2KG_AVAILABLE = False
    iText2KG = None


class EntityRelationExtractor:
    """
    Wrapper around entity and relation extraction using itext2kg.
    
    Extracts entities (people, organizations, concepts, etc.) and relations
    between them from text using the itext2kg module which leverages LLMs.
    """
    
    def __init__(
        self, 
        openai_api_key: Optional[str] = None,
        llm_client=None, 
        model_name: Optional[str] = None
    ):
        """
        Initialize the entity-relation extractor.
        
        Args:
            openai_api_key: OpenAI API key for itext2kg. If not provided, 
                          will try to get from environment variable OPENAI_API_KEY.
            llm_client: Optional LLM client instance (for future use).
            model_name: Optional model name to use for extraction.
        """
        if not ITEXT2KG_AVAILABLE:
            raise ImportError(
                "itext2kg is not installed. Please install it using: pip install itext2kg"
            )
        
        import os
        # Get API key from parameter or environment
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Provide it as parameter or set OPENAI_API_KEY environment variable."
            )
        
        self.itext2kg = iText2KG(openai_api_key=api_key)
        self.llm_client = llm_client
        self.model_name = model_name
        self._entity_map: Dict[str, str] = {}  # Map entity names to IDs
    
    def extract_entities(self, text: str, entity_types: Optional[List[str]] = None) -> List[Entity]:
        """
        Extract entities from text using itext2kg.
        
        Args:
            text: Text content to extract entities from.
            entity_types: Optional list of entity types to focus on (not used by itext2kg).
            
        Returns:
            List of Entity objects.
        """
        # Use itext2kg to extract entities and relations
        # itext2kg works with sections, so we'll split text into sections
        sections = self._prepare_sections(text)
        
        # Build graph using itext2kg
        global_entities, global_relations = self.itext2kg.build_graph(sections=sections)
        
        # Convert itext2kg entities to our Entity model
        entities = self._convert_itext2kg_entities(global_entities)
        
        return entities
    
    def extract_relations(self, text: str, entities: Optional[List[Entity]] = None) -> List[Relation]:
        """
        Extract relations from text using itext2kg.
        
        Args:
            text: Text content to extract relations from.
            entities: Optional pre-extracted entities. If not provided, will extract them.
            
        Returns:
            List of Relation objects.
        """
        # Use itext2kg to extract entities and relations
        sections = self._prepare_sections(text)
        
        # Build graph using itext2kg
        global_entities, global_relations = self.itext2kg.build_graph(sections=sections)
        
        # Convert entities if not provided
        if entities is None:
            entities = self._convert_itext2kg_entities(global_entities)
        
        # Build entity map for relation conversion
        self._build_entity_map(entities)
        
        # Convert itext2kg relations to our Relation model
        relations = self._convert_itext2kg_relations(global_relations, entities)
        
        return relations
    
    def extract_statements(self, text: str) -> List[Statement]:
        """
        Extract complete statements (triples) from text using itext2kg.
        
        Args:
            text: Text content to extract statements from.
            
        Returns:
            List of Statement (triple) objects.
        """
        sections = self._prepare_sections(text)
        global_entities, global_relations = self.itext2kg.build_graph(sections=sections)
        
        statements = []
        for relation in global_relations:
            # itext2kg relations typically have head, relation, tail structure
            # Convert to Statement format
            statement = Statement(
                subject=relation.get('head', ''),
                predicate=relation.get('relation', ''),
                object=relation.get('tail', ''),
                confidence=relation.get('confidence'),
                source=relation.get('source'),
                metadata=relation.get('metadata', {})
            )
            statements.append(statement)
        
        return statements
    
    def extract_batch(self, texts: List[str]) -> Dict[str, Any]:
        """
        Extract entities and relations from multiple texts in batch.
        
        Args:
            texts: List of text contents.
            
        Returns:
            Dictionary with 'entities' and 'relations' keys.
        """
        all_entities = []
        all_relations = []
        
        for text in texts:
            entities = self.extract_entities(text)
            relations = self.extract_relations(text, entities=entities)
            all_entities.extend(entities)
            all_relations.extend(relations)
        
        return {
            'entities': all_entities,
            'relations': all_relations
        }
    
    def extract_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract both entities and relations from text in one call.
        
        Args:
            text: Text content to extract from.
            
        Returns:
            Dictionary with 'entities' and 'relations' keys.
        """
        entities = self.extract_entities(text)
        relations = self.extract_relations(text, entities=entities)
        
        return {
            'entities': entities,
            'relations': relations
        }
    
    def _prepare_sections(self, text: str) -> List[str]:
        """
        Prepare text sections for itext2kg.
        
        Args:
            text: Raw text content.
            
        Returns:
            List of text sections.
        """
        # Split text into semantic sections (paragraphs)
        # itext2kg works best with meaningful sections
        sections = [s.strip() for s in text.split('\n\n') if s.strip()]
        
        # If sections are too long, split them further
        max_section_length = 2000  # characters
        final_sections = []
        for section in sections:
            if len(section) > max_section_length:
                # Split long sections by sentences
                sentences = section.split('. ')
                current_section = ""
                for sentence in sentences:
                    if len(current_section) + len(sentence) < max_section_length:
                        current_section += sentence + ". "
                    else:
                        if current_section:
                            final_sections.append(current_section.strip())
                        current_section = sentence + ". "
                if current_section:
                    final_sections.append(current_section.strip())
            else:
                final_sections.append(section)
        
        return final_sections
    
    def _convert_itext2kg_entities(self, global_entities: List[Dict[str, Any]]) -> List[Entity]:
        """
        Convert itext2kg entity format to our Entity model.
        
        Args:
            global_entities: List of entities from itext2kg.
            
        Returns:
            List of Entity objects.
        """
        entities = []
        for idx, entity_data in enumerate(global_entities):
            # Generate unique ID for entity
            entity_id = str(uuid.uuid4())
            
            # Extract entity information
            # itext2kg entities may have different structures, adapt as needed
            entity_name = entity_data.get('name', entity_data.get('text', f'Entity_{idx}'))
            entity_type = entity_data.get('type', entity_data.get('label', 'UNKNOWN'))
            
            # Create Entity object
            entity = Entity(
                id=entity_id,
                name=entity_name,
                entity_type=entity_type,
                properties=entity_data.get('properties', {}),
                metadata={
                    'source': 'itext2kg',
                    'original_data': entity_data
                }
            )
            
            entities.append(entity)
            # Store mapping for relation conversion
            self._entity_map[entity_name] = entity_id
        
        return entities
    
    def _convert_itext2kg_relations(
        self, 
        global_relations: List[Dict[str, Any]], 
        entities: List[Entity]
    ) -> List[Relation]:
        """
        Convert itext2kg relation format to our Relation model.
        
        Args:
            global_relations: List of relations from itext2kg.
            entities: List of Entity objects for ID lookup.
            
        Returns:
            List of Relation objects.
        """
        relations = []
        
        # Build name to entity mapping
        name_to_entity = {e.name: e for e in entities}
        
        for idx, relation_data in enumerate(global_relations):
            # Extract relation information
            # itext2kg relations typically have head, relation, tail structure
            head_name = relation_data.get('head', relation_data.get('source', ''))
            tail_name = relation_data.get('tail', relation_data.get('target', ''))
            relation_type = relation_data.get('relation', relation_data.get('type', 'RELATED_TO'))
            
            # Find entity IDs
            source_entity = name_to_entity.get(head_name)
            target_entity = name_to_entity.get(tail_name)
            
            if source_entity and target_entity:
                relation_id = str(uuid.uuid4())
                
                relation = Relation(
                    id=relation_id,
                    source_entity_id=source_entity.id,
                    target_entity_id=target_entity.id,
                    relation_type=relation_type,
                    properties=relation_data.get('properties', {}),
                    metadata={
                        'source': 'itext2kg',
                        'head_name': head_name,
                        'tail_name': tail_name,
                        'original_data': relation_data
                    },
                    confidence=relation_data.get('confidence')
                )
                
                relations.append(relation)
        
        return relations
    
    def _build_entity_map(self, entities: List[Entity]) -> None:
        """
        Build mapping from entity names to IDs.
        
        Args:
            entities: List of Entity objects.
        """
        self._entity_map = {e.name: e.id for e in entities}

