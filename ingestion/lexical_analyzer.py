"""
Lexical Analysis module for text processing.

Performs tokenization, POS tagging, word frequency analysis, and text statistics
on text extracted from documents before feeding to knowledge graph construction.
"""

from typing import Dict, Any, List, Tuple, Optional
import re
from collections import Counter

try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.tag import pos_tag
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
    
    def _ensure_nltk_data():
        """Ensure NLTK data is downloaded (lazy loading)."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
except ImportError:
    NLTK_AVAILABLE = False
    word_tokenize = None
    sent_tokenize = None
    pos_tag = None
    stopwords = None
    
    def _ensure_nltk_data():
        """Placeholder when NLTK not available."""
        pass

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None


class LexicalAnalyzer:
    """
    Performs lexical analysis on text including tokenization, POS tagging,
    word frequency analysis, and text statistics.
    """
    
    def __init__(self, use_spacy: bool = False, spacy_model: str = "en_core_web_sm"):
        """
        Initialize the lexical analyzer.
        
        Args:
            use_spacy: Whether to use spaCy for advanced NLP (requires spaCy installation).
            spacy_model: spaCy model to use if use_spacy is True.
            
        Raises:
            ImportError: If NLTK is not available.
        """
        if not NLTK_AVAILABLE:
            raise ImportError(
                "NLTK is required for lexical analysis. Install it using: pip install nltk"
            )
        
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        self.spacy_nlp = None
        
        if self.use_spacy:
            try:
                self.spacy_nlp = spacy.load(spacy_model)
            except OSError:
                # Model not found, fall back to NLTK only
                self.use_spacy = False
                print(f"Warning: spaCy model '{spacy_model}' not found. Using NLTK only.")
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Perform full lexical analysis on text.
        
        Args:
            text: Text content to analyze.
            
        Returns:
            Dictionary containing:
                - tokens: List of word tokens
                - sentences: List of sentences
                - pos_tags: List of (word, POS tag) tuples
                - word_frequencies: Dictionary of word frequencies
                - statistics: Text statistics dictionary
                - processed_text: Cleaned and normalized text ready for KG construction
                - stopwords_removed: List of stopwords found (if applicable)
        """
        if not text or not text.strip():
            return self._empty_analysis()
        
        # Preprocess text
        processed_text = self.preprocess(text)
        
        # Tokenize
        tokens = self.tokenize(processed_text)
        sentences = self.sent_tokenize(processed_text)
        
        # POS tagging
        pos_tags = self.pos_tag(processed_text)
        
        # Word frequency analysis
        word_frequencies = self.get_word_frequencies(tokens)
        
        # Text statistics
        statistics = self.get_statistics(processed_text, tokens, sentences)
        
        # Extract stopwords if needed
        stopwords_list = []
        if stopwords:
            try:
                stop_words = set(stopwords.words('english'))
                stopwords_list = [token for token in tokens if token.lower() in stop_words]
            except Exception:
                pass
        
        return {
            'tokens': tokens,
            'sentences': sentences,
            'pos_tags': pos_tags,
            'word_frequencies': word_frequencies,
            'statistics': statistics,
            'processed_text': processed_text,
            'stopwords_removed': stopwords_list
        }
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Text to tokenize.
            
        Returns:
            List of word tokens.
        """
        if not text:
            return []
        
        if _ensure_nltk_data:
            _ensure_nltk_data()
        
        try:
            tokens = word_tokenize(text)
            # Filter out punctuation-only tokens
            tokens = [token for token in tokens if re.match(r'^[a-zA-Z0-9]+', token)]
            return tokens
        except Exception as e:
            # Fallback to simple whitespace tokenization
            return text.split()
    
    def sent_tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into sentences.
        
        Args:
            text: Text to tokenize.
            
        Returns:
            List of sentences.
        """
        if not text:
            return []
        
        if _ensure_nltk_data:
            _ensure_nltk_data()
        
        try:
            return sent_tokenize(text)
        except Exception:
            # Fallback to simple sentence splitting
            return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    
    def pos_tag(self, text: str) -> List[Tuple[str, str]]:
        """
        Perform part-of-speech tagging on text.
        
        Args:
            text: Text to tag.
            
        Returns:
            List of (word, POS tag) tuples.
        """
        if not text:
            return []
        
        if _ensure_nltk_data:
            _ensure_nltk_data()
        
        tokens = self.tokenize(text)
        if not tokens:
            return []
        
        try:
            return pos_tag(tokens)
        except Exception:
            return [(token, 'UNKNOWN') for token in tokens]
    
    def get_word_frequencies(self, tokens: Optional[List[str]] = None, text: Optional[str] = None) -> Dict[str, int]:
        """
        Calculate word frequencies from tokens or text.
        
        Args:
            tokens: Optional list of tokens. If not provided, text will be tokenized.
            text: Optional text to tokenize if tokens not provided.
            
        Returns:
            Dictionary mapping words to their frequencies.
        """
        if tokens is None:
            if text is None:
                return {}
            tokens = self.tokenize(text)
        
        # Convert to lowercase for frequency counting
        tokens_lower = [token.lower() for token in tokens]
        return dict(Counter(tokens_lower))
    
    def get_statistics(self, text: str, tokens: Optional[List[str]] = None, sentences: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate text statistics.
        
        Args:
            text: Text content.
            tokens: Optional pre-tokenized tokens.
            sentences: Optional pre-tokenized sentences.
            
        Returns:
            Dictionary containing:
                - character_count: Total character count
                - word_count: Total word count
                - sentence_count: Total sentence count
                - average_word_length: Average length of words
                - average_sentence_length: Average words per sentence
                - unique_words: Number of unique words
        """
        if not text:
            return {
                'character_count': 0,
                'word_count': 0,
                'sentence_count': 0,
                'average_word_length': 0.0,
                'average_sentence_length': 0.0,
                'unique_words': 0
            }
        
        if tokens is None:
            tokens = self.tokenize(text)
        
        if sentences is None:
            sentences = self.sent_tokenize(text)
        
        character_count = len(text)
        word_count = len(tokens)
        sentence_count = len(sentences) if sentences else 1
        
        # Calculate average word length
        if tokens:
            total_word_length = sum(len(token) for token in tokens)
            average_word_length = total_word_length / len(tokens)
        else:
            average_word_length = 0.0
        
        # Calculate average sentence length (words per sentence)
        if sentence_count > 0:
            average_sentence_length = word_count / sentence_count
        else:
            average_sentence_length = 0.0
        
        # Count unique words
        unique_words = len(set(token.lower() for token in tokens))
        
        return {
            'character_count': character_count,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'average_word_length': round(average_word_length, 2),
            'average_sentence_length': round(average_sentence_length, 2),
            'unique_words': unique_words
        }
    
    def preprocess(self, text: str) -> str:
        """
        Preprocess and normalize text before analysis.
        
        Performs:
        - Whitespace normalization
        - Basic cleaning
        - Normalization for better tokenization
        
        Args:
            text: Raw text to preprocess.
            
        Returns:
            Preprocessed text ready for analysis and KG construction.
        """
        if not text:
            return ""
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Trim whitespace
        text = text.strip()
        
        return text
    
    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis structure for empty text."""
        return {
            'tokens': [],
            'sentences': [],
            'pos_tags': [],
            'word_frequencies': {},
            'statistics': self.get_statistics(""),
            'processed_text': '',
            'stopwords_removed': []
        }
    
    def get_dependency_parse(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """
        Perform dependency parsing using spaCy (if available).
        
        Args:
            text: Text to parse.
            
        Returns:
            List of dependency parse dictionaries, or None if spaCy not available.
            Each dictionary contains: token, dep, head_token, head_pos
        """
        if not self.use_spacy or not self.spacy_nlp:
            return None
        
        if not text:
            return []
        
        doc = self.spacy_nlp(text)
        dependencies = []
        
        for token in doc:
            dependencies.append({
                'token': token.text,
                'dep': token.dep_,
                'head_token': token.head.text,
                'head_pos': token.head.pos_
            })
        
        return dependencies

