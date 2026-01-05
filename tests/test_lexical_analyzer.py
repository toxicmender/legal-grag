"""
Unit tests for LexicalAnalyzer module.
"""

import pytest
from ingestion.lexical_analyzer import LexicalAnalyzer, NLTK_AVAILABLE


@pytest.mark.skipif(not NLTK_AVAILABLE, reason="NLTK not available")
class TestLexicalAnalyzer:
    """Tests for LexicalAnalyzer class."""
    
    def test_initialization(self):
        """Test LexicalAnalyzer initialization."""
        analyzer = LexicalAnalyzer()
        assert analyzer is not None
        assert analyzer.use_spacy is False
    
    def test_initialization_without_nltk(self, monkeypatch):
        """Test initialization fails without NLTK."""
        monkeypatch.setattr('ingestion.lexical_analyzer.NLTK_AVAILABLE', False)
        with pytest.raises(ImportError):
            LexicalAnalyzer()
    
    def test_tokenize_basic(self, sample_text):
        """Test basic tokenization."""
        analyzer = LexicalAnalyzer()
        tokens = analyzer.tokenize(sample_text)
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert all(isinstance(token, str) for token in tokens)
    
    def test_tokenize_empty_text(self):
        """Test tokenization of empty text."""
        analyzer = LexicalAnalyzer()
        tokens = analyzer.tokenize("")
        assert tokens == []
    
    def test_tokenize_whitespace_only(self):
        """Test tokenization of whitespace-only text."""
        analyzer = LexicalAnalyzer()
        tokens = analyzer.tokenize("   \n\n   ")
        assert len(tokens) == 0 or all(not token.strip() for token in tokens)
    
    def test_sent_tokenize_basic(self, sample_text):
        """Test sentence tokenization."""
        analyzer = LexicalAnalyzer()
        sentences = analyzer.sent_tokenize(sample_text)
        assert isinstance(sentences, list)
        assert len(sentences) > 0
        assert all(isinstance(sent, str) for sent in sentences)
    
    def test_sent_tokenize_empty(self):
        """Test sentence tokenization of empty text."""
        analyzer = LexicalAnalyzer()
        sentences = analyzer.sent_tokenize("")
        assert sentences == []
    
    def test_pos_tag_basic(self, sample_text):
        """Test POS tagging."""
        analyzer = LexicalAnalyzer()
        pos_tags = analyzer.pos_tag(sample_text)
        assert isinstance(pos_tags, list)
        assert len(pos_tags) > 0
        assert all(isinstance(tag, tuple) and len(tag) == 2 for tag in pos_tags)
        assert all(isinstance(word, str) and isinstance(pos, str) for word, pos in pos_tags)
    
    def test_pos_tag_empty(self):
        """Test POS tagging of empty text."""
        analyzer = LexicalAnalyzer()
        pos_tags = analyzer.pos_tag("")
        assert pos_tags == []
    
    def test_get_word_frequencies_from_tokens(self):
        """Test word frequency calculation from tokens."""
        analyzer = LexicalAnalyzer()
        tokens = ["the", "quick", "brown", "fox", "the", "quick"]
        frequencies = analyzer.get_word_frequencies(tokens=tokens)
        assert frequencies["the"] == 2
        assert frequencies["quick"] == 2
        assert frequencies["brown"] == 1
        assert frequencies["fox"] == 1
    
    def test_get_word_frequencies_from_text(self):
        """Test word frequency calculation from text."""
        analyzer = LexicalAnalyzer()
        text = "the quick brown fox the quick"
        frequencies = analyzer.get_word_frequencies(text=text)
        assert "the" in frequencies
        assert "quick" in frequencies
    
    def test_get_word_frequencies_empty(self):
        """Test word frequency with empty input."""
        analyzer = LexicalAnalyzer()
        frequencies = analyzer.get_word_frequencies()
        assert frequencies == {}
    
    def test_get_statistics_basic(self, sample_text):
        """Test text statistics calculation."""
        analyzer = LexicalAnalyzer()
        stats = analyzer.get_statistics(sample_text)
        
        assert 'character_count' in stats
        assert 'word_count' in stats
        assert 'sentence_count' in stats
        assert 'average_word_length' in stats
        assert 'average_sentence_length' in stats
        assert 'unique_words' in stats
        
        assert stats['character_count'] > 0
        assert stats['word_count'] > 0
        assert stats['sentence_count'] > 0
        assert stats['average_word_length'] > 0
        assert stats['unique_words'] > 0
    
    def test_get_statistics_empty(self):
        """Test statistics for empty text."""
        analyzer = LexicalAnalyzer()
        stats = analyzer.get_statistics("")
        
        assert stats['character_count'] == 0
        assert stats['word_count'] == 0
        assert stats['sentence_count'] == 0
        assert stats['average_word_length'] == 0.0
    
    def test_preprocess_basic(self):
        """Test text preprocessing."""
        analyzer = LexicalAnalyzer()
        text = "  This   is   a   test  \n\n\n  with   multiple   spaces  "
        processed = analyzer.preprocess(text)
        assert processed == "This is a test \n\n with multiple spaces"
    
    def test_preprocess_empty(self):
        """Test preprocessing of empty text."""
        analyzer = LexicalAnalyzer()
        processed = analyzer.preprocess("")
        assert processed == ""
    
    def test_preprocess_normalize_whitespace(self):
        """Test whitespace normalization."""
        analyzer = LexicalAnalyzer()
        text = "word1    word2\tword3\nword4"
        processed = analyzer.preprocess(text)
        assert "    " not in processed
        assert "\t" not in processed
    
    def test_analyze_full(self, sample_text):
        """Test full lexical analysis."""
        analyzer = LexicalAnalyzer()
        result = analyzer.analyze(sample_text)
        
        assert 'tokens' in result
        assert 'sentences' in result
        assert 'pos_tags' in result
        assert 'word_frequencies' in result
        assert 'statistics' in result
        assert 'processed_text' in result
        assert 'stopwords_removed' in result
        
        assert isinstance(result['tokens'], list)
        assert isinstance(result['sentences'], list)
        assert isinstance(result['pos_tags'], list)
        assert isinstance(result['word_frequencies'], dict)
        assert isinstance(result['statistics'], dict)
        assert isinstance(result['processed_text'], str)
    
    def test_analyze_empty_text(self):
        """Test analysis of empty text."""
        analyzer = LexicalAnalyzer()
        result = analyzer.analyze("")
        
        assert result['tokens'] == []
        assert result['sentences'] == []
        assert result['pos_tags'] == []
        assert result['word_frequencies'] == {}
        assert result['processed_text'] == ''
    
    def test_analyze_whitespace_only(self):
        """Test analysis of whitespace-only text."""
        analyzer = LexicalAnalyzer()
        result = analyzer.analyze("   \n\n   ")
        
        # Should return empty analysis
        assert len(result['tokens']) == 0 or all(not t.strip() for t in result['tokens'])
    
    def test_analyze_special_characters(self):
        """Test analysis with special characters."""
        analyzer = LexicalAnalyzer()
        text = "Hello! This is a test. @#$%^&*()"
        result = analyzer.analyze(text)
        
        assert len(result['tokens']) > 0
        assert result['statistics']['word_count'] > 0
    
    def test_get_dependency_parse_without_spacy(self):
        """Test dependency parsing without spaCy."""
        analyzer = LexicalAnalyzer(use_spacy=False)
        result = analyzer.get_dependency_parse("Test text")
        assert result is None
    
    @pytest.mark.skipif(True, reason="spaCy tests require spaCy installation")
    def test_get_dependency_parse_with_spacy(self):
        """Test dependency parsing with spaCy."""
        analyzer = LexicalAnalyzer(use_spacy=True)
        result = analyzer.get_dependency_parse("The quick brown fox jumps.")
        
        if result is not None:
            assert isinstance(result, list)
            assert len(result) > 0
            assert 'token' in result[0]
            assert 'dep' in result[0]
            assert 'head_token' in result[0]

