"""
Unit tests for serving module.
"""

import unittest
from serving.session_manager import SessionManager, Session
from serving.config import ServerConfig


class TestSessionManager(unittest.TestCase):
    """Tests for SessionManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = SessionManager()
    
    def test_create_session(self):
        """Test session creation."""
        session = self.manager.create_session()
        self.assertIsNotNone(session)
        self.assertIsNotNone(session.session_id)
    
    def test_get_session(self):
        """Test session retrieval."""
        session = self.manager.create_session()
        retrieved = self.manager.get_session(session.session_id)
        self.assertEqual(session, retrieved)
    
    def test_delete_session(self):
        """Test session deletion."""
        session = self.manager.create_session()
        session_id = session.session_id
        self.assertTrue(self.manager.delete_session(session_id))
        self.assertIsNone(self.manager.get_session(session_id))


class TestSession(unittest.TestCase):
    """Tests for Session."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.session = Session()
    
    def test_add_message(self):
        """Test adding messages to session."""
        self.session.add_message("user", "Hello")
        self.assertEqual(len(self.session.messages), 1)
        self.assertEqual(self.session.messages[0]['role'], "user")
        self.assertEqual(self.session.messages[0]['content'], "Hello")
    
    def test_get_history(self):
        """Test getting conversation history."""
        self.session.add_message("user", "Hello")
        self.session.add_message("assistant", "Hi there")
        
        history = self.session.get_history()
        self.assertEqual(len(history), 2)
        
        limited = self.session.get_history(limit=1)
        self.assertEqual(len(limited), 1)


class TestServerConfig(unittest.TestCase):
    """Tests for ServerConfig."""
    
    def test_config_initialization(self):
        """Test config initialization."""
        config = ServerConfig()
        self.assertIsNotNone(config.database)
        self.assertIsNotNone(config.llm)
        self.assertIsNotNone(config.storage)
    
    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            'host': 'localhost',
            'port': 9000,
            'database': {'uri': 'neo4j://localhost:7687'},
            'llm': {'model_name': 'gpt-3.5-turbo'}
        }
        config = ServerConfig.from_dict(config_dict)
        self.assertEqual(config.host, 'localhost')
        self.assertEqual(config.port, 9000)
        self.assertEqual(config.database.uri, 'neo4j://localhost:7687')
        self.assertEqual(config.llm.model_name, 'gpt-3.5-turbo')


if __name__ == '__main__':
    unittest.main()

