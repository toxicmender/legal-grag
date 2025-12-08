"""
Manage conversation sessions, history, state (using e.g. openscilab Memor module).
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid


class Session:
    """Represents a conversation session."""
    
    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize a session.
        
        Args:
            session_id: Optional session ID. If not provided, generates a new one.
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.messages: List[Dict[str, Any]] = []
        self.attachments: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Add a message to the session.
        
        Args:
            role: Message role ('user', 'assistant', 'system').
            content: Message content.
            metadata: Optional message metadata.
        """
        message = {
            'id': str(uuid.uuid4()),
            'role': role,
            'content': content,
            'timestamp': datetime.now(),
            'metadata': metadata or {}
        }
        self.messages.append(message)
        self.updated_at = datetime.now()
        return message
    
    def add_attachment(self, attachment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add an attachment to the session metadata.
        
        Args:
            attachment: Attachment metadata (e.g., filename, url, content_type, size).
        """
        attachment_record = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.now(),
            **attachment
        }
        self.attachments.append(attachment_record)
        self.updated_at = datetime.now()
        return attachment_record
    
    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get conversation history.
        
        Args:
            limit: Optional limit on number of messages to return.
            
        Returns:
            List of message dictionaries.
        """
        if limit:
            return self.messages[-limit:]
        return self.messages
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert session to dictionary.
        
        Returns:
            Dictionary representation of the session.
        """
        return {
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'message_count': len(self.messages),
            'attachment_count': len(self.attachments),
            'metadata': self.metadata
        }


class SessionManager:
    """
    Manages conversation sessions, history, and state.
    
    Can integrate with openscilab Memor module or other memory systems.
    """
    
    def __init__(self):
        """Initialize the session manager."""
        self.sessions: Dict[str, Session] = {}
        # TODO: Integrate with openscilab Memor module if needed
    
    def create_session(self, session_id: Optional[str] = None) -> Session:
        """
        Create a new session.
        
        Args:
            session_id: Optional session ID. If not provided, generates a new one.
            
        Returns:
            Created Session object.
        """
        if session_id and session_id in self.sessions:
            raise ValueError(f"Session {session_id} already exists")
        
        session = Session(session_id=session_id)
        self.sessions[session.session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get a session by ID.
        
        Args:
            session_id: Session ID.
            
        Returns:
            Session object or None if not found.
        """
        return self.sessions.get(session_id)
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> Session:
        """
        Get an existing session or create a new one.
        
        Args:
            session_id: Optional session ID.
            
        Returns:
            Session object.
        """
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        return self.create_session(session_id)
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: Session ID.
            
        Returns:
            True if session was deleted, False if not found.
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all sessions.
        
        Returns:
            List of session dictionaries.
        """
        return [session.to_dict() for session in self.sessions.values()]
    
    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """
        Clean up old sessions.
        
        Args:
            max_age_hours: Maximum age in hours before session is considered old.
            
        Returns:
            Number of sessions deleted.
        """
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        
        deleted = 0
        session_ids_to_delete = []
        
        for session_id, session in self.sessions.items():
            if session.updated_at < cutoff:
                session_ids_to_delete.append(session_id)
        
        for session_id in session_ids_to_delete:
            self.delete_session(session_id)
            deleted += 1
        
        return deleted

