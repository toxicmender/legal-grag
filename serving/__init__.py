"""
Backend and frontend serving module.

This module provides the API server, session management, and configuration
for serving the legal knowledge graph system.
"""

from .session_manager import SessionManager
from .config import ServerConfig

__all__ = [
    'SessionManager',
    'ServerConfig',
]

