"""Thinking session management."""

import asyncio
import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .sequential import ThinkingRequest, ThinkingResponse


class ThinkingSession(BaseModel):
    """A thinking session."""
    
    session_id: str = Field(description="Session identifier")
    thoughts: List[ThinkingResponse] = Field(default_factory=list, description="List of thoughts")
    current_step: int = Field(default=0, description="Current step")
    max_steps: int = Field(default=10, description="Maximum number of steps")
    completed: bool = Field(default=False, description="Whether session is completed")


class SessionManager:
    """Manager for thinking sessions."""
    
    def __init__(self):
        self.sessions: Dict[str, ThinkingSession] = {}
    
    def create_session(self, session_id: str, max_steps: int = 10) -> ThinkingSession:
        """Create a new thinking session."""
        session = ThinkingSession(
            session_id=session_id,
            max_steps=max_steps
        )
        self.sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[ThinkingSession]:
        """Get a thinking session by ID."""
        return self.sessions.get(session_id)
    
    def update_session(self, session: ThinkingSession) -> None:
        """Update a thinking session."""
        self.sessions[session.session_id] = session
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a thinking session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def list_sessions(self) -> List[str]:
        """List all session IDs."""
        return list(self.sessions.keys())


# Global session manager instance
session_manager = SessionManager()


def get_session_manager() -> SessionManager:
    """Get the global session manager."""
    return session_manager