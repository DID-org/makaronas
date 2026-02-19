"""In-memory session store — development stub for SessionStore.

Python dict-backed storage for ephemeral game sessions. TTL is enforced
on read: get_session checks expires_at and lazily deletes expired entries.
No background sweeper — a stub that loses data on restart doesn't need one.

TEAM: Replace this with your real session store (Redis, etc.). Subclass
SessionStore from backend.hooks.interfaces and implement all three
abstract methods. Your implementation MUST enforce TTL — callers never
check expires_at themselves.

Tier 2 service module: imports from backend.hooks.interfaces (Tier 1)
and backend.schemas (Tier 1).

Usage:
    from backend.hooks.sessions import InMemorySessionStore

    sessions = InMemorySessionStore()
    await sessions.save_session(game_session)
    await sessions.get_session("session-id")  # None if expired
"""

from datetime import datetime, timezone

from backend.hooks.interfaces import SessionStore
from backend.schemas import GameSession


class InMemorySessionStore(SessionStore):
    """STUB — dict-backed session storage, loses data on restart.

    Sessions are keyed by session_id. Expired sessions are lazily
    deleted on read — get_session checks expires_at against UTC now
    and removes stale entries.

    TEAM: Replace with your session store. Satisfy the SessionStore
    interface from backend.hooks.interfaces. TTL enforcement is your
    responsibility — callers trust get_session to return None for
    expired sessions.
    """

    def __init__(self) -> None:
        """Initialises empty session store."""
        self._sessions: dict[str, GameSession] = {}

    async def get_session(self, session_id: str) -> GameSession | None:
        """Retrieves a session, returning None if expired or missing.

        Enforces TTL: if the session's expires_at is in the past, the
        entry is deleted and None is returned.

        Args:
            session_id: The session identifier.

        Returns:
            The GameSession if it exists and hasn't expired, None otherwise.
        """
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if session.expires_at <= datetime.now(timezone.utc):
            del self._sessions[session_id]
            return None
        return session

    async def save_session(self, session: GameSession) -> None:
        """Stores a session, keyed by session_id. Creates or overwrites.

        Args:
            session: The GameSession to store.
        """
        self._sessions[session.session_id] = session

    async def delete_session(self, session_id: str) -> None:
        """Deletes a session. No-op if not found (idempotent).

        Args:
            session_id: The session identifier.
        """
        self._sessions.pop(session_id, None)
