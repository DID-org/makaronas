"""Contract tests for SessionStore — behavioral specification.

Verifies that any SessionStore implementation satisfies:
- CRUD operations for game sessions
- TTL enforcement (expired sessions return None)
- Idempotent deletes

TTL tests use explicit expires_at values (past or future) rather than
clock mocking — this works with any backend (Redis server-side TTL,
database scheduled cleanup, lazy deletion, etc.).

Run against registered implementations:
    python -m pytest backend/tests/contracts/test_session_contract.py -v
"""

from datetime import datetime, timedelta, timezone

import pytest

from backend.schemas import GameSession


class TestSessionContract:
    """Behavioral contract for SessionStore implementations."""

    # -- CRUD basics -------------------------------------------------------

    @pytest.mark.asyncio
    async def test_save_then_get_returns_session(
        self, session_store, sample_session
    ) -> None:
        """Save a session, then get it back by session_id."""
        await session_store.save_session(sample_session)
        result = await session_store.get_session(sample_session.session_id)
        assert result is not None
        assert result.session_id == sample_session.session_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self, session_store) -> None:
        """Getting a session that doesn't exist must return None."""
        result = await session_store.get_session("nonexistent-session")
        assert result is None

    @pytest.mark.asyncio
    async def test_save_overwrites_existing(self, session_store) -> None:
        """Second save with same session_id overwrites the first."""
        session_v1 = GameSession(
            session_id="overwrite-test",
            student_id="s1",
            school_id="school-a",
            current_task=None,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
        )
        session_v2 = GameSession(
            session_id="overwrite-test",
            student_id="s1",
            school_id="school-a",
            current_task="task-5",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
        )
        await session_store.save_session(session_v1)
        await session_store.save_session(session_v2)
        result = await session_store.get_session("overwrite-test")
        assert result is not None
        assert result.current_task == "task-5"

    @pytest.mark.asyncio
    async def test_delete_then_get_returns_none(
        self, session_store, sample_session
    ) -> None:
        """Deleted session must not be retrievable."""
        await session_store.save_session(sample_session)
        await session_store.delete_session(sample_session.session_id)
        result = await session_store.get_session(sample_session.session_id)
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_is_idempotent(self, session_store) -> None:
        """Deleting a non-existent session must not raise an error."""
        await session_store.delete_session("nonexistent-session")

    # -- TTL enforcement (the core session contract) -----------------------

    @pytest.mark.asyncio
    async def test_expired_session_returns_none(self, session_store) -> None:
        """Session with expires_at in the past must return None from get."""
        expired = GameSession(
            session_id="expired-sess",
            student_id="s1",
            school_id="school-a",
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        await session_store.save_session(expired)
        result = await session_store.get_session("expired-sess")
        assert result is None

    @pytest.mark.asyncio
    async def test_future_session_is_retrievable(self, session_store) -> None:
        """Session with expires_at in the future must be retrievable."""
        future = GameSession(
            session_id="future-sess",
            student_id="s1",
            school_id="school-a",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        await session_store.save_session(future)
        result = await session_store.get_session("future-sess")
        assert result is not None
        assert result.session_id == "future-sess"

    # -- Data integrity ----------------------------------------------------

    @pytest.mark.asyncio
    async def test_save_preserves_session_data(
        self, session_store, sample_session
    ) -> None:
        """Saved session must retain all field values."""
        await session_store.save_session(sample_session)
        result = await session_store.get_session(sample_session.session_id)
        assert result is not None
        assert result.student_id == sample_session.student_id
        assert result.school_id == sample_session.school_id
        assert result.language == sample_session.language
        assert result.current_task == sample_session.current_task
