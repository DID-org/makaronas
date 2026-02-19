"""Tests for backend.hooks.interfaces — ABC contract enforcement.

Verifies that the abstract base classes enforce their contracts:
- Direct instantiation raises TypeError
- Incomplete subclasses raise TypeError at instantiation
- Complete subclasses with all methods implemented instantiate successfully

These tests verify Python's ABC mechanics, not async behavior.
Behavioral contract tests live in Phase 5b (backend/tests/contracts/).
"""

import pytest

from backend.hooks.interfaces import (
    AuthService,
    DatabaseAdapter,
    FileStorage,
    RateLimiter,
    SessionStore,
)


# ---------------------------------------------------------------------------
# AuthService
# ---------------------------------------------------------------------------


class TestAuthService:
    """AuthService ABC — validate_token and get_user."""

    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            AuthService()  # type: ignore[abstract]

    def test_incomplete_subclass_missing_get_user(self) -> None:
        class Partial(AuthService):
            async def validate_token(self, token):
                return None

        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]

    def test_incomplete_subclass_missing_validate_token(self) -> None:
        class Partial(AuthService):
            async def get_user(self, user_id):
                return None

        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]

    def test_complete_subclass_instantiates(self) -> None:
        class Complete(AuthService):
            async def validate_token(self, token):
                return None

            async def get_user(self, user_id):
                return None

        instance = Complete()
        assert isinstance(instance, AuthService)


# ---------------------------------------------------------------------------
# DatabaseAdapter
# ---------------------------------------------------------------------------


class TestDatabaseAdapter:
    """DatabaseAdapter ABC — five abstract methods for persistent storage."""

    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            DatabaseAdapter()  # type: ignore[abstract]

    def test_incomplete_subclass_missing_one_method(self) -> None:
        class Partial(DatabaseAdapter):
            async def get_student_profile(self, student_id, school_id):
                return None

            async def save_student_profile(self, profile):
                pass

            async def delete_student_profile(self, student_id, school_id):
                pass

            async def export_student_data(self, student_id, school_id):
                return {}

            # Missing: get_class_insights

        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]

    def test_complete_subclass_instantiates(self) -> None:
        class Complete(DatabaseAdapter):
            async def get_student_profile(self, student_id, school_id):
                return None

            async def save_student_profile(self, profile):
                pass

            async def delete_student_profile(self, student_id, school_id):
                pass

            async def export_student_data(self, student_id, school_id):
                return {}

            async def get_class_insights(self, class_id, school_id):
                return None

        instance = Complete()
        assert isinstance(instance, DatabaseAdapter)


# ---------------------------------------------------------------------------
# SessionStore
# ---------------------------------------------------------------------------


class TestSessionStore:
    """SessionStore ABC — get, save, delete session."""

    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            SessionStore()  # type: ignore[abstract]

    def test_incomplete_subclass_missing_delete(self) -> None:
        class Partial(SessionStore):
            async def get_session(self, session_id):
                return None

            async def save_session(self, session):
                pass

            # Missing: delete_session

        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]

    def test_complete_subclass_instantiates(self) -> None:
        class Complete(SessionStore):
            async def get_session(self, session_id):
                return None

            async def save_session(self, session):
                pass

            async def delete_session(self, session_id):
                pass

        instance = Complete()
        assert isinstance(instance, SessionStore)


# ---------------------------------------------------------------------------
# FileStorage
# ---------------------------------------------------------------------------


class TestFileStorage:
    """FileStorage ABC — get_asset_url and store_asset."""

    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            FileStorage()  # type: ignore[abstract]

    def test_incomplete_subclass_missing_store(self) -> None:
        class Partial(FileStorage):
            async def get_asset_url(self, task_id, filename):
                return ""

            # Missing: store_asset

        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]

    def test_complete_subclass_instantiates(self) -> None:
        class Complete(FileStorage):
            async def get_asset_url(self, task_id, filename):
                return ""

            async def store_asset(self, task_id, filename, data):
                return ""

        instance = Complete()
        assert isinstance(instance, FileStorage)


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------


class TestRateLimiter:
    """RateLimiter ABC — stub contract for team implementation."""

    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            RateLimiter()  # type: ignore[abstract]

    def test_incomplete_subclass_missing_record(self) -> None:
        class Partial(RateLimiter):
            async def check_rate_limit(self, user_id, action):
                return True

            # Missing: record_action

        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]

    def test_complete_subclass_instantiates(self) -> None:
        class Complete(RateLimiter):
            async def check_rate_limit(self, user_id, action):
                return True

            async def record_action(self, user_id, action):
                pass

        instance = Complete()
        assert isinstance(instance, RateLimiter)


# ---------------------------------------------------------------------------
# Import verification
# ---------------------------------------------------------------------------


class TestImports:
    """Verifies all five ABCs are importable from the interfaces module."""

    def test_all_abcs_importable(self) -> None:
        from backend.hooks.interfaces import (
            AuthService,
            DatabaseAdapter,
            FileStorage,
            RateLimiter,
            SessionStore,
        )

        assert AuthService is not None
        assert DatabaseAdapter is not None
        assert SessionStore is not None
        assert FileStorage is not None
        assert RateLimiter is not None
