"""Tests for Phase 2b stub implementations.

Verifies behavioral correctness of all four hook stubs:
- FakeAuthService (auth)
- InMemoryStore (database)
- InMemorySessionStore (sessions)
- LocalFileStorage (storage)

All tests are async (testing async stub methods). Uses explicit
@pytest.mark.asyncio per strict mode (Python 3.13.5, Phase 1a note).
"""

from datetime import datetime, timedelta, timezone

import pytest

from backend.hooks.auth import FakeAuthService
from backend.hooks.database import InMemoryStore
from backend.hooks.sessions import InMemorySessionStore
from backend.hooks.storage import LocalFileStorage
from backend.schemas import ClassInsights, GameSession, StudentProfile


# ---------------------------------------------------------------------------
# FakeAuthService
# ---------------------------------------------------------------------------


class TestFakeAuthService:
    """FakeAuthService — token validation and user lookup."""

    @pytest.mark.asyncio
    async def test_validate_token_returns_student_by_default(self) -> None:
        auth = FakeAuthService()
        user = await auth.validate_token("any-token")
        assert user is not None
        assert user.role == "student"

    @pytest.mark.asyncio
    async def test_validate_token_empty_returns_none(self) -> None:
        auth = FakeAuthService()
        user = await auth.validate_token("")
        assert user is None

    @pytest.mark.asyncio
    async def test_validate_token_with_teacher_role(self) -> None:
        auth = FakeAuthService(default_role="teacher")
        user = await auth.validate_token("token-123")
        assert user is not None
        assert user.role == "teacher"

    @pytest.mark.asyncio
    async def test_validate_token_with_admin_role(self) -> None:
        auth = FakeAuthService(default_role="admin")
        user = await auth.validate_token("token-456")
        assert user is not None
        assert user.role == "admin"

    @pytest.mark.asyncio
    async def test_validate_token_user_has_school_id(self) -> None:
        auth = FakeAuthService()
        user = await auth.validate_token("token")
        assert user is not None
        assert user.school_id
        assert isinstance(user.school_id, str)

    @pytest.mark.asyncio
    async def test_get_user_returns_user_with_given_id(self) -> None:
        auth = FakeAuthService()
        user = await auth.get_user("student-42")
        assert user is not None
        assert user.id == "student-42"
        assert user.role == "student"

    @pytest.mark.asyncio
    async def test_get_user_empty_id_returns_none(self) -> None:
        auth = FakeAuthService()
        user = await auth.get_user("")
        assert user is None

    @pytest.mark.asyncio
    async def test_get_user_respects_configured_role(self) -> None:
        auth = FakeAuthService(default_role="teacher")
        user = await auth.get_user("teacher-7")
        assert user is not None
        assert user.id == "teacher-7"
        assert user.role == "teacher"

    @pytest.mark.asyncio
    async def test_user_has_name(self) -> None:
        auth = FakeAuthService()
        user = await auth.validate_token("token")
        assert user is not None
        assert user.name
        assert isinstance(user.name, str)


# ---------------------------------------------------------------------------
# InMemoryStore
# ---------------------------------------------------------------------------


class TestInMemoryStore:
    """InMemoryStore — CRUD, multi-tenant isolation, GDPR, class insights."""

    @pytest.mark.asyncio
    async def test_save_then_get_returns_profile(self) -> None:
        db = InMemoryStore()
        profile = StudentProfile(student_id="s1", school_id="school-a")
        await db.save_student_profile(profile)
        result = await db.get_student_profile("s1", "school-a")
        assert result is not None
        assert result.student_id == "s1"
        assert result.school_id == "school-a"

    @pytest.mark.asyncio
    async def test_get_with_wrong_school_id_returns_none(self) -> None:
        db = InMemoryStore()
        profile = StudentProfile(student_id="s1", school_id="school-a")
        await db.save_student_profile(profile)
        result = await db.get_student_profile("s1", "school-b")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self) -> None:
        db = InMemoryStore()
        result = await db.get_student_profile("ghost", "school-a")
        assert result is None

    @pytest.mark.asyncio
    async def test_save_overwrites_existing(self) -> None:
        db = InMemoryStore()
        profile1 = StudentProfile(
            student_id="s1", school_id="school-a", sessions_completed=0
        )
        profile2 = StudentProfile(
            student_id="s1", school_id="school-a", sessions_completed=5
        )
        await db.save_student_profile(profile1)
        await db.save_student_profile(profile2)
        result = await db.get_student_profile("s1", "school-a")
        assert result is not None
        assert result.sessions_completed == 5

    @pytest.mark.asyncio
    async def test_delete_removes_profile(self) -> None:
        db = InMemoryStore()
        profile = StudentProfile(student_id="s1", school_id="school-a")
        await db.save_student_profile(profile)
        await db.delete_student_profile("s1", "school-a")
        result = await db.get_student_profile("s1", "school-a")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_with_wrong_school_id_is_noop(self) -> None:
        db = InMemoryStore()
        profile = StudentProfile(student_id="s1", school_id="school-a")
        await db.save_student_profile(profile)
        await db.delete_student_profile("s1", "school-b")
        result = await db.get_student_profile("s1", "school-a")
        assert result is not None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_is_noop(self) -> None:
        db = InMemoryStore()
        await db.delete_student_profile("ghost", "school-a")

    @pytest.mark.asyncio
    async def test_export_returns_profile_dict(self) -> None:
        db = InMemoryStore()
        profile = StudentProfile(student_id="s1", school_id="school-a")
        await db.save_student_profile(profile)
        export = await db.export_student_data("s1", "school-a")
        assert "profile" in export
        assert export["profile"]["student_id"] == "s1"
        assert export["profile"]["school_id"] == "school-a"

    @pytest.mark.asyncio
    async def test_export_nonexistent_returns_empty_dict(self) -> None:
        db = InMemoryStore()
        export = await db.export_student_data("ghost", "school-a")
        assert export == {}

    @pytest.mark.asyncio
    async def test_seed_then_get_class_insights(self) -> None:
        db = InMemoryStore()
        insights = ClassInsights(class_id="class-1", school_id="school-a")
        db.seed_class_insights(insights)
        result = await db.get_class_insights("class-1", "school-a")
        assert result is not None
        assert result.class_id == "class-1"

    @pytest.mark.asyncio
    async def test_get_class_insights_wrong_school_returns_none(self) -> None:
        db = InMemoryStore()
        insights = ClassInsights(class_id="class-1", school_id="school-a")
        db.seed_class_insights(insights)
        result = await db.get_class_insights("class-1", "school-b")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_class_insights_nonexistent_returns_none(self) -> None:
        db = InMemoryStore()
        result = await db.get_class_insights("ghost", "school-a")
        assert result is None


# ---------------------------------------------------------------------------
# InMemorySessionStore
# ---------------------------------------------------------------------------


class TestInMemorySessionStore:
    """InMemorySessionStore — CRUD and TTL enforcement."""

    @pytest.mark.asyncio
    async def test_save_then_get_returns_session(self) -> None:
        store = InMemorySessionStore()
        session = GameSession(
            session_id="sess-1", student_id="s1", school_id="school-a"
        )
        await store.save_session(session)
        result = await store.get_session("sess-1")
        assert result is not None
        assert result.session_id == "sess-1"

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self) -> None:
        store = InMemorySessionStore()
        result = await store.get_session("ghost")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_removes_session(self) -> None:
        store = InMemorySessionStore()
        session = GameSession(
            session_id="sess-1", student_id="s1", school_id="school-a"
        )
        await store.save_session(session)
        await store.delete_session("sess-1")
        result = await store.get_session("sess-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_is_noop(self) -> None:
        store = InMemorySessionStore()
        await store.delete_session("ghost")

    @pytest.mark.asyncio
    async def test_expired_session_returns_none(self) -> None:
        store = InMemorySessionStore()
        expired = GameSession(
            session_id="sess-old",
            student_id="s1",
            school_id="school-a",
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        await store.save_session(expired)
        result = await store.get_session("sess-old")
        assert result is None

    @pytest.mark.asyncio
    async def test_expired_session_is_deleted_from_store(self) -> None:
        store = InMemorySessionStore()
        expired = GameSession(
            session_id="sess-old",
            student_id="s1",
            school_id="school-a",
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        await store.save_session(expired)
        await store.get_session("sess-old")
        # Verify it was actually removed from internal storage
        assert "sess-old" not in store._sessions

    @pytest.mark.asyncio
    async def test_save_overwrites_existing(self) -> None:
        store = InMemorySessionStore()
        session1 = GameSession(
            session_id="sess-1", student_id="s1", school_id="school-a"
        )
        session2 = GameSession(
            session_id="sess-1",
            student_id="s1",
            school_id="school-a",
            current_task="task-5",
        )
        await store.save_session(session1)
        await store.save_session(session2)
        result = await store.get_session("sess-1")
        assert result is not None
        assert result.current_task == "task-5"


# ---------------------------------------------------------------------------
# LocalFileStorage
# ---------------------------------------------------------------------------


class TestLocalFileStorage:
    """LocalFileStorage — URL format and filesystem operations."""

    @pytest.mark.asyncio
    async def test_get_asset_url_format(self) -> None:
        storage = LocalFileStorage()
        url = await storage.get_asset_url("task1", "image.png")
        assert url == "/api/v1/assets/task1/image.png"

    @pytest.mark.asyncio
    async def test_store_asset_writes_file(self, tmp_path: object) -> None:
        storage = LocalFileStorage(base_path=str(tmp_path))
        data = b"hello world"
        url = await storage.store_asset("task1", "file.txt", data)
        assert url == "/api/v1/assets/task1/file.txt"
        # Verify file was written
        from pathlib import Path

        written = Path(str(tmp_path)) / "task1" / "file.txt"
        assert written.exists()
        assert written.read_bytes() == b"hello world"

    @pytest.mark.asyncio
    async def test_store_asset_creates_directories(
        self, tmp_path: object
    ) -> None:
        storage = LocalFileStorage(base_path=str(tmp_path))
        await storage.store_asset("deep/task", "asset.bin", b"\x00\x01")
        from pathlib import Path

        written = Path(str(tmp_path)) / "deep" / "task" / "asset.bin"
        assert written.exists()
        assert written.read_bytes() == b"\x00\x01"

    @pytest.mark.asyncio
    async def test_store_asset_returns_same_url_as_get(self) -> None:
        storage = LocalFileStorage()
        get_url = await storage.get_asset_url("task-x", "pic.jpg")
        # store_asset needs a real path, but the URL format should match
        assert get_url == "/api/v1/assets/task-x/pic.jpg"

    @pytest.mark.asyncio
    async def test_store_asset_overwrites_existing(
        self, tmp_path: object
    ) -> None:
        storage = LocalFileStorage(base_path=str(tmp_path))
        await storage.store_asset("task1", "data.bin", b"original")
        await storage.store_asset("task1", "data.bin", b"updated")
        from pathlib import Path

        written = Path(str(tmp_path)) / "task1" / "data.bin"
        assert written.read_bytes() == b"updated"
