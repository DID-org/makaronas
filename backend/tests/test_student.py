"""Tests for Phase 4a — Student-facing API endpoints.

Covers: session creation, session lookup (404), ownership enforcement (403),
SSE streaming (respond + debrief), radar profile (student + teacher access),
GDPR deletion + export, auth enforcement on all endpoints.

Uses httpx.AsyncClient with ASGITransport (async test client). All tests use
explicit @pytest.mark.asyncio per strict mode (Python 3.13.5, Phase 1a note).
"""

import json

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

from backend.api import deps
from backend.api.deps import get_current_user
from backend.main import app
from backend.schemas import GameSession, StudentProfile, User

AUTH_HEADER = {"Authorization": "Bearer test-token-123"}

# The user returned by FakeAuthService for any valid token
FAKE_USER_ID = "fake-user-1"
FAKE_SCHOOL_ID = "school-test-1"

TEACHER_USER = User(
    id="teacher-1", role="teacher", name="Test Teacher", school_id=FAKE_SCHOOL_ID
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> httpx.AsyncClient:
    """Async test client wired to the app."""
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest_asyncio.fixture
async def session_id() -> str:
    """Creates a session in the store and returns its ID.

    Uses the fake auth user's ID so ownership checks pass.
    """
    session = GameSession(
        session_id="test-session-aaa",
        student_id=FAKE_USER_ID,
        school_id=FAKE_SCHOOL_ID,
    )
    await deps._session_store.save_session(session)
    return session.session_id


@pytest_asyncio.fixture
async def other_session_id() -> str:
    """Creates a session owned by a different student."""
    session = GameSession(
        session_id="test-session-other",
        student_id="other-user",
        school_id=FAKE_SCHOOL_ID,
    )
    await deps._session_store.save_session(session)
    return session.session_id


@pytest.fixture(autouse=True)
def _cleanup_overrides():
    """Ensures dependency overrides are cleaned up after each test."""
    yield
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_sse_events(body: str) -> list[dict]:
    """Parses raw SSE body into a list of {type, data} dicts."""
    events = []
    for block in body.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event_type = None
        data_json = None
        for line in block.split("\n"):
            if line.startswith("event: "):
                event_type = line[7:]
            elif line.startswith("data: "):
                data_json = line[6:]
        if event_type and data_json:
            events.append({"type": event_type, "data": json.loads(data_json)})
    return events


# ---------------------------------------------------------------------------
# POST /session
# ---------------------------------------------------------------------------


class TestCreateSession:
    """POST /api/v1/student/session — creates a new game session."""

    @pytest.mark.asyncio
    async def test_creates_session_returns_200(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.post(
                "/api/v1/student/session",
                json={},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert "session_id" in body["data"]
        assert body["data"]["language"] == "lt"

    @pytest.mark.asyncio
    async def test_custom_language(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.post(
                "/api/v1/student/session",
                json={"language": "en"},
                headers=AUTH_HEADER,
            )
        body = resp.json()
        assert body["data"]["language"] == "en"

    @pytest.mark.asyncio
    async def test_custom_roadmap_id(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.post(
                "/api/v1/student/session",
                json={"roadmap_id": "intro-sequence"},
                headers=AUTH_HEADER,
            )
        body = resp.json()
        assert body["ok"] is True

    @pytest.mark.asyncio
    async def test_session_persisted_in_store(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.post(
                "/api/v1/student/session",
                json={},
                headers=AUTH_HEADER,
            )
        sid = resp.json()["data"]["session_id"]
        session = await deps._session_store.get_session(sid)
        assert session is not None
        assert session.student_id == FAKE_USER_ID
        assert session.school_id == FAKE_SCHOOL_ID

    @pytest.mark.asyncio
    async def test_no_auth_returns_401(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.post("/api/v1/student/session", json={})
        assert resp.status_code == 401
        body = resp.json()
        assert body["ok"] is False
        assert body["error"]["code"] == "UNAUTHORIZED"


# ---------------------------------------------------------------------------
# GET /session/{session_id}/next
# ---------------------------------------------------------------------------


class TestNextTask:
    """GET /api/v1/student/session/{id}/next — stub task content."""

    @pytest.mark.asyncio
    async def test_returns_stub_task(
        self, client: httpx.AsyncClient, session_id: str
    ) -> None:
        async with client:
            resp = await client.get(
                f"/api/v1/student/session/{session_id}/next",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert "task_id" in body["data"]
        assert "content" in body["data"]
        assert "available_actions" in body["data"]

    @pytest.mark.asyncio
    async def test_nonexistent_session_returns_404(
        self, client: httpx.AsyncClient
    ) -> None:
        async with client:
            resp = await client.get(
                "/api/v1/student/session/nonexistent-id/next",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 404
        body = resp.json()
        assert body["ok"] is False
        assert body["error"]["code"] == "SESSION_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_other_students_session_returns_403(
        self, client: httpx.AsyncClient, other_session_id: str
    ) -> None:
        async with client:
            resp = await client.get(
                f"/api/v1/student/session/{other_session_id}/next",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 403
        body = resp.json()
        assert body["ok"] is False
        assert body["error"]["code"] == "FORBIDDEN"

    @pytest.mark.asyncio
    async def test_no_auth_returns_401(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.get("/api/v1/student/session/any-id/next")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# POST /session/{session_id}/respond
# ---------------------------------------------------------------------------


class TestRespond:
    """POST /api/v1/student/session/{id}/respond — SSE stream."""

    @pytest.mark.asyncio
    async def test_returns_sse_stream(
        self, client: httpx.AsyncClient, session_id: str
    ) -> None:
        async with client:
            resp = await client.post(
                f"/api/v1/student/session/{session_id}/respond",
                json={"action": "freeform", "payload": "I think this is fake"},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        events = _parse_sse_events(resp.text)
        token_events = [e for e in events if e["type"] == "token"]
        done_events = [e for e in events if e["type"] == "done"]

        assert len(token_events) >= 1
        assert len(done_events) == 1
        assert done_events[0]["data"]["data"]["action_received"] == "freeform"

    @pytest.mark.asyncio
    async def test_nonexistent_session_returns_404(
        self, client: httpx.AsyncClient
    ) -> None:
        async with client:
            resp = await client.post(
                "/api/v1/student/session/bad-id/respond",
                json={"action": "freeform", "payload": "hello"},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_invalid_action_returns_422(
        self, client: httpx.AsyncClient, session_id: str
    ) -> None:
        async with client:
            resp = await client.post(
                f"/api/v1/student/session/{session_id}/respond",
                json={"action": "invalid_action", "payload": "test"},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_missing_payload_returns_422(
        self, client: httpx.AsyncClient, session_id: str
    ) -> None:
        async with client:
            resp = await client.post(
                f"/api/v1/student/session/{session_id}/respond",
                json={"action": "freeform"},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_other_students_session_returns_403(
        self, client: httpx.AsyncClient, other_session_id: str
    ) -> None:
        async with client:
            resp = await client.post(
                f"/api/v1/student/session/{other_session_id}/respond",
                json={"action": "freeform", "payload": "test"},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_no_auth_returns_401(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.post(
                "/api/v1/student/session/any-id/respond",
                json={"action": "freeform", "payload": "test"},
            )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# GET /session/{session_id}/debrief
# ---------------------------------------------------------------------------


class TestDebrief:
    """GET /api/v1/student/session/{id}/debrief — SSE stream."""

    @pytest.mark.asyncio
    async def test_returns_sse_stream(
        self, client: httpx.AsyncClient, session_id: str
    ) -> None:
        async with client:
            resp = await client.get(
                f"/api/v1/student/session/{session_id}/debrief",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        events = _parse_sse_events(resp.text)
        token_events = [e for e in events if e["type"] == "token"]
        done_events = [e for e in events if e["type"] == "done"]

        assert len(token_events) >= 1
        assert len(done_events) == 1
        assert "You did well today. " == done_events[0]["data"]["full_text"]

    @pytest.mark.asyncio
    async def test_nonexistent_session_returns_404(
        self, client: httpx.AsyncClient
    ) -> None:
        async with client:
            resp = await client.get(
                "/api/v1/student/session/bad-id/debrief",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_other_students_session_returns_403(
        self, client: httpx.AsyncClient, other_session_id: str
    ) -> None:
        async with client:
            resp = await client.get(
                f"/api/v1/student/session/{other_session_id}/debrief",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_no_auth_returns_401(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.get("/api/v1/student/session/any-id/debrief")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# GET /profile/{student_id}/radar
# ---------------------------------------------------------------------------


class TestRadarProfile:
    """GET /api/v1/student/profile/{id}/radar — radar profile data."""

    @pytest.mark.asyncio
    async def test_student_own_profile(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.get(
                f"/api/v1/student/profile/{FAKE_USER_ID}/radar",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["data"]["student_id"] == FAKE_USER_ID

    @pytest.mark.asyncio
    async def test_student_other_profile_returns_403(
        self, client: httpx.AsyncClient
    ) -> None:
        async with client:
            resp = await client.get(
                "/api/v1/student/profile/someone-else/radar",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 403
        body = resp.json()
        assert body["error"]["code"] == "FORBIDDEN"

    @pytest.mark.asyncio
    async def test_teacher_can_view_any_student(
        self, client: httpx.AsyncClient
    ) -> None:
        app.dependency_overrides[get_current_user] = lambda: TEACHER_USER

        async with client:
            resp = await client.get(
                "/api/v1/student/profile/any-student-id/radar",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True

    @pytest.mark.asyncio
    async def test_returns_real_profile_when_exists(
        self, client: httpx.AsyncClient
    ) -> None:
        profile = StudentProfile(
            student_id=FAKE_USER_ID,
            school_id=FAKE_SCHOOL_ID,
            sessions_completed=3,
        )
        await deps._database.save_student_profile(profile)

        async with client:
            resp = await client.get(
                f"/api/v1/student/profile/{FAKE_USER_ID}/radar",
                headers=AUTH_HEADER,
            )
        body = resp.json()
        assert body["data"]["sessions_completed"] == 3

        # Cleanup
        await deps._database.delete_student_profile(FAKE_USER_ID, FAKE_SCHOOL_ID)

    @pytest.mark.asyncio
    async def test_no_auth_returns_401(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.get(
                f"/api/v1/student/profile/{FAKE_USER_ID}/radar"
            )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# DELETE /profile/{student_id}
# ---------------------------------------------------------------------------


class TestDeleteProfile:
    """DELETE /api/v1/student/profile/{id} — GDPR deletion."""

    @pytest.mark.asyncio
    async def test_student_deletes_own_profile(
        self, client: httpx.AsyncClient
    ) -> None:
        # Seed a profile
        profile = StudentProfile(
            student_id=FAKE_USER_ID,
            school_id=FAKE_SCHOOL_ID,
            sessions_completed=5,
        )
        await deps._database.save_student_profile(profile)

        async with client:
            resp = await client.delete(
                f"/api/v1/student/profile/{FAKE_USER_ID}",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["data"]["deleted"] is True

        # Verify actually deleted
        result = await deps._database.get_student_profile(
            FAKE_USER_ID, FAKE_SCHOOL_ID
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_student_cannot_delete_other_profile(
        self, client: httpx.AsyncClient
    ) -> None:
        async with client:
            resp = await client.delete(
                "/api/v1/student/profile/someone-else",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 403
        body = resp.json()
        assert body["error"]["code"] == "FORBIDDEN"

    @pytest.mark.asyncio
    async def test_teacher_can_delete_student_profile(
        self, client: httpx.AsyncClient
    ) -> None:
        app.dependency_overrides[get_current_user] = lambda: TEACHER_USER

        # Seed a profile
        profile = StudentProfile(
            student_id="student-to-delete",
            school_id=FAKE_SCHOOL_ID,
        )
        await deps._database.save_student_profile(profile)

        async with client:
            resp = await client.delete(
                "/api/v1/student/profile/student-to-delete",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True

    @pytest.mark.asyncio
    async def test_delete_nonexistent_profile_still_200(
        self, client: httpx.AsyncClient
    ) -> None:
        """Deletion is idempotent — deleting nothing is fine."""
        async with client:
            resp = await client.delete(
                f"/api/v1/student/profile/{FAKE_USER_ID}",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_no_auth_returns_401(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.delete(
                f"/api/v1/student/profile/{FAKE_USER_ID}"
            )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# GET /profile/{student_id}/export
# ---------------------------------------------------------------------------


class TestExportProfile:
    """GET /api/v1/student/profile/{id}/export — GDPR data export."""

    @pytest.mark.asyncio
    async def test_export_own_data(self, client: httpx.AsyncClient) -> None:
        # Seed a profile
        profile = StudentProfile(
            student_id=FAKE_USER_ID,
            school_id=FAKE_SCHOOL_ID,
            sessions_completed=7,
        )
        await deps._database.save_student_profile(profile)

        async with client:
            resp = await client.get(
                f"/api/v1/student/profile/{FAKE_USER_ID}/export",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert "profile" in body["data"]

        # Cleanup
        await deps._database.delete_student_profile(FAKE_USER_ID, FAKE_SCHOOL_ID)

    @pytest.mark.asyncio
    async def test_export_empty_returns_empty_dict(
        self, client: httpx.AsyncClient
    ) -> None:
        async with client:
            resp = await client.get(
                f"/api/v1/student/profile/{FAKE_USER_ID}/export",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["data"] == {}

    @pytest.mark.asyncio
    async def test_student_cannot_export_other_data(
        self, client: httpx.AsyncClient
    ) -> None:
        async with client:
            resp = await client.get(
                "/api/v1/student/profile/someone-else/export",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_teacher_can_export_student_data(
        self, client: httpx.AsyncClient
    ) -> None:
        app.dependency_overrides[get_current_user] = lambda: TEACHER_USER

        async with client:
            resp = await client.get(
                "/api/v1/student/profile/any-student/export",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True

    @pytest.mark.asyncio
    async def test_no_auth_returns_401(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.get(
                f"/api/v1/student/profile/{FAKE_USER_ID}/export"
            )
        assert resp.status_code == 401
