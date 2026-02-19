"""Tests for Phase 4c — Composer AI and asset serving endpoints.

Covers: composer chat (SSE), roadmap generate, roadmap refine, asset serving,
auth enforcement (401), role enforcement (403 for students on composer,
allowed for students on assets), path traversal rejection (400), missing
asset (404).

Uses httpx.AsyncClient with ASGITransport (async test client). All tests use
explicit @pytest.mark.asyncio per strict mode (Python 3.13.5, Phase 1a note).
"""

import json

import httpx
import pytest
from httpx import ASGITransport

from backend.api.deps import get_current_user, get_file_storage
from backend.hooks.storage import LocalFileStorage
from backend.main import app
from backend.schemas import User

AUTH_HEADER = {"Authorization": "Bearer test-token-123"}

FAKE_SCHOOL_ID = "school-test-1"

TEACHER_USER = User(
    id="teacher-1", role="teacher", name="Test Teacher", school_id=FAKE_SCHOOL_ID
)

ADMIN_USER = User(
    id="admin-1", role="admin", name="Test Admin", school_id=FAKE_SCHOOL_ID
)

STUDENT_USER = User(
    id="student-1", role="student", name="Test Student", school_id=FAKE_SCHOOL_ID
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> httpx.AsyncClient:
    """Async test client wired to the app."""
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture(autouse=True)
def _cleanup_overrides():
    """Ensures dependency overrides are cleaned up after each test."""
    yield
    app.dependency_overrides.clear()


def _use_teacher() -> None:
    """Injects teacher user for the current test."""
    app.dependency_overrides[get_current_user] = lambda: TEACHER_USER


def _use_admin() -> None:
    """Injects admin user for the current test."""
    app.dependency_overrides[get_current_user] = lambda: ADMIN_USER


def _use_student() -> None:
    """Injects student user for the current test."""
    app.dependency_overrides[get_current_user] = lambda: STUDENT_USER


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
# POST /composer/chat
# ---------------------------------------------------------------------------


class TestComposerChat:
    """POST /api/v1/composer/chat — SSE streaming composer reply."""

    @pytest.mark.asyncio
    async def test_returns_sse_stream(self, client: httpx.AsyncClient) -> None:
        _use_teacher()
        async with client:
            resp = await client.post(
                "/api/v1/composer/chat",
                json={"message": "I need a 30-minute session on urgency"},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        events = _parse_sse_events(resp.text)
        token_events = [e for e in events if e["type"] == "token"]
        done_events = [e for e in events if e["type"] == "done"]

        assert len(token_events) >= 1
        assert len(done_events) == 1
        assert "conversation_id" in done_events[0]["data"]["data"]

    @pytest.mark.asyncio
    async def test_generates_conversation_id_when_none(self, client: httpx.AsyncClient) -> None:
        _use_teacher()
        async with client:
            resp = await client.post(
                "/api/v1/composer/chat",
                json={"message": "Hello"},
                headers=AUTH_HEADER,
            )
        events = _parse_sse_events(resp.text)
        done_events = [e for e in events if e["type"] == "done"]
        assert len(done_events) == 1
        conv_id = done_events[0]["data"]["data"]["conversation_id"]
        assert conv_id is not None
        assert len(conv_id) > 0

    @pytest.mark.asyncio
    async def test_echoes_provided_conversation_id(self, client: httpx.AsyncClient) -> None:
        _use_teacher()
        async with client:
            resp = await client.post(
                "/api/v1/composer/chat",
                json={"message": "Continue", "conversation_id": "conv-abc-123"},
                headers=AUTH_HEADER,
            )
        events = _parse_sse_events(resp.text)
        done_events = [e for e in events if e["type"] == "done"]
        assert done_events[0]["data"]["data"]["conversation_id"] == "conv-abc-123"

    @pytest.mark.asyncio
    async def test_admin_can_access(self, client: httpx.AsyncClient) -> None:
        _use_admin()
        async with client:
            resp = await client.post(
                "/api/v1/composer/chat",
                json={"message": "Hello"},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_student_returns_403(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.post(
                "/api/v1/composer/chat",
                json={"message": "Hello"},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 403
        body = resp.json()
        assert body["ok"] is False
        assert body["error"]["code"] == "FORBIDDEN"

    @pytest.mark.asyncio
    async def test_no_auth_returns_401(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.post(
                "/api/v1/composer/chat",
                json={"message": "Hello"},
            )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# POST /composer/roadmap/generate
# ---------------------------------------------------------------------------


class TestGenerateRoadmap:
    """POST /api/v1/composer/roadmap/generate — stub roadmap generation."""

    @pytest.mark.asyncio
    async def test_returns_proposed_roadmap(self, client: httpx.AsyncClient) -> None:
        _use_teacher()
        async with client:
            resp = await client.post(
                "/api/v1/composer/roadmap/generate",
                json={"description": "30 minutes on urgency triggers"},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        data = body["data"]
        assert "roadmap_id" in data
        assert "title" in data
        assert "tasks" in data
        assert isinstance(data["tasks"], list)
        assert len(data["tasks"]) >= 1
        assert "reasoning" in data

    @pytest.mark.asyncio
    async def test_constraints_optional(self, client: httpx.AsyncClient) -> None:
        _use_teacher()
        async with client:
            resp = await client.post(
                "/api/v1/composer/roadmap/generate",
                json={"description": "Quick session"},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    @pytest.mark.asyncio
    async def test_with_constraints(self, client: httpx.AsyncClient) -> None:
        _use_teacher()
        async with client:
            resp = await client.post(
                "/api/v1/composer/roadmap/generate",
                json={
                    "description": "Short session",
                    "constraints": {"max_time_minutes": 20, "difficulty": "beginner"},
                },
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    @pytest.mark.asyncio
    async def test_student_returns_403(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.post(
                "/api/v1/composer/roadmap/generate",
                json={"description": "Should fail"},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 403
        body = resp.json()
        assert body["error"]["code"] == "FORBIDDEN"

    @pytest.mark.asyncio
    async def test_no_auth_returns_401(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.post(
                "/api/v1/composer/roadmap/generate",
                json={"description": "No auth"},
            )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_missing_description_returns_422(self, client: httpx.AsyncClient) -> None:
        _use_teacher()
        async with client:
            resp = await client.post(
                "/api/v1/composer/roadmap/generate",
                json={},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /composer/roadmap/refine
# ---------------------------------------------------------------------------


class TestRefineRoadmap:
    """POST /api/v1/composer/roadmap/refine — stub roadmap refinement."""

    @pytest.mark.asyncio
    async def test_returns_refined_roadmap(self, client: httpx.AsyncClient) -> None:
        _use_teacher()
        async with client:
            resp = await client.post(
                "/api/v1/composer/roadmap/refine",
                json={
                    "roadmap_id": "roadmap-test-001",
                    "instruction": "Make it shorter, 20 minutes max",
                },
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        data = body["data"]
        assert data["roadmap_id"] == "roadmap-test-001"
        assert "title" in data
        assert "tasks" in data
        assert isinstance(data["tasks"], list)
        assert "changes" in data

    @pytest.mark.asyncio
    async def test_student_returns_403(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.post(
                "/api/v1/composer/roadmap/refine",
                json={
                    "roadmap_id": "roadmap-test-001",
                    "instruction": "Should fail",
                },
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 403
        body = resp.json()
        assert body["error"]["code"] == "FORBIDDEN"

    @pytest.mark.asyncio
    async def test_no_auth_returns_401(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.post(
                "/api/v1/composer/roadmap/refine",
                json={
                    "roadmap_id": "roadmap-test-001",
                    "instruction": "No auth",
                },
            )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_missing_fields_returns_422(self, client: httpx.AsyncClient) -> None:
        _use_teacher()
        async with client:
            resp = await client.post(
                "/api/v1/composer/roadmap/refine",
                json={"roadmap_id": "roadmap-test-001"},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /assets/{task_id}/{filename}
# ---------------------------------------------------------------------------


class TestServeAsset:
    """GET /api/v1/assets/{task_id}/{filename} — static file serving."""

    @pytest.mark.asyncio
    async def test_serves_existing_file(self, client: httpx.AsyncClient, tmp_path) -> None:
        _use_student()
        # Create a test asset on disk
        task_dir = tmp_path / "task-img-001"
        task_dir.mkdir()
        asset_file = task_dir / "graph.png"
        asset_file.write_bytes(b"\x89PNG\r\n\x1a\nfake-png-data")

        app.dependency_overrides[get_file_storage] = lambda: LocalFileStorage(
            base_path=str(tmp_path)
        )

        async with client:
            resp = await client.get(
                "/api/v1/assets/task-img-001/graph.png",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        assert b"fake-png-data" in resp.content

    @pytest.mark.asyncio
    async def test_content_type_detection(self, client: httpx.AsyncClient, tmp_path) -> None:
        _use_student()
        task_dir = tmp_path / "task-audio-001"
        task_dir.mkdir()
        asset_file = task_dir / "clip.mp3"
        asset_file.write_bytes(b"fake-mp3-data")

        app.dependency_overrides[get_file_storage] = lambda: LocalFileStorage(
            base_path=str(tmp_path)
        )

        async with client:
            resp = await client.get(
                "/api/v1/assets/task-audio-001/clip.mp3",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        assert "audio" in resp.headers["content-type"]

    @pytest.mark.asyncio
    async def test_teacher_can_access(self, client: httpx.AsyncClient, tmp_path) -> None:
        _use_teacher()
        task_dir = tmp_path / "task-t-001"
        task_dir.mkdir()
        (task_dir / "preview.jpg").write_bytes(b"fake-jpg")

        app.dependency_overrides[get_file_storage] = lambda: LocalFileStorage(
            base_path=str(tmp_path)
        )

        async with client:
            resp = await client.get(
                "/api/v1/assets/task-t-001/preview.jpg",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_missing_file_returns_404(self, client: httpx.AsyncClient, tmp_path) -> None:
        _use_student()
        app.dependency_overrides[get_file_storage] = lambda: LocalFileStorage(
            base_path=str(tmp_path)
        )

        async with client:
            resp = await client.get(
                "/api/v1/assets/nonexistent-task/missing.png",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 404
        body = resp.json()
        assert body["ok"] is False
        assert body["error"]["code"] == "ASSET_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_path_traversal_dotdot_in_task_id_returns_400(self, client: httpx.AsyncClient, tmp_path) -> None:
        _use_student()
        app.dependency_overrides[get_file_storage] = lambda: LocalFileStorage(
            base_path=str(tmp_path)
        )

        async with client:
            resp = await client.get(
                "/api/v1/assets/..secret/passwd",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 400
        body = resp.json()
        assert body["ok"] is False
        assert body["error"]["code"] == "BAD_REQUEST"

    @pytest.mark.asyncio
    async def test_path_traversal_dotdot_in_filename_returns_400(self, client: httpx.AsyncClient, tmp_path) -> None:
        _use_student()
        app.dependency_overrides[get_file_storage] = lambda: LocalFileStorage(
            base_path=str(tmp_path)
        )

        async with client:
            resp = await client.get(
                "/api/v1/assets/task-001/..secret.txt",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 400
        body = resp.json()
        assert body["error"]["code"] == "BAD_REQUEST"

    @pytest.mark.asyncio
    async def test_no_auth_returns_401(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.get("/api/v1/assets/task-001/file.png")
        assert resp.status_code == 401
