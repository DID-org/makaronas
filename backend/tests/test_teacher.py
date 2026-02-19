"""Tests for Phase 4b — Teacher-facing API endpoints.

Covers: task library browsing, task detail, roadmap listing, roadmap creation,
class insights (with seeded data and 404), auth enforcement (401) and role
enforcement (403) on all endpoints.

Uses httpx.AsyncClient with ASGITransport (async test client). All tests use
explicit @pytest.mark.asyncio per strict mode (Python 3.13.5, Phase 1a note).
"""

import httpx
import pytest
from httpx import ASGITransport

from backend.api import deps
from backend.api.deps import get_current_user
from backend.main import app
from backend.schemas import ClassInsights, User

AUTH_HEADER = {"Authorization": "Bearer test-token-123"}

FAKE_SCHOOL_ID = "school-test-1"

TEACHER_USER = User(
    id="teacher-1", role="teacher", name="Test Teacher", school_id=FAKE_SCHOOL_ID
)

ADMIN_USER = User(
    id="admin-1", role="admin", name="Test Admin", school_id=FAKE_SCHOOL_ID
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


# ---------------------------------------------------------------------------
# GET /library
# ---------------------------------------------------------------------------


class TestListLibrary:
    """GET /api/v1/teacher/library — browse task library."""

    @pytest.mark.asyncio
    async def test_returns_task_list(self, client: httpx.AsyncClient) -> None:
        _use_teacher()
        async with client:
            resp = await client.get(
                "/api/v1/teacher/library",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        tasks = body["data"]["tasks"]
        assert isinstance(tasks, list)
        assert len(tasks) >= 1
        # Verify summary shape
        first = tasks[0]
        assert "task_id" in first
        assert "title" in first
        assert "trigger" in first
        assert "technique" in first

    @pytest.mark.asyncio
    async def test_accepts_query_params(self, client: httpx.AsyncClient) -> None:
        _use_teacher()
        async with client:
            resp = await client.get(
                "/api/v1/teacher/library",
                params={
                    "trigger": "urgency",
                    "technique": "bandwagon",
                    "medium": "article",
                    "difficulty": "beginner",
                    "time_max": 15,
                    "tags": "urgency,news",
                    "search": "deadline",
                },
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True

    @pytest.mark.asyncio
    async def test_admin_can_access(self, client: httpx.AsyncClient) -> None:
        _use_admin()
        async with client:
            resp = await client.get(
                "/api/v1/teacher/library",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_student_returns_403(self, client: httpx.AsyncClient) -> None:
        # Default auth returns student role — don't override
        async with client:
            resp = await client.get(
                "/api/v1/teacher/library",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 403
        body = resp.json()
        assert body["ok"] is False
        assert body["error"]["code"] == "FORBIDDEN"

    @pytest.mark.asyncio
    async def test_no_auth_returns_401(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.get("/api/v1/teacher/library")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# GET /library/{task_id}
# ---------------------------------------------------------------------------


class TestGetTaskDetail:
    """GET /api/v1/teacher/library/{task_id} — full task detail."""

    @pytest.mark.asyncio
    async def test_returns_task_detail(self, client: httpx.AsyncClient) -> None:
        _use_teacher()
        async with client:
            resp = await client.get(
                "/api/v1/teacher/library/task-urgency-article-001",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        data = body["data"]
        assert "task_id" in data
        assert "title" in data
        # Detail fields beyond summary
        assert "description" in data
        assert "content_preview" in data
        assert "learning_objectives" in data

    @pytest.mark.asyncio
    async def test_admin_can_access(self, client: httpx.AsyncClient) -> None:
        _use_admin()
        async with client:
            resp = await client.get(
                "/api/v1/teacher/library/any-task-id",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_student_returns_403(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.get(
                "/api/v1/teacher/library/any-task-id",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 403
        body = resp.json()
        assert body["error"]["code"] == "FORBIDDEN"

    @pytest.mark.asyncio
    async def test_no_auth_returns_401(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.get("/api/v1/teacher/library/any-task-id")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# GET /roadmaps
# ---------------------------------------------------------------------------


class TestListRoadmaps:
    """GET /api/v1/teacher/roadmaps — list available roadmaps."""

    @pytest.mark.asyncio
    async def test_returns_roadmap_list(self, client: httpx.AsyncClient) -> None:
        _use_teacher()
        async with client:
            resp = await client.get(
                "/api/v1/teacher/roadmaps",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        roadmaps = body["data"]["roadmaps"]
        assert isinstance(roadmaps, list)
        assert len(roadmaps) >= 1
        # Verify shape
        first = roadmaps[0]
        assert "roadmap_id" in first
        assert "title" in first

    @pytest.mark.asyncio
    async def test_student_returns_403(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.get(
                "/api/v1/teacher/roadmaps",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_no_auth_returns_401(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.get("/api/v1/teacher/roadmaps")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# POST /roadmaps
# ---------------------------------------------------------------------------


class TestCreateRoadmap:
    """POST /api/v1/teacher/roadmaps — create custom roadmap."""

    @pytest.mark.asyncio
    async def test_creates_roadmap(self, client: httpx.AsyncClient) -> None:
        _use_teacher()
        async with client:
            resp = await client.post(
                "/api/v1/teacher/roadmaps",
                json={
                    "title": "My Custom Sequence",
                    "task_ids": ["task-001", "task-002", "task-003"],
                    "notes": "Focus on urgency triggers",
                },
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        data = body["data"]
        assert "roadmap_id" in data
        assert data["title"] == "My Custom Sequence"
        assert data["task_count"] == 3

    @pytest.mark.asyncio
    async def test_notes_optional(self, client: httpx.AsyncClient) -> None:
        _use_teacher()
        async with client:
            resp = await client.post(
                "/api/v1/teacher/roadmaps",
                json={
                    "title": "No Notes Roadmap",
                    "task_ids": ["task-001"],
                },
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True

    @pytest.mark.asyncio
    async def test_missing_title_returns_422(self, client: httpx.AsyncClient) -> None:
        _use_teacher()
        async with client:
            resp = await client.post(
                "/api/v1/teacher/roadmaps",
                json={"task_ids": ["task-001"]},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_missing_task_ids_returns_422(self, client: httpx.AsyncClient) -> None:
        _use_teacher()
        async with client:
            resp = await client.post(
                "/api/v1/teacher/roadmaps",
                json={"title": "No Tasks"},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_student_returns_403(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.post(
                "/api/v1/teacher/roadmaps",
                json={
                    "title": "Should Fail",
                    "task_ids": ["task-001"],
                },
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_no_auth_returns_401(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.post(
                "/api/v1/teacher/roadmaps",
                json={
                    "title": "No Auth",
                    "task_ids": ["task-001"],
                },
            )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# GET /class/{class_id}/insights
# ---------------------------------------------------------------------------


class TestClassInsights:
    """GET /api/v1/teacher/class/{class_id}/insights — class-level patterns."""

    @pytest.mark.asyncio
    async def test_returns_insights_when_seeded(self, client: httpx.AsyncClient) -> None:
        _use_teacher()
        insights = ClassInsights(
            class_id="class-9a",
            school_id=FAKE_SCHOOL_ID,
            trigger_distribution={"urgency": 0.6, "belonging": 0.3, "injustice": 0.1},
            common_failure_points=["manufactured_deadline", "bandwagon"],
            growth_trends={"week_1": 0.4, "week_2": 0.6},
        )
        # seed_class_insights is SYNC — no await
        deps._database.seed_class_insights(insights)

        async with client:
            resp = await client.get(
                "/api/v1/teacher/class/class-9a/insights",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        data = body["data"]
        assert data["class_id"] == "class-9a"
        assert data["school_id"] == FAKE_SCHOOL_ID
        assert data["trigger_distribution"]["urgency"] == 0.6
        assert "manufactured_deadline" in data["common_failure_points"]

    @pytest.mark.asyncio
    async def test_wrong_school_returns_404(self, client: httpx.AsyncClient) -> None:
        """Teacher from school-test-1 can't see insights from another school."""
        _use_teacher()
        insights = ClassInsights(
            class_id="class-other",
            school_id="school-other",
        )
        deps._database.seed_class_insights(insights)

        async with client:
            resp = await client.get(
                "/api/v1/teacher/class/class-other/insights",
                headers=AUTH_HEADER,
            )
        # Teacher's school_id is school-test-1, insights are for school-other
        assert resp.status_code == 404
        body = resp.json()
        assert body["error"]["code"] == "CLASS_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_nonexistent_class_returns_404(self, client: httpx.AsyncClient) -> None:
        _use_teacher()
        async with client:
            resp = await client.get(
                "/api/v1/teacher/class/nonexistent-class/insights",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 404
        body = resp.json()
        assert body["ok"] is False
        assert body["error"]["code"] == "CLASS_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_admin_can_access(self, client: httpx.AsyncClient) -> None:
        _use_admin()
        insights = ClassInsights(
            class_id="class-admin-test",
            school_id=FAKE_SCHOOL_ID,
        )
        deps._database.seed_class_insights(insights)

        async with client:
            resp = await client.get(
                "/api/v1/teacher/class/class-admin-test/insights",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_student_returns_403(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.get(
                "/api/v1/teacher/class/any-class/insights",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 403
        body = resp.json()
        assert body["error"]["code"] == "FORBIDDEN"

    @pytest.mark.asyncio
    async def test_no_auth_returns_401(self, client: httpx.AsyncClient) -> None:
        async with client:
            resp = await client.get("/api/v1/teacher/class/any-class/insights")
        assert resp.status_code == 401
