"""Tests for teacher-facing API endpoints.

Covers: task library browsing (registry-backed), task detail with content_preview,
roadmap listing, roadmap creation, class insights (with seeded data and 404),
auth enforcement (401) and role enforcement (403) on all endpoints.

Uses httpx.AsyncClient with ASGITransport (async test client). All tests use
explicit @pytest.mark.asyncio per strict mode (Python 3.13.5, Phase 1a note).

Updated: Phase 4a — rewrote TestListLibrary and TestGetTaskDetail to use
real registry data instead of stubs.
"""

from pathlib import Path

import httpx
import pytest
from httpx import ASGITransport

from backend.api import deps
from backend.api.deps import get_current_user, get_task_registry
from backend.main import app
from backend.schemas import ClassInsights, User
from backend.tasks.registry import TaskRegistry
from backend.tasks.schemas import TaskCartridge

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
# Registry test helpers
# ---------------------------------------------------------------------------


def _minimal_cartridge_data(task_id: str, **overrides: object) -> dict:
    """Returns a minimal valid cartridge dict with optional overrides."""
    data: dict = {
        "task_id": task_id,
        "task_type": "static",
        "title": "Testas",
        "description": "Testo aprašymas",
        "version": "1.0",
        "trigger": "urgency",
        "technique": "headline_manipulation",
        "medium": "article",
        "learning_objectives": ["Atpažinti manipuliaciją"],
        "difficulty": 3,
        "time_minutes": 15,
        "is_evergreen": True,
        "is_clean": False,
        "initial_phase": "phase_intro",
        "phases": [
            {
                "id": "phase_intro",
                "title": "Įvadas",
                "is_ai_phase": False,
                "interaction": {
                    "type": "button",
                    "choices": [
                        {"label": "Tęsti", "target_phase": "phase_reveal"},
                    ],
                },
            },
            {
                "id": "phase_reveal",
                "title": "Atskleidimas",
                "is_terminal": True,
                "evaluation_outcome": "trickster_loses",
            },
        ],
        "evaluation": {
            "patterns_embedded": [
                {
                    "id": "p1",
                    "description": "Urgency pattern",
                    "technique": "headline_manipulation",
                    "real_world_connection": "Common in news",
                },
            ],
            "checklist": [
                {
                    "id": "c1",
                    "description": "Identified urgency",
                    "pattern_refs": ["p1"],
                    "is_mandatory": True,
                },
            ],
            "pass_conditions": {
                "trickster_wins": "Mokinys pasidalino",
                "partial": "Mokinys perskaitė, bet praleido",
                "trickster_loses": "Mokinys atpažino technikas",
            },
        },
        "reveal": {"key_lesson": "Antraštė buvo sukurta skubos jausmui sukelti"},
        "safety": {
            "content_boundaries": ["no_real_harm"],
            "intensity_ceiling": 3,
            "cold_start_safe": True,
        },
    }
    data.update(overrides)
    return data


def _build_cartridge(task_id: str, **overrides: object) -> TaskCartridge:
    """Builds a validated TaskCartridge from minimal data with overrides."""
    data = _minimal_cartridge_data(task_id, **overrides)
    return TaskCartridge.model_validate(data)


def _use_registry_with(cartridges: list[TaskCartridge]) -> None:
    """Injects a pre-loaded registry into app dependency overrides."""
    registry = TaskRegistry(Path("/tmp"), Path("/tmp"))
    for c in cartridges:
        registry._by_id[c.task_id] = c
        registry._by_status.setdefault(c.status, set()).add(c.task_id)
        registry._by_trigger[c.trigger].add(c.task_id)
        registry._by_technique[c.technique].add(c.task_id)
        registry._by_medium[c.medium].add(c.task_id)
        for tag in c.tags:
            registry._by_tag[tag].add(c.task_id)
    app.dependency_overrides[get_task_registry] = lambda: registry


# ---------------------------------------------------------------------------
# GET /library
# ---------------------------------------------------------------------------


class TestListLibrary:
    """GET /api/v1/teacher/library — browse task library (registry-backed)."""

    @pytest.mark.asyncio
    async def test_returns_all_active_tasks(self, client: httpx.AsyncClient) -> None:
        _use_teacher()
        c1 = _build_cartridge("task-01")
        c2 = _build_cartridge("task-02", trigger="belonging")
        _use_registry_with([c1, c2])

        async with client:
            resp = await client.get(
                "/api/v1/teacher/library",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        tasks = body["data"]["tasks"]
        assert len(tasks) == 2
        # Verify summary shape
        first = tasks[0]
        assert "task_id" in first
        assert "title" in first
        assert "trigger" in first
        assert "technique" in first
        assert "difficulty" in first
        assert isinstance(first["difficulty"], int)
        assert "status" in first
        assert "task_type" in first

    @pytest.mark.asyncio
    async def test_filters_by_trigger(self, client: httpx.AsyncClient) -> None:
        _use_teacher()
        c1 = _build_cartridge("task-01", trigger="urgency")
        c2 = _build_cartridge("task-02", trigger="belonging")
        _use_registry_with([c1, c2])

        async with client:
            resp = await client.get(
                "/api/v1/teacher/library",
                params={"trigger": "urgency"},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        tasks = resp.json()["data"]["tasks"]
        assert len(tasks) == 1
        assert tasks[0]["task_id"] == "task-01"

    @pytest.mark.asyncio
    async def test_filters_by_difficulty(self, client: httpx.AsyncClient) -> None:
        _use_teacher()
        c1 = _build_cartridge("task-01", difficulty=2)
        c2 = _build_cartridge("task-02", difficulty=4)
        _use_registry_with([c1, c2])

        async with client:
            resp = await client.get(
                "/api/v1/teacher/library",
                params={"difficulty": 2},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        tasks = resp.json()["data"]["tasks"]
        assert len(tasks) == 1
        assert tasks[0]["task_id"] == "task-01"
        assert tasks[0]["difficulty"] == 2

    @pytest.mark.asyncio
    async def test_filters_by_multiple_criteria(self, client: httpx.AsyncClient) -> None:
        _use_teacher()
        c1 = _build_cartridge("task-01", trigger="urgency", medium="article")
        c2 = _build_cartridge("task-02", trigger="urgency", medium="social_post")
        c3 = _build_cartridge("task-03", trigger="belonging", medium="article")
        _use_registry_with([c1, c2, c3])

        async with client:
            resp = await client.get(
                "/api/v1/teacher/library",
                params={"trigger": "urgency", "medium": "article"},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        tasks = resp.json()["data"]["tasks"]
        assert len(tasks) == 1
        assert tasks[0]["task_id"] == "task-01"

    @pytest.mark.asyncio
    async def test_time_max_post_filter(self, client: httpx.AsyncClient) -> None:
        _use_teacher()
        c1 = _build_cartridge("task-01", time_minutes=8)
        c2 = _build_cartridge("task-02", time_minutes=20)
        _use_registry_with([c1, c2])

        async with client:
            resp = await client.get(
                "/api/v1/teacher/library",
                params={"time_max": 10},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        tasks = resp.json()["data"]["tasks"]
        assert len(tasks) == 1
        assert tasks[0]["task_id"] == "task-01"

    @pytest.mark.asyncio
    async def test_tags_comma_separated(self, client: httpx.AsyncClient) -> None:
        _use_teacher()
        c1 = _build_cartridge("task-01", tags=["urgency", "news"])
        c2 = _build_cartridge("task-02", tags=["visual"])
        _use_registry_with([c1, c2])

        async with client:
            resp = await client.get(
                "/api/v1/teacher/library",
                params={"tags": "urgency,news"},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        tasks = resp.json()["data"]["tasks"]
        assert len(tasks) == 1
        assert tasks[0]["task_id"] == "task-01"

    @pytest.mark.asyncio
    async def test_status_defaults_to_active(self, client: httpx.AsyncClient) -> None:
        _use_teacher()
        c1 = _build_cartridge("task-active")
        c2 = _build_cartridge("task-draft", status="draft")
        _use_registry_with([c1, c2])

        async with client:
            resp = await client.get(
                "/api/v1/teacher/library",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        tasks = resp.json()["data"]["tasks"]
        assert len(tasks) == 1
        assert tasks[0]["task_id"] == "task-active"

    @pytest.mark.asyncio
    async def test_status_all_includes_drafts(self, client: httpx.AsyncClient) -> None:
        _use_teacher()
        c1 = _build_cartridge("task-active")
        c2 = _build_cartridge("task-draft", status="draft")
        _use_registry_with([c1, c2])

        async with client:
            resp = await client.get(
                "/api/v1/teacher/library",
                params={"status": "all"},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        tasks = resp.json()["data"]["tasks"]
        assert len(tasks) == 2

    @pytest.mark.asyncio
    async def test_pagination(self, client: httpx.AsyncClient) -> None:
        _use_teacher()
        cartridges = [_build_cartridge(f"task-{i:02d}") for i in range(5)]
        _use_registry_with(cartridges)

        async with client:
            resp = await client.get(
                "/api/v1/teacher/library",
                params={"limit": 2, "offset": 0},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        body = resp.json()
        tasks = body["data"]["tasks"]
        assert len(tasks) == 2
        assert body["data"]["total"] == 5

    @pytest.mark.asyncio
    async def test_empty_registry_returns_empty_list(self, client: httpx.AsyncClient) -> None:
        _use_teacher()
        _use_registry_with([])

        async with client:
            resp = await client.get(
                "/api/v1/teacher/library",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["tasks"] == []
        assert body["data"]["total"] == 0

    @pytest.mark.asyncio
    async def test_search_param_accepted_but_ignored(self, client: httpx.AsyncClient) -> None:
        _use_teacher()
        c1 = _build_cartridge("task-01")
        _use_registry_with([c1])

        async with client:
            resp = await client.get(
                "/api/v1/teacher/library",
                params={"search": "nonexistent search term"},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        # search is ignored — still returns all tasks
        assert len(resp.json()["data"]["tasks"]) == 1

    @pytest.mark.asyncio
    async def test_response_includes_total(self, client: httpx.AsyncClient) -> None:
        _use_teacher()
        c1 = _build_cartridge("task-01")
        c2 = _build_cartridge("task-02")
        _use_registry_with([c1, c2])

        async with client:
            resp = await client.get(
                "/api/v1/teacher/library",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        assert resp.json()["data"]["total"] == 2

    @pytest.mark.asyncio
    async def test_admin_can_access(self, client: httpx.AsyncClient) -> None:
        _use_admin()
        _use_registry_with([_build_cartridge("task-01")])
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
    """GET /api/v1/teacher/library/{task_id} — full task detail (registry-backed)."""

    @pytest.mark.asyncio
    async def test_returns_detail_for_existing_task(self, client: httpx.AsyncClient) -> None:
        _use_teacher()
        c = _build_cartridge("task-01")
        _use_registry_with([c])

        async with client:
            resp = await client.get(
                "/api/v1/teacher/library/task-01",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        data = body["data"]
        assert data["task_id"] == "task-01"
        assert data["title"] == "Testas"
        # Detail fields beyond summary
        assert "description" in data
        assert "content_preview" in data
        assert "learning_objectives" in data

    @pytest.mark.asyncio
    async def test_content_preview_from_text_block(self, client: httpx.AsyncClient) -> None:
        _use_teacher()
        c = _build_cartridge(
            "task-01",
            presentation_blocks=[
                {
                    "id": "pb-01",
                    "type": "text",
                    "text": "SKUBUS PRANEŠIMAS: Naujas reguliavimas įsigalioja vidurnaktį.",
                },
            ],
        )
        _use_registry_with([c])

        async with client:
            resp = await client.get(
                "/api/v1/teacher/library/task-01",
                headers=AUTH_HEADER,
            )
        data = resp.json()["data"]
        assert data["content_preview"] == "SKUBUS PRANEŠIMAS: Naujas reguliavimas įsigalioja vidurnaktį."

    @pytest.mark.asyncio
    async def test_content_preview_skips_image_block(self, client: httpx.AsyncClient) -> None:
        _use_teacher()
        c = _build_cartridge(
            "task-01",
            presentation_blocks=[
                {"id": "pb-img", "type": "image", "src": "chart.png", "alt_text": "Grafikas"},
                {"id": "pb-text", "type": "text", "text": "Šis grafikas rodo manipuliuotą statistiką."},
            ],
        )
        _use_registry_with([c])

        async with client:
            resp = await client.get(
                "/api/v1/teacher/library/task-01",
                headers=AUTH_HEADER,
            )
        data = resp.json()["data"]
        assert data["content_preview"] == "Šis grafikas rodo manipuliuotą statistiką."

    @pytest.mark.asyncio
    async def test_content_preview_empty_for_visual_only(self, client: httpx.AsyncClient) -> None:
        _use_teacher()
        c = _build_cartridge(
            "task-01",
            presentation_blocks=[
                {"id": "pb-img1", "type": "image", "src": "img1.png", "alt_text": "Nuotrauka 1"},
                {"id": "pb-img2", "type": "image", "src": "img2.png", "alt_text": "Nuotrauka 2"},
            ],
        )
        _use_registry_with([c])

        async with client:
            resp = await client.get(
                "/api/v1/teacher/library/task-01",
                headers=AUTH_HEADER,
            )
        data = resp.json()["data"]
        assert data["content_preview"] == ""

    @pytest.mark.asyncio
    async def test_content_preview_truncates_long_text(self, client: httpx.AsyncClient) -> None:
        _use_teacher()
        long_text = "A" * 300
        c = _build_cartridge(
            "task-01",
            presentation_blocks=[
                {"id": "pb-01", "type": "text", "text": long_text},
            ],
        )
        _use_registry_with([c])

        async with client:
            resp = await client.get(
                "/api/v1/teacher/library/task-01",
                headers=AUTH_HEADER,
            )
        data = resp.json()["data"]
        assert data["content_preview"] == "A" * 200 + "..."
        assert len(data["content_preview"]) == 203

    @pytest.mark.asyncio
    async def test_nonexistent_task_returns_404(self, client: httpx.AsyncClient) -> None:
        _use_teacher()
        _use_registry_with([])

        async with client:
            resp = await client.get(
                "/api/v1/teacher/library/nonexistent-task",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 404
        body = resp.json()
        assert body["ok"] is False
        assert body["error"]["code"] == "TASK_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_draft_task_hidden_by_default(self, client: httpx.AsyncClient) -> None:
        _use_teacher()
        c = _build_cartridge("task-draft", status="draft")
        _use_registry_with([c])

        async with client:
            resp = await client.get(
                "/api/v1/teacher/library/task-draft",
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "TASK_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_draft_task_visible_with_include_drafts(self, client: httpx.AsyncClient) -> None:
        _use_teacher()
        c = _build_cartridge("task-draft", status="draft")
        _use_registry_with([c])

        async with client:
            resp = await client.get(
                "/api/v1/teacher/library/task-draft",
                params={"include_drafts": "true"},
                headers=AUTH_HEADER,
            )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["task_id"] == "task-draft"
        assert data["status"] == "draft"

    @pytest.mark.asyncio
    async def test_detail_includes_new_fields(self, client: httpx.AsyncClient) -> None:
        _use_teacher()
        c = _build_cartridge("task-01")
        _use_registry_with([c])

        async with client:
            resp = await client.get(
                "/api/v1/teacher/library/task-01",
                headers=AUTH_HEADER,
            )
        data = resp.json()["data"]
        assert "is_clean" in data
        assert "version" in data
        assert "phase_count" in data
        assert "is_evergreen" in data
        assert "task_type" in data
        assert data["is_clean"] is False
        assert data["version"] == "1.0"
        assert data["phase_count"] == 2
        assert data["is_evergreen"] is True

    @pytest.mark.asyncio
    async def test_admin_can_access(self, client: httpx.AsyncClient) -> None:
        _use_admin()
        _use_registry_with([_build_cartridge("task-01")])
        async with client:
            resp = await client.get(
                "/api/v1/teacher/library/task-01",
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
