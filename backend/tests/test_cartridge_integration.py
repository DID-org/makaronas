"""Integration tests for reference cartridges (Phases 5a + 5b).

Loads real cartridge JSON files from content/tasks/ and verifies them
through the full pipeline: loader -> registry -> teacher API -> student API.
No mocks for cartridge data — the point is to prove real content works.

Phase 5a: 2 full-content cartridges (clickbait-trap, follow-money).
Phase 5b: 4 skeleton cartridges (cherry-pick, phantom-quote, wedge, misleading-frame).

Uses httpx.AsyncClient with ASGITransport (async test client). All tests
use explicit @pytest.mark.asyncio per strict mode.
"""

from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

from backend.api import deps
from backend.api.deps import get_current_user, get_task_registry
from backend.main import app
from backend.schemas import GameSession, User
from backend.tasks.loader import TaskLoader
from backend.tasks.registry import TaskRegistry
from backend.tasks.schemas import (
    ChatMessageBlock,
    FreeformInteraction,
    ImageBlock,
    SearchResultBlock,
    SocialPostBlock,
    TaskCartridge,
)

# ---------------------------------------------------------------------------
# Skeleton task IDs (Phase 5b)
# ---------------------------------------------------------------------------

SKELETON_IDS = [
    "task-cherry-pick-001",
    "task-phantom-quote-001",
    "task-wedge-001",
    "task-misleading-frame-001",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AUTH_HEADER = {"Authorization": "Bearer test-token-123"}
FAKE_USER_ID = "fake-user-1"
FAKE_SCHOOL_ID = "school-test-1"

# Resolve the project content directory relative to this test file
# backend/tests/test_cartridge_integration.py -> backend/tests -> backend -> project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONTENT_DIR = PROJECT_ROOT / "content"
TAXONOMY_PATH = CONTENT_DIR / "taxonomy.json"

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


@pytest.fixture(autouse=True)
def _cleanup_overrides():
    """Ensures dependency overrides are cleaned up after each test."""
    yield
    app.dependency_overrides.clear()


@pytest.fixture(scope="module")
def taxonomy() -> dict:
    """Loads the real taxonomy from content/taxonomy.json."""
    loader = TaskLoader()
    return loader.load_taxonomy(TAXONOMY_PATH)


@pytest.fixture(scope="module")
def registry() -> TaskRegistry:
    """Builds a real registry loaded from the content directory."""
    reg = TaskRegistry(content_dir=CONTENT_DIR, taxonomy_path=TAXONOMY_PATH)
    reg.load()
    return reg


def _use_teacher() -> None:
    """Injects teacher user for the current test."""
    app.dependency_overrides[get_current_user] = lambda: TEACHER_USER


def _use_student() -> None:
    """Injects student user for the current test."""
    app.dependency_overrides[get_current_user] = lambda: User(
        id=FAKE_USER_ID, role="student", name="Test Student",
        school_id=FAKE_SCHOOL_ID,
    )


def _use_registry(reg: TaskRegistry) -> None:
    """Injects the given registry into the app dependency system."""
    app.dependency_overrides[get_task_registry] = lambda: reg


# ---------------------------------------------------------------------------
# Loader validation tests
# ---------------------------------------------------------------------------


class TestLoaderValidation:
    """Both cartridges load via TaskLoader with zero errors and no demotion."""

    def test_task_clickbait_loads_clean(self, taxonomy: dict) -> None:
        """Task 1 loads with no errors and no demotion-triggering warnings."""
        loader = TaskLoader()
        task_dir = CONTENT_DIR / "tasks" / "task-clickbait-trap-001"
        result = loader.load_task(task_dir, taxonomy)

        assert result.cartridge.task_id == "task-clickbait-trap-001"
        assert result.cartridge.status == "active"
        assert result.cartridge.task_type == "hybrid"
        # No demotion warnings — status stays active
        demotion_warnings = [
            w for w in result.warnings
            if w.warning_type not in ("unknown_taxonomy", "missing_prompt_dir")
        ]
        assert demotion_warnings == []

    def test_task_follow_money_loads_clean(self, taxonomy: dict) -> None:
        """Task 4 loads with no errors and no demotion-triggering warnings."""
        loader = TaskLoader()
        task_dir = CONTENT_DIR / "tasks" / "task-follow-money-001"
        result = loader.load_task(task_dir, taxonomy)

        assert result.cartridge.task_id == "task-follow-money-001"
        assert result.cartridge.status == "active"
        assert result.cartridge.task_type == "hybrid"
        demotion_warnings = [
            w for w in result.warnings
            if w.warning_type not in ("unknown_taxonomy", "missing_prompt_dir")
        ]
        assert demotion_warnings == []

    def test_task_clickbait_has_correct_metadata(self, taxonomy: dict) -> None:
        """Task 1 has expected classification fields."""
        loader = TaskLoader()
        task_dir = CONTENT_DIR / "tasks" / "task-clickbait-trap-001"
        result = loader.load_task(task_dir, taxonomy)
        c = result.cartridge

        assert c.trigger == "urgency"
        assert c.technique == "headline_manipulation"
        assert c.medium == "article"
        assert c.difficulty == 2
        assert c.time_minutes == 10
        assert c.is_evergreen is True
        assert c.is_clean is False
        assert c.language == "lt"

    def test_task_follow_money_has_correct_metadata(self, taxonomy: dict) -> None:
        """Task 4 has expected classification fields."""
        loader = TaskLoader()
        task_dir = CONTENT_DIR / "tasks" / "task-follow-money-001"
        result = loader.load_task(task_dir, taxonomy)
        c = result.cartridge

        assert c.trigger == "cynicism"
        assert c.technique == "omission"
        assert c.medium == "investigation"
        assert c.difficulty == 4
        assert c.time_minutes == 20
        assert c.is_evergreen is True
        assert c.is_clean is False
        assert c.language == "lt"


# ---------------------------------------------------------------------------
# Graph integrity tests
# ---------------------------------------------------------------------------


class TestGraphIntegrity:
    """Phase graph validation: reachability, terminals, bounded cycles."""

    def _get_cartridge(self, task_id: str, taxonomy: dict) -> TaskCartridge:
        """Loads a single cartridge by task_id."""
        loader = TaskLoader()
        task_dir = CONTENT_DIR / "tasks" / task_id
        return loader.load_task(task_dir, taxonomy).cartridge

    def test_clickbait_all_phases_reachable(self, taxonomy: dict) -> None:
        """All 7 phases are reachable from initial_phase via BFS."""
        c = self._get_cartridge("task-clickbait-trap-001", taxonomy)
        phase_map = {p.id: p for p in c.phases}
        reachable = set()
        queue = [c.initial_phase]

        while queue:
            pid = queue.pop(0)
            if pid in reachable:
                continue
            reachable.add(pid)
            phase = phase_map.get(pid)
            if phase is None:
                continue
            # Collect targets from buttons
            if phase.interaction and hasattr(phase.interaction, "choices"):
                for choice in phase.interaction.choices:
                    queue.append(choice.target_phase)
            # Collect targets from AI transitions
            if phase.ai_transitions:
                queue.extend([
                    phase.ai_transitions.on_success,
                    phase.ai_transitions.on_partial,
                    phase.ai_transitions.on_max_exchanges,
                ])

        assert reachable == set(phase_map.keys()), (
            f"Unreachable phases: {set(phase_map.keys()) - reachable}"
        )

    def test_clickbait_has_terminal_phases(self, taxonomy: dict) -> None:
        """Task 1 has at least one terminal phase."""
        c = self._get_cartridge("task-clickbait-trap-001", taxonomy)
        terminals = [p for p in c.phases if p.is_terminal]
        assert len(terminals) >= 1
        # Verify all terminal phases have evaluation outcomes
        for t in terminals:
            assert t.evaluation_outcome is not None

    def test_follow_money_all_phases_reachable(self, taxonomy: dict) -> None:
        """All 6 phases are reachable from initial_phase via BFS."""
        c = self._get_cartridge("task-follow-money-001", taxonomy)
        phase_map = {p.id: p for p in c.phases}
        reachable = set()
        queue = [c.initial_phase]

        while queue:
            pid = queue.pop(0)
            if pid in reachable:
                continue
            reachable.add(pid)
            phase = phase_map.get(pid)
            if phase is None:
                continue
            if phase.interaction and hasattr(phase.interaction, "choices"):
                for choice in phase.interaction.choices:
                    queue.append(choice.target_phase)
            if phase.interaction and hasattr(phase.interaction, "submit_target"):
                queue.append(phase.interaction.submit_target)
            if phase.ai_transitions:
                queue.extend([
                    phase.ai_transitions.on_success,
                    phase.ai_transitions.on_partial,
                    phase.ai_transitions.on_max_exchanges,
                ])

        assert reachable == set(phase_map.keys()), (
            f"Unreachable phases: {set(phase_map.keys()) - reachable}"
        )

    def test_follow_money_has_terminal_phases(self, taxonomy: dict) -> None:
        """Task 4 has at least one terminal phase."""
        c = self._get_cartridge("task-follow-money-001", taxonomy)
        terminals = [p for p in c.phases if p.is_terminal]
        assert len(terminals) >= 1
        for t in terminals:
            assert t.evaluation_outcome is not None

    def test_both_tasks_have_ai_and_static_phases(self, taxonomy: dict) -> None:
        """Hybrid tasks must have at least one AI and one static phase."""
        for task_id in ("task-clickbait-trap-001", "task-follow-money-001"):
            c = self._get_cartridge(task_id, taxonomy)
            ai_phases = [p for p in c.phases if p.is_ai_phase]
            static_phases = [p for p in c.phases if not p.is_ai_phase]
            assert len(ai_phases) >= 1, f"{task_id}: no AI phases"
            assert len(static_phases) >= 1, f"{task_id}: no static phases"


# ---------------------------------------------------------------------------
# Registry indexing tests
# ---------------------------------------------------------------------------


class TestRegistryIndexing:
    """Both cartridges index correctly and are queryable."""

    def test_both_tasks_indexed(self, registry: TaskRegistry) -> None:
        """Registry contains both cartridges as active."""
        assert registry.count("active") >= 2
        assert registry.get_task("task-clickbait-trap-001") is not None
        assert registry.get_task("task-follow-money-001") is not None

    def test_query_by_trigger(self, registry: TaskRegistry) -> None:
        """Query by trigger returns the correct task."""
        results = registry.query(trigger="urgency")
        ids = [t.task_id for t in results]
        assert "task-clickbait-trap-001" in ids

        results = registry.query(trigger="cynicism")
        ids = [t.task_id for t in results]
        assert "task-follow-money-001" in ids

    def test_query_by_medium(self, registry: TaskRegistry) -> None:
        """Query by medium returns the correct task."""
        results = registry.query(medium="article")
        ids = [t.task_id for t in results]
        assert "task-clickbait-trap-001" in ids

        results = registry.query(medium="investigation")
        ids = [t.task_id for t in results]
        assert "task-follow-money-001" in ids

    def test_query_by_technique(self, registry: TaskRegistry) -> None:
        """Query by technique returns the correct task."""
        results = registry.query(technique="headline_manipulation")
        ids = [t.task_id for t in results]
        assert "task-clickbait-trap-001" in ids

        results = registry.query(technique="omission")
        ids = [t.task_id for t in results]
        assert "task-follow-money-001" in ids

    def test_query_by_difficulty_range(self, registry: TaskRegistry) -> None:
        """Query by difficulty range returns correct tasks."""
        easy = registry.query(difficulty_max=2)
        easy_ids = [t.task_id for t in easy]
        assert "task-clickbait-trap-001" in easy_ids

        hard = registry.query(difficulty_min=4)
        hard_ids = [t.task_id for t in hard]
        assert "task-follow-money-001" in hard_ids

    def test_phase_validity_checks(self, registry: TaskRegistry) -> None:
        """Phase validity returns correct results for real phases."""
        assert registry.is_phase_valid("task-clickbait-trap-001", "intro") is True
        assert registry.is_phase_valid("task-clickbait-trap-001", "evaluate") is True
        assert registry.is_phase_valid("task-clickbait-trap-001", "nonexistent") is False
        assert registry.is_phase_valid("task-follow-money-001", "investigation") is True
        assert registry.is_phase_valid("nonexistent-task", "intro") is False


# ---------------------------------------------------------------------------
# Teacher API tests
# ---------------------------------------------------------------------------


class TestTeacherAPI:
    """Teacher library endpoints serve real cartridge data."""

    @pytest.mark.asyncio
    async def test_library_returns_both_tasks(
        self, client: httpx.AsyncClient, registry: TaskRegistry,
    ) -> None:
        """GET /teacher/library returns both tasks in the listing."""
        _use_teacher()
        _use_registry(registry)

        resp = await client.get(
            "/api/v1/teacher/library", headers=AUTH_HEADER,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        task_ids = [t["task_id"] for t in body["data"]["tasks"]]
        assert "task-clickbait-trap-001" in task_ids
        assert "task-follow-money-001" in task_ids

    @pytest.mark.asyncio
    async def test_library_filter_by_medium(
        self, client: httpx.AsyncClient, registry: TaskRegistry,
    ) -> None:
        """GET /teacher/library?medium=investigation returns only Task 4."""
        _use_teacher()
        _use_registry(registry)

        resp = await client.get(
            "/api/v1/teacher/library?medium=investigation", headers=AUTH_HEADER,
        )
        assert resp.status_code == 200
        body = resp.json()
        task_ids = [t["task_id"] for t in body["data"]["tasks"]]
        assert "task-follow-money-001" in task_ids
        assert "task-clickbait-trap-001" not in task_ids

    @pytest.mark.asyncio
    async def test_task_detail_has_content_preview(
        self, client: httpx.AsyncClient, registry: TaskRegistry,
    ) -> None:
        """GET /teacher/library/{task_id} returns non-empty content_preview."""
        _use_teacher()
        _use_registry(registry)

        resp = await client.get(
            "/api/v1/teacher/library/task-clickbait-trap-001",
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 200
        body = resp.json()
        detail = body["data"]
        assert detail["task_id"] == "task-clickbait-trap-001"
        assert isinstance(detail["content_preview"], str)
        assert len(detail["content_preview"]) > 0
        assert isinstance(detail["difficulty"], int)
        assert detail["difficulty"] == 2

    @pytest.mark.asyncio
    async def test_task_detail_follow_money(
        self, client: httpx.AsyncClient, registry: TaskRegistry,
    ) -> None:
        """GET /teacher/library/{task_id} returns correct Task 4 detail."""
        _use_teacher()
        _use_registry(registry)

        resp = await client.get(
            "/api/v1/teacher/library/task-follow-money-001",
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 200
        body = resp.json()
        detail = body["data"]
        assert detail["task_id"] == "task-follow-money-001"
        assert detail["difficulty"] == 4
        assert detail["medium"] == "investigation"
        assert len(detail["content_preview"]) > 0


# ---------------------------------------------------------------------------
# Student API tests
# ---------------------------------------------------------------------------


class TestStudentAPI:
    """Student endpoint serves correct initial phase data."""

    @pytest_asyncio.fixture
    async def session_id(self) -> str:
        """Creates a session in the store and returns its ID."""
        session = GameSession(
            session_id="test-integration-session",
            student_id=FAKE_USER_ID,
            school_id=FAKE_SCHOOL_ID,
        )
        await deps._session_store.save_session(session)
        return session.session_id

    @pytest.mark.asyncio
    async def test_clickbait_initial_phase(
        self, client: httpx.AsyncClient, registry: TaskRegistry,
        session_id: str,
    ) -> None:
        """Next-task for clickbait returns button_click action and content."""
        _use_student()
        _use_registry(registry)

        resp = await client.get(
            f"/api/v1/student/session/{session_id}/next"
            "?task_id=task-clickbait-trap-001",
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        data = body["data"]
        assert data["task_id"] == "task-clickbait-trap-001"
        assert data["current_phase"] == "intro"
        assert "button_click" in data["available_actions"]
        assert len(data["content"]) > 0

    @pytest.mark.asyncio
    async def test_follow_money_initial_phase(
        self, client: httpx.AsyncClient, registry: TaskRegistry,
        session_id: str,
    ) -> None:
        """Next-task for follow-money returns button_click action and content."""
        _use_student()
        _use_registry(registry)

        resp = await client.get(
            f"/api/v1/student/session/{session_id}/next"
            "?task_id=task-follow-money-001",
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        data = body["data"]
        assert data["task_id"] == "task-follow-money-001"
        assert data["current_phase"] == "intro"
        assert "button_click" in data["available_actions"]
        assert len(data["content"]) >= 2  # two article blocks

    @pytest.mark.asyncio
    async def test_clickbait_has_trickster_intro(
        self, client: httpx.AsyncClient, registry: TaskRegistry,
        session_id: str,
    ) -> None:
        """Initial phase for clickbait has trickster_intro set."""
        _use_student()
        _use_registry(registry)

        resp = await client.get(
            f"/api/v1/student/session/{session_id}/next"
            "?task_id=task-clickbait-trap-001",
            headers=AUTH_HEADER,
        )
        body = resp.json()
        assert body["data"]["trickster_intro"] is not None
        assert len(body["data"]["trickster_intro"]) > 0


# ---------------------------------------------------------------------------
# Investigation tree navigability tests
# ---------------------------------------------------------------------------


class TestInvestigationTree:
    """Task 4's investigation tree is structurally navigable."""

    def _get_follow_money(self, taxonomy: dict) -> TaskCartridge:
        """Loads the Follow the Money cartridge."""
        loader = TaskLoader()
        task_dir = CONTENT_DIR / "tasks" / "task-follow-money-001"
        return loader.load_task(task_dir, taxonomy).cartridge

    def test_all_child_queries_resolve(self, taxonomy: dict) -> None:
        """Every child_queries string matches some SearchResultBlock's query."""
        c = self._get_follow_money(taxonomy)
        sr_blocks = [
            b for b in c.presentation_blocks
            if isinstance(b, SearchResultBlock)
        ]
        # Build a set of all available query strings
        available_queries = {b.query for b in sr_blocks}

        # Check all child_queries resolve
        for block in sr_blocks:
            for child_q in block.child_queries:
                assert child_q in available_queries, (
                    f"SearchResultBlock '{block.id}' has child_query "
                    f"'{child_q}' that doesn't match any block's query field. "
                    f"Available queries: {sorted(available_queries)}"
                )

    def test_starting_queries_resolve(self, taxonomy: dict) -> None:
        """InvestigationInteraction starting_queries all resolve to blocks."""
        c = self._get_follow_money(taxonomy)
        sr_blocks = [
            b for b in c.presentation_blocks
            if isinstance(b, SearchResultBlock)
        ]
        available_queries = {b.query for b in sr_blocks}

        # Find the investigation phase
        inv_phase = None
        for phase in c.phases:
            if phase.id == "investigation":
                inv_phase = phase
                break
        assert inv_phase is not None, "No investigation phase found"
        assert inv_phase.interaction is not None

        for sq in inv_phase.interaction.starting_queries:
            assert sq in available_queries, (
                f"Starting query '{sq}' doesn't match any block's query field"
            )

    def test_key_findings_count(self, taxonomy: dict) -> None:
        """Task 4 has at least 2 key findings (matching min_key_findings)."""
        c = self._get_follow_money(taxonomy)
        sr_blocks = [
            b for b in c.presentation_blocks
            if isinstance(b, SearchResultBlock)
        ]
        key_findings = [b for b in sr_blocks if b.is_key_finding]
        assert len(key_findings) >= 2, (
            f"Expected at least 2 key findings, got {len(key_findings)}"
        )

    def test_dead_ends_have_no_children(self, taxonomy: dict) -> None:
        """Dead-end blocks should not have child_queries."""
        c = self._get_follow_money(taxonomy)
        sr_blocks = [
            b for b in c.presentation_blocks
            if isinstance(b, SearchResultBlock)
        ]
        for block in sr_blocks:
            if block.is_dead_end:
                assert block.child_queries == [], (
                    f"Dead-end block '{block.id}' has child_queries: "
                    f"{block.child_queries}"
                )

    def test_both_branches_reachable(self, taxonomy: dict) -> None:
        """Both key financial connections are discoverable from starting queries."""
        c = self._get_follow_money(taxonomy)
        sr_blocks = {
            b.query: b for b in c.presentation_blocks
            if isinstance(b, SearchResultBlock)
        }

        # BFS from starting queries to find key findings
        inv_phase = next(p for p in c.phases if p.id == "investigation")
        visited = set()
        queue = list(inv_phase.interaction.starting_queries)
        found_key_findings = []

        while queue:
            query = queue.pop(0)
            if query in visited:
                continue
            visited.add(query)
            block = sr_blocks.get(query)
            if block is None:
                continue
            if block.is_key_finding:
                found_key_findings.append(block.id)
            for child_q in block.child_queries:
                queue.append(child_q)

        assert len(found_key_findings) >= 2, (
            f"Only {len(found_key_findings)} key findings reachable: "
            f"{found_key_findings}"
        )


# ---------------------------------------------------------------------------
# Phase 5b: Skeleton cartridge loader tests
# ---------------------------------------------------------------------------


class TestSkeletonLoaderValidation:
    """All 4 skeleton cartridges load via TaskLoader with no errors."""

    def _load(self, task_id: str, taxonomy: dict) -> tuple:
        """Loads a skeleton cartridge and returns (cartridge, warnings)."""
        loader = TaskLoader()
        task_dir = CONTENT_DIR / "tasks" / task_id
        result = loader.load_task(task_dir, taxonomy)
        return result.cartridge, result.warnings

    def test_cherry_pick_loads_as_draft(self, taxonomy: dict) -> None:
        """Task 2 loads cleanly with draft status preserved."""
        c, warnings = self._load("task-cherry-pick-001", taxonomy)
        assert c.task_id == "task-cherry-pick-001"
        assert c.status == "draft"
        assert c.task_type == "hybrid"
        assert c.medium == "social_post"
        assert c.trigger == "authority"
        assert c.technique == "cherry_picking"
        assert c.difficulty == 2
        assert len(warnings) == 0

    def test_phantom_quote_loads_as_draft(self, taxonomy: dict) -> None:
        """Task 3 loads cleanly with draft status preserved."""
        c, warnings = self._load("task-phantom-quote-001", taxonomy)
        assert c.task_id == "task-phantom-quote-001"
        assert c.status == "draft"
        assert c.task_type == "ai_driven"
        assert c.medium == "article"
        assert c.trigger == "belonging"
        assert c.technique == "phantom_quote"
        assert c.difficulty == 3
        assert len(warnings) == 0

    def test_wedge_loads_as_draft(self, taxonomy: dict) -> None:
        """Task 5 loads cleanly with draft status preserved."""
        c, warnings = self._load("task-wedge-001", taxonomy)
        assert c.task_id == "task-wedge-001"
        assert c.status == "draft"
        assert c.task_type == "ai_driven"
        assert c.medium == "chat"
        assert c.trigger == "identity"
        assert c.technique == "wedge_driving"
        assert c.difficulty == 3
        assert len(warnings) == 0

    def test_misleading_frame_loads_as_draft(self, taxonomy: dict) -> None:
        """Task 6 loads cleanly with draft status and no asset warnings."""
        c, warnings = self._load("task-misleading-frame-001", taxonomy)
        assert c.task_id == "task-misleading-frame-001"
        assert c.status == "draft"
        assert c.task_type == "ai_driven"
        assert c.medium == "image"
        assert c.trigger == "fear"
        assert c.technique == "emotional_framing"
        assert c.difficulty == 3
        # Placeholder PNGs prevent asset-related warnings
        assert len(warnings) == 0

    def test_all_skeletons_load_together(self, taxonomy: dict) -> None:
        """All 6 cartridges load via load_all_tasks with zero errors."""
        loader = TaskLoader()
        results, errors = loader.load_all_tasks(CONTENT_DIR, taxonomy)
        # TEMPLATE directory produces a known path_mismatch error — filter it
        real_errors = [e for e in errors if "TEMPLATE" not in e.task_dir]
        assert len(real_errors) == 0, f"Load errors: {real_errors}"
        assert len(results) == 6
        loaded_ids = sorted(r.cartridge.task_id for r in results)
        assert "task-cherry-pick-001" in loaded_ids
        assert "task-phantom-quote-001" in loaded_ids
        assert "task-wedge-001" in loaded_ids
        assert "task-misleading-frame-001" in loaded_ids


# ---------------------------------------------------------------------------
# Phase 5b: Skeleton graph integrity tests
# ---------------------------------------------------------------------------


class TestSkeletonGraphIntegrity:
    """Phase graph validation for all 4 skeleton cartridges."""

    def _get_cartridge(self, task_id: str, taxonomy: dict) -> TaskCartridge:
        """Loads a single cartridge by task_id."""
        loader = TaskLoader()
        task_dir = CONTENT_DIR / "tasks" / task_id
        return loader.load_task(task_dir, taxonomy).cartridge

    def _bfs_reachable(self, cartridge: TaskCartridge) -> set[str]:
        """Returns the set of phase IDs reachable from initial_phase via BFS."""
        phase_map = {p.id: p for p in cartridge.phases}
        reachable = set()
        queue = [cartridge.initial_phase]
        while queue:
            pid = queue.pop(0)
            if pid in reachable:
                continue
            reachable.add(pid)
            phase = phase_map.get(pid)
            if phase is None:
                continue
            if phase.interaction and hasattr(phase.interaction, "choices"):
                for choice in phase.interaction.choices:
                    queue.append(choice.target_phase)
            if phase.interaction and hasattr(phase.interaction, "submit_target"):
                queue.append(phase.interaction.submit_target)
            if phase.ai_transitions:
                queue.extend([
                    phase.ai_transitions.on_success,
                    phase.ai_transitions.on_partial,
                    phase.ai_transitions.on_max_exchanges,
                ])
        return reachable

    @pytest.mark.parametrize("task_id", SKELETON_IDS)
    def test_all_phases_reachable(self, task_id: str, taxonomy: dict) -> None:
        """Every phase is reachable from initial_phase."""
        c = self._get_cartridge(task_id, taxonomy)
        phase_ids = {p.id for p in c.phases}
        reachable = self._bfs_reachable(c)
        assert reachable == phase_ids, (
            f"{task_id}: unreachable phases: {phase_ids - reachable}"
        )

    @pytest.mark.parametrize("task_id", SKELETON_IDS)
    def test_terminal_phases_have_outcomes(
        self, task_id: str, taxonomy: dict,
    ) -> None:
        """Every skeleton has terminal phases with evaluation outcomes."""
        c = self._get_cartridge(task_id, taxonomy)
        terminals = [p for p in c.phases if p.is_terminal]
        assert len(terminals) >= 1, f"{task_id}: no terminal phases"
        for t in terminals:
            assert t.evaluation_outcome is not None, (
                f"{task_id}: terminal '{t.id}' missing evaluation_outcome"
            )

    def test_hybrid_has_ai_and_static(self, taxonomy: dict) -> None:
        """Task 2 (hybrid) has both AI and static phases."""
        c = self._get_cartridge("task-cherry-pick-001", taxonomy)
        ai_phases = [p for p in c.phases if p.is_ai_phase]
        static_phases = [p for p in c.phases if not p.is_ai_phase]
        assert len(ai_phases) >= 1, "hybrid task has no AI phases"
        assert len(static_phases) >= 1, "hybrid task has no static phases"

    @pytest.mark.parametrize("task_id", [
        "task-phantom-quote-001",
        "task-wedge-001",
        "task-misleading-frame-001",
    ])
    def test_ai_driven_has_ai_phase(
        self, task_id: str, taxonomy: dict,
    ) -> None:
        """ai_driven tasks have at least one AI phase."""
        c = self._get_cartridge(task_id, taxonomy)
        ai_phases = [p for p in c.phases if p.is_ai_phase]
        assert len(ai_phases) >= 1, f"{task_id}: no AI phases"


# ---------------------------------------------------------------------------
# Phase 5b: Skeleton block type tests
# ---------------------------------------------------------------------------


class TestSkeletonBlockTypes:
    """Verifies that skeleton cartridges use the expected block types."""

    def _get_cartridge(self, task_id: str, taxonomy: dict) -> TaskCartridge:
        """Loads a single cartridge by task_id."""
        loader = TaskLoader()
        task_dir = CONTENT_DIR / "tasks" / task_id
        return loader.load_task(task_dir, taxonomy).cartridge

    def test_cherry_pick_has_social_post_block(self, taxonomy: dict) -> None:
        """Task 2 contains a SocialPostBlock with expected fields."""
        c = self._get_cartridge("task-cherry-pick-001", taxonomy)
        sp_blocks = [
            b for b in c.presentation_blocks
            if isinstance(b, SocialPostBlock)
        ]
        assert len(sp_blocks) == 1
        sp = sp_blocks[0]
        assert sp.id == "social-post"
        assert sp.author == "SveikasProtas"
        assert sp.engagement is not None
        assert sp.cited_source is not None

    def test_wedge_has_chat_message_blocks(self, taxonomy: dict) -> None:
        """Task 5 contains 5 ChatMessageBlocks with expected structure."""
        c = self._get_cartridge("task-wedge-001", taxonomy)
        cm_blocks = [
            b for b in c.presentation_blocks
            if isinstance(b, ChatMessageBlock)
        ]
        assert len(cm_blocks) == 5
        # Exactly one highlighted
        highlighted = [b for b in cm_blocks if b.is_highlighted]
        assert len(highlighted) == 1
        assert highlighted[0].id == "msg-wedge"
        # All have usernames
        for b in cm_blocks:
            assert len(b.username) > 0

    def test_misleading_frame_has_image_blocks(self, taxonomy: dict) -> None:
        """Task 6 contains 2 ImageBlocks with accessibility fields."""
        c = self._get_cartridge("task-misleading-frame-001", taxonomy)
        img_blocks = [
            b for b in c.presentation_blocks
            if isinstance(b, ImageBlock)
        ]
        assert len(img_blocks) == 2
        for ib in img_blocks:
            assert len(ib.alt_text) > 0, f"ImageBlock '{ib.id}' missing alt_text"
            assert ib.audio_description is not None, (
                f"ImageBlock '{ib.id}' missing audio_description"
            )

    def test_phantom_quote_initial_phase_is_ai(self, taxonomy: dict) -> None:
        """Task 3 (ai_driven) starts directly with an AI freeform phase."""
        c = self._get_cartridge("task-phantom-quote-001", taxonomy)
        phase_map = {p.id: p for p in c.phases}
        initial = phase_map[c.initial_phase]
        assert initial.is_ai_phase is True
        assert isinstance(initial.interaction, FreeformInteraction)
        assert initial.interaction.min_exchanges == 3
        assert initial.interaction.max_exchanges == 8


# ---------------------------------------------------------------------------
# Phase 5b: Full registry indexing (all 6 cartridges)
# ---------------------------------------------------------------------------


class TestFullRegistryIndexing:
    """All 6 cartridges index correctly in the registry."""

    def test_total_cartridge_count(self, registry: TaskRegistry) -> None:
        """Registry contains 2 active + 4 draft = 6 total."""
        assert registry.count("active") >= 2
        assert registry.count("draft") >= 4

    def test_all_six_retrievable(self, registry: TaskRegistry) -> None:
        """All 6 cartridges are retrievable by ID."""
        all_ids = [
            "task-clickbait-trap-001",
            "task-follow-money-001",
            "task-cherry-pick-001",
            "task-phantom-quote-001",
            "task-wedge-001",
            "task-misleading-frame-001",
        ]
        for task_id in all_ids:
            c = registry.get_task(task_id)
            assert c is not None, f"{task_id} not found in registry"

    def test_query_drafts(self, registry: TaskRegistry) -> None:
        """Querying status=draft returns all 4 skeletons."""
        drafts = registry.query(status="draft")
        draft_ids = {t.task_id for t in drafts}
        for sk_id in SKELETON_IDS:
            assert sk_id in draft_ids, f"{sk_id} not in draft results"

    def test_query_by_new_mediums(self, registry: TaskRegistry) -> None:
        """New mediums are queryable: social_post, chat, image."""
        social = registry.query(medium="social_post", status="draft")
        assert any(t.task_id == "task-cherry-pick-001" for t in social)

        chat = registry.query(medium="chat", status="draft")
        assert any(t.task_id == "task-wedge-001" for t in chat)

        image = registry.query(medium="image", status="draft")
        assert any(t.task_id == "task-misleading-frame-001" for t in image)

    def test_query_by_new_triggers(self, registry: TaskRegistry) -> None:
        """New triggers are queryable: authority, belonging, identity, fear."""
        for trigger, expected_id in [
            ("authority", "task-cherry-pick-001"),
            ("belonging", "task-phantom-quote-001"),
            ("identity", "task-wedge-001"),
            ("fear", "task-misleading-frame-001"),
        ]:
            results = registry.query(trigger=trigger, status="draft")
            ids = [t.task_id for t in results]
            assert expected_id in ids, (
                f"trigger={trigger}: expected {expected_id}, got {ids}"
            )

    def test_phase_validity_for_skeletons(self, registry: TaskRegistry) -> None:
        """Phase validity works for skeleton cartridges."""
        assert registry.is_phase_valid("task-cherry-pick-001", "intro")
        assert registry.is_phase_valid("task-cherry-pick-001", "evaluate")
        assert registry.is_phase_valid("task-phantom-quote-001", "evaluate")
        assert registry.is_phase_valid("task-wedge-001", "evaluate")
        assert registry.is_phase_valid("task-misleading-frame-001", "evaluate")
        assert not registry.is_phase_valid("task-wedge-001", "nonexistent")


# ---------------------------------------------------------------------------
# Phase 5b: Teacher API with all 6 cartridges
# ---------------------------------------------------------------------------


class TestTeacherAPIFull:
    """Teacher library serves all 6 cartridges with correct filtering."""

    @pytest.mark.asyncio
    async def test_library_drafts_listing(
        self, client: httpx.AsyncClient, registry: TaskRegistry,
    ) -> None:
        """GET /teacher/library?status=draft returns the 4 skeletons."""
        _use_teacher()
        _use_registry(registry)

        resp = await client.get(
            "/api/v1/teacher/library?status=draft", headers=AUTH_HEADER,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        task_ids = {t["task_id"] for t in body["data"]["tasks"]}
        for sk_id in SKELETON_IDS:
            assert sk_id in task_ids, f"{sk_id} not in draft listing"

    @pytest.mark.asyncio
    async def test_library_filter_by_medium_social_post(
        self, client: httpx.AsyncClient, registry: TaskRegistry,
    ) -> None:
        """Filtering by medium=social_post returns Task 2."""
        _use_teacher()
        _use_registry(registry)

        resp = await client.get(
            "/api/v1/teacher/library?medium=social_post&status=draft",
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 200
        body = resp.json()
        task_ids = [t["task_id"] for t in body["data"]["tasks"]]
        assert "task-cherry-pick-001" in task_ids

    @pytest.mark.asyncio
    async def test_skeleton_detail_with_include_drafts(
        self, client: httpx.AsyncClient, registry: TaskRegistry,
    ) -> None:
        """Draft skeletons are accessible via detail endpoint with include_drafts."""
        _use_teacher()
        _use_registry(registry)

        resp = await client.get(
            "/api/v1/teacher/library/task-cherry-pick-001?include_drafts=true",
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 200
        body = resp.json()
        detail = body["data"]
        assert detail["task_id"] == "task-cherry-pick-001"
        assert detail["status"] == "draft"
        assert detail["medium"] == "social_post"
        # SocialPostBlock produces a content_preview
        assert len(detail["content_preview"]) > 0

    @pytest.mark.asyncio
    async def test_skeleton_detail_hidden_without_include_drafts(
        self, client: httpx.AsyncClient, registry: TaskRegistry,
    ) -> None:
        """Draft skeletons return 404 when include_drafts is not set."""
        _use_teacher()
        _use_registry(registry)

        resp = await client.get(
            "/api/v1/teacher/library/task-cherry-pick-001",
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Phase 5b: Student API draft rejection
# ---------------------------------------------------------------------------


class TestStudentAPIDraftRejection:
    """Student endpoint correctly rejects draft cartridges."""

    @pytest_asyncio.fixture
    async def session_id(self) -> str:
        """Creates a session in the store and returns its ID."""
        session = GameSession(
            session_id="test-skeleton-session",
            student_id=FAKE_USER_ID,
            school_id=FAKE_SCHOOL_ID,
        )
        await deps._session_store.save_session(session)
        return session.session_id

    @pytest.mark.asyncio
    @pytest.mark.parametrize("task_id", SKELETON_IDS)
    async def test_draft_task_returns_404(
        self, task_id: str, client: httpx.AsyncClient,
        registry: TaskRegistry, session_id: str,
    ) -> None:
        """Student API returns 404 for draft skeleton cartridges."""
        _use_student()
        _use_registry(registry)

        resp = await client.get(
            f"/api/v1/student/session/{session_id}/next?task_id={task_id}",
            headers=AUTH_HEADER,
        )
        assert resp.status_code == 404
