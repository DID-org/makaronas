"""Tests for backend.tasks.registry — load, query, reload, stale phase detection,
dependency function, and empty directory handling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.tasks.loader import LoadError, LoadWarning
from backend.tasks.registry import TaskRegistry
from backend.tasks.schemas import TaskCartridge


# ---------------------------------------------------------------------------
# Test taxonomy
# ---------------------------------------------------------------------------

TAXONOMY: dict = {
    "triggers": {"urgency": "Skubumas", "belonging": "Priklausymas"},
    "techniques": {
        "headline_manipulation": "Antraštės manipuliacija",
        "manufactured_deadline": "Dirbtinis terminas",
        "cherry_picking": "Selektyvus citavimas",
    },
    "mediums": {"article": "Straipsnis", "social_post": "Socialinis įrašas"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_cartridge(task_id: str) -> dict:
    """Returns the smallest valid cartridge dict — passes all validation layers."""
    return {
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
                    "technique": "manufactured_deadline",
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


def _write_task(
    tmp_path: Path, task_id: str, overrides: dict | None = None,
) -> Path:
    """Creates ``content/tasks/{task_id}/task.json`` under tmp_path.

    Uses tmp_path as content_dir (tasks/ subdir created inside it).
    Returns the task directory path.
    """
    task_dir = tmp_path / "tasks" / task_id
    task_dir.mkdir(parents=True)
    data = _minimal_cartridge(task_id)
    if overrides:
        data.update(overrides)
    (task_dir / "task.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8",
    )
    return task_dir


def _write_taxonomy(tmp_path: Path, taxonomy: dict | None = None) -> Path:
    """Writes taxonomy.json to tmp_path and returns the path."""
    tax = taxonomy or TAXONOMY
    tax_path = tmp_path / "taxonomy.json"
    tax_path.write_text(json.dumps(tax, ensure_ascii=False), encoding="utf-8")
    return tax_path


def _make_registry(tmp_path: Path) -> TaskRegistry:
    """Creates a TaskRegistry pointing at tmp_path as content_dir."""
    tax_path = _write_taxonomy(tmp_path)
    return TaskRegistry(content_dir=tmp_path, taxonomy_path=tax_path)


# ---------------------------------------------------------------------------
# Load behavior
# ---------------------------------------------------------------------------


class TestLoad:
    """TaskRegistry.load — initial loading and index building."""

    def test_empty_dir_produces_empty_registry(self, tmp_path: Path) -> None:
        registry = _make_registry(tmp_path)
        registry.load()
        assert registry.count() == 0
        assert registry.count("all") == 0
        assert registry.load_errors == []

    def test_no_tasks_subdir_produces_empty_registry(self, tmp_path: Path) -> None:
        registry = _make_registry(tmp_path)
        registry.load()
        assert registry.count() == 0

    def test_loads_valid_tasks(self, tmp_path: Path) -> None:
        _write_task(tmp_path, "task-01")
        _write_task(tmp_path, "task-02")
        registry = _make_registry(tmp_path)
        registry.load()
        assert registry.count() == 2
        assert registry.count("all") == 2

    def test_mix_of_valid_and_invalid(self, tmp_path: Path) -> None:
        _write_task(tmp_path, "task-good")
        # Create invalid cartridge
        bad_dir = tmp_path / "tasks" / "task-bad"
        bad_dir.mkdir(parents=True)
        (bad_dir / "task.json").write_text("{invalid json", encoding="utf-8")

        registry = _make_registry(tmp_path)
        registry.load()
        assert registry.count() == 1
        assert registry.get_task("task-good") is not None
        assert registry.get_task("task-bad") is None
        assert len(registry.load_errors) == 1
        assert registry.load_errors[0].error_type == "invalid_json"

    def test_load_errors_accessible(self, tmp_path: Path) -> None:
        bad_dir = tmp_path / "tasks" / "task-bad"
        bad_dir.mkdir(parents=True)
        (bad_dir / "task.json").write_text("not json!", encoding="utf-8")

        registry = _make_registry(tmp_path)
        registry.load()
        errors = registry.load_errors
        assert len(errors) == 1
        assert isinstance(errors[0], LoadError)

    def test_load_warnings_accessible(self, tmp_path: Path) -> None:
        # Create a task with an unknown trigger to generate a taxonomy warning
        _write_task(tmp_path, "task-warn", {"trigger": "unknown_trigger"})
        registry = _make_registry(tmp_path)
        registry.load()

        warnings = registry.load_warnings
        assert "task-warn" in warnings
        assert any(w.warning_type == "unknown_taxonomy" for w in warnings["task-warn"])


# ---------------------------------------------------------------------------
# get_task
# ---------------------------------------------------------------------------


class TestGetTask:
    """TaskRegistry.get_task — single cartridge lookup by ID."""

    def test_returns_cartridge(self, tmp_path: Path) -> None:
        _write_task(tmp_path, "task-01")
        registry = _make_registry(tmp_path)
        registry.load()
        c = registry.get_task("task-01")
        assert c is not None
        assert isinstance(c, TaskCartridge)
        assert c.task_id == "task-01"

    def test_returns_none_for_unknown(self, tmp_path: Path) -> None:
        registry = _make_registry(tmp_path)
        registry.load()
        assert registry.get_task("nonexistent") is None

    def test_finds_draft_task(self, tmp_path: Path) -> None:
        _write_task(tmp_path, "task-draft", {"status": "draft"})
        registry = _make_registry(tmp_path)
        registry.load()
        c = registry.get_task("task-draft")
        assert c is not None
        assert c.status == "draft"

    def test_finds_deprecated_task(self, tmp_path: Path) -> None:
        _write_task(tmp_path, "task-dep", {"status": "deprecated"})
        registry = _make_registry(tmp_path)
        registry.load()
        c = registry.get_task("task-dep")
        assert c is not None
        assert c.status == "deprecated"


# ---------------------------------------------------------------------------
# Query — single criterion
# ---------------------------------------------------------------------------


class TestQuerySingleCriterion:
    """TaskRegistry.query — filtering by one criterion at a time."""

    def test_filter_by_trigger(self, tmp_path: Path) -> None:
        _write_task(tmp_path, "task-01", {"trigger": "urgency"})
        _write_task(tmp_path, "task-02", {"trigger": "belonging"})
        registry = _make_registry(tmp_path)
        registry.load()

        results = registry.query(trigger="urgency")
        assert len(results) == 1
        assert results[0].task_id == "task-01"

    def test_filter_by_technique(self, tmp_path: Path) -> None:
        _write_task(tmp_path, "task-01", {"technique": "headline_manipulation"})
        _write_task(tmp_path, "task-02", {"technique": "cherry_picking"})
        registry = _make_registry(tmp_path)
        registry.load()

        results = registry.query(technique="cherry_picking")
        assert len(results) == 1
        assert results[0].task_id == "task-02"

    def test_filter_by_medium(self, tmp_path: Path) -> None:
        _write_task(tmp_path, "task-01", {"medium": "article"})
        _write_task(tmp_path, "task-02", {"medium": "social_post"})
        registry = _make_registry(tmp_path)
        registry.load()

        results = registry.query(medium="social_post")
        assert len(results) == 1
        assert results[0].task_id == "task-02"

    def test_filter_by_difficulty_range(self, tmp_path: Path) -> None:
        _write_task(tmp_path, "task-01", {"difficulty": 1})
        _write_task(tmp_path, "task-02", {"difficulty": 3})
        _write_task(tmp_path, "task-03", {"difficulty": 5})
        registry = _make_registry(tmp_path)
        registry.load()

        results = registry.query(difficulty_min=2, difficulty_max=4)
        assert len(results) == 1
        assert results[0].task_id == "task-02"

    def test_filter_by_difficulty_min_only(self, tmp_path: Path) -> None:
        _write_task(tmp_path, "task-01", {"difficulty": 1})
        _write_task(tmp_path, "task-02", {"difficulty": 4})
        registry = _make_registry(tmp_path)
        registry.load()

        results = registry.query(difficulty_min=3)
        assert len(results) == 1
        assert results[0].task_id == "task-02"

    def test_filter_by_difficulty_max_only(self, tmp_path: Path) -> None:
        _write_task(tmp_path, "task-01", {"difficulty": 2})
        _write_task(tmp_path, "task-02", {"difficulty": 5})
        registry = _make_registry(tmp_path)
        registry.load()

        results = registry.query(difficulty_max=3)
        assert len(results) == 1
        assert results[0].task_id == "task-01"

    def test_filter_by_tags(self, tmp_path: Path) -> None:
        _write_task(tmp_path, "task-01", {"tags": ["beginner", "visual"]})
        _write_task(tmp_path, "task-02", {"tags": ["advanced", "visual"]})
        registry = _make_registry(tmp_path)
        registry.load()

        results = registry.query(tags=["beginner"])
        assert len(results) == 1
        assert results[0].task_id == "task-01"

    def test_tags_use_and_logic(self, tmp_path: Path) -> None:
        _write_task(tmp_path, "task-01", {"tags": ["beginner", "visual"]})
        _write_task(tmp_path, "task-02", {"tags": ["advanced", "visual"]})
        registry = _make_registry(tmp_path)
        registry.load()

        # Both have "visual", only task-01 has "beginner"
        results = registry.query(tags=["beginner", "visual"])
        assert len(results) == 1
        assert results[0].task_id == "task-01"

    def test_empty_result_returns_empty_list(self, tmp_path: Path) -> None:
        _write_task(tmp_path, "task-01")
        registry = _make_registry(tmp_path)
        registry.load()

        results = registry.query(trigger="nonexistent")
        assert results == []

    def test_no_filter_returns_all_active(self, tmp_path: Path) -> None:
        _write_task(tmp_path, "task-01")
        _write_task(tmp_path, "task-02")
        registry = _make_registry(tmp_path)
        registry.load()

        results = registry.query()
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Query — multi-criteria
# ---------------------------------------------------------------------------


class TestQueryMultiCriteria:
    """TaskRegistry.query — combining multiple filter criteria."""

    def test_trigger_and_technique(self, tmp_path: Path) -> None:
        _write_task(tmp_path, "task-01", {
            "trigger": "urgency", "technique": "headline_manipulation",
        })
        _write_task(tmp_path, "task-02", {
            "trigger": "urgency", "technique": "cherry_picking",
        })
        _write_task(tmp_path, "task-03", {
            "trigger": "belonging", "technique": "headline_manipulation",
        })
        registry = _make_registry(tmp_path)
        registry.load()

        results = registry.query(trigger="urgency", technique="headline_manipulation")
        assert len(results) == 1
        assert results[0].task_id == "task-01"

    def test_trigger_and_medium_and_difficulty(self, tmp_path: Path) -> None:
        _write_task(tmp_path, "task-01", {
            "trigger": "urgency", "medium": "article", "difficulty": 2,
        })
        _write_task(tmp_path, "task-02", {
            "trigger": "urgency", "medium": "article", "difficulty": 5,
        })
        registry = _make_registry(tmp_path)
        registry.load()

        results = registry.query(trigger="urgency", medium="article", difficulty_max=3)
        assert len(results) == 1
        assert results[0].task_id == "task-01"


# ---------------------------------------------------------------------------
# Query — status filtering
# ---------------------------------------------------------------------------


class TestQueryStatus:
    """TaskRegistry.query — status partitioning."""

    def test_default_returns_active_only(self, tmp_path: Path) -> None:
        _write_task(tmp_path, "task-active")
        _write_task(tmp_path, "task-draft", {"status": "draft"})
        registry = _make_registry(tmp_path)
        registry.load()

        results = registry.query()
        assert len(results) == 1
        assert results[0].task_id == "task-active"

    def test_status_draft(self, tmp_path: Path) -> None:
        _write_task(tmp_path, "task-active")
        _write_task(tmp_path, "task-draft", {"status": "draft"})
        registry = _make_registry(tmp_path)
        registry.load()

        results = registry.query(status="draft")
        assert len(results) == 1
        assert results[0].task_id == "task-draft"

    def test_status_deprecated(self, tmp_path: Path) -> None:
        _write_task(tmp_path, "task-dep", {"status": "deprecated"})
        registry = _make_registry(tmp_path)
        registry.load()

        results = registry.query(status="deprecated")
        assert len(results) == 1
        assert results[0].task_id == "task-dep"

    def test_status_all(self, tmp_path: Path) -> None:
        _write_task(tmp_path, "task-active")
        _write_task(tmp_path, "task-draft", {"status": "draft"})
        _write_task(tmp_path, "task-dep", {"status": "deprecated"})
        registry = _make_registry(tmp_path)
        registry.load()

        results = registry.query(status="all")
        assert len(results) == 3

    def test_demoted_task_in_draft_partition(self, tmp_path: Path) -> None:
        # Task declared active but with is_evergreen=False triggers demotion
        _write_task(tmp_path, "task-demoted", {"is_evergreen": False})
        registry = _make_registry(tmp_path)
        registry.load()

        # Not in active
        active = registry.query(status="active")
        assert all(r.task_id != "task-demoted" for r in active)

        # Should be in draft
        drafts = registry.query(status="draft")
        assert any(r.task_id == "task-demoted" for r in drafts)


# ---------------------------------------------------------------------------
# Query — pagination
# ---------------------------------------------------------------------------


class TestQueryPagination:
    """TaskRegistry.query — limit/offset pagination."""

    def test_limit(self, tmp_path: Path) -> None:
        for i in range(5):
            _write_task(tmp_path, f"task-{i:02d}")
        registry = _make_registry(tmp_path)
        registry.load()

        results = registry.query(limit=2)
        assert len(results) == 2

    def test_offset(self, tmp_path: Path) -> None:
        for i in range(5):
            _write_task(tmp_path, f"task-{i:02d}")
        registry = _make_registry(tmp_path)
        registry.load()

        all_results = registry.query(limit=50)
        offset_results = registry.query(offset=2, limit=50)
        assert offset_results == all_results[2:]

    def test_limit_and_offset(self, tmp_path: Path) -> None:
        for i in range(5):
            _write_task(tmp_path, f"task-{i:02d}")
        registry = _make_registry(tmp_path)
        registry.load()

        results = registry.query(offset=1, limit=2)
        assert len(results) == 2
        all_results = registry.query(limit=50)
        assert results == all_results[1:3]

    def test_results_sorted_by_task_id(self, tmp_path: Path) -> None:
        # Create in non-alphabetical order
        _write_task(tmp_path, "task-zz")
        _write_task(tmp_path, "task-aa")
        _write_task(tmp_path, "task-mm")
        registry = _make_registry(tmp_path)
        registry.load()

        results = registry.query()
        ids = [r.task_id for r in results]
        assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# get_all_task_ids / count
# ---------------------------------------------------------------------------


class TestTaskIdsMethods:
    """TaskRegistry.get_all_task_ids and count."""

    def test_get_all_task_ids_active(self, tmp_path: Path) -> None:
        _write_task(tmp_path, "task-01")
        _write_task(tmp_path, "task-02")
        _write_task(tmp_path, "task-draft", {"status": "draft"})
        registry = _make_registry(tmp_path)
        registry.load()

        ids = registry.get_all_task_ids()
        assert ids == ["task-01", "task-02"]

    def test_get_all_task_ids_all(self, tmp_path: Path) -> None:
        _write_task(tmp_path, "task-01")
        _write_task(tmp_path, "task-draft", {"status": "draft"})
        registry = _make_registry(tmp_path)
        registry.load()

        ids = registry.get_all_task_ids("all")
        assert len(ids) == 2

    def test_count(self, tmp_path: Path) -> None:
        _write_task(tmp_path, "task-01")
        _write_task(tmp_path, "task-02")
        registry = _make_registry(tmp_path)
        registry.load()

        assert registry.count() == 2
        assert registry.count("draft") == 0
        assert registry.count("all") == 2

    def test_get_all_task_ids_sorted(self, tmp_path: Path) -> None:
        _write_task(tmp_path, "task-zz")
        _write_task(tmp_path, "task-aa")
        registry = _make_registry(tmp_path)
        registry.load()

        ids = registry.get_all_task_ids()
        assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# Reload
# ---------------------------------------------------------------------------


class TestReload:
    """TaskRegistry.reload — atomic index replacement."""

    def test_reload_picks_up_new_task(self, tmp_path: Path) -> None:
        _write_task(tmp_path, "task-01")
        registry = _make_registry(tmp_path)
        registry.load()
        assert registry.count() == 1

        _write_task(tmp_path, "task-02")
        registry.reload()
        assert registry.count() == 2
        assert registry.get_task("task-02") is not None

    def test_reload_removes_deleted_task(self, tmp_path: Path) -> None:
        td = _write_task(tmp_path, "task-01")
        registry = _make_registry(tmp_path)
        registry.load()
        assert registry.count() == 1

        # Remove the task from disk
        (td / "task.json").unlink()
        td.rmdir()
        registry.reload()
        assert registry.count() == 0
        assert registry.get_task("task-01") is None

    def test_reload_picks_up_modified_task(self, tmp_path: Path) -> None:
        _write_task(tmp_path, "task-01", {"difficulty": 2})
        registry = _make_registry(tmp_path)
        registry.load()
        assert registry.get_task("task-01").difficulty == 2

        # Overwrite with new difficulty
        task_dir = tmp_path / "tasks" / "task-01"
        data = _minimal_cartridge("task-01")
        data["difficulty"] = 5
        (task_dir / "task.json").write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8",
        )
        registry.reload()
        assert registry.get_task("task-01").difficulty == 5

    def test_reload_re_reads_taxonomy(self, tmp_path: Path) -> None:
        # Start with a known trigger
        _write_task(tmp_path, "task-01", {"trigger": "urgency"})
        registry = _make_registry(tmp_path)
        registry.load()
        assert registry.count() == 1

        # Add new trigger to taxonomy and add a task using it
        tax_path = tmp_path / "taxonomy.json"
        new_tax = dict(TAXONOMY)
        new_tax["triggers"]["conspiracy"] = "Sąmokslas"
        tax_path.write_text(json.dumps(new_tax), encoding="utf-8")

        _write_task(tmp_path, "task-02", {"trigger": "conspiracy"})
        registry.reload()
        assert registry.count() == 2
        # No taxonomy warning for the new trigger
        warnings = registry.load_warnings
        if "task-02" in warnings:
            assert not any(w.warning_type == "unknown_taxonomy" and "trigger" in w.message
                          for w in warnings["task-02"])

    def test_reload_preserves_old_index_on_total_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _write_task(tmp_path, "task-01")
        registry = _make_registry(tmp_path)
        registry.load()
        assert registry.count() == 1

        # Make taxonomy unreadable to trigger total failure
        def _boom(*args, **kwargs):
            raise RuntimeError("Disk on fire")

        monkeypatch.setattr(registry._loader, "load_taxonomy", _boom)
        registry.reload()

        # Old index preserved
        assert registry.count() == 1
        assert registry.get_task("task-01") is not None

    def test_reload_with_broken_new_cartridge(self, tmp_path: Path) -> None:
        _write_task(tmp_path, "task-01")
        registry = _make_registry(tmp_path)
        registry.load()
        assert registry.count() == 1

        # Add a broken cartridge — should not blow up, just error
        bad_dir = tmp_path / "tasks" / "task-bad"
        bad_dir.mkdir(parents=True)
        (bad_dir / "task.json").write_text("nope", encoding="utf-8")

        registry.reload()
        # task-01 still there, task-bad errored
        assert registry.count() == 1
        assert len(registry.load_errors) == 1


# ---------------------------------------------------------------------------
# Stale phase detection
# ---------------------------------------------------------------------------


class TestIsPhaseValid:
    """TaskRegistry.is_phase_valid — stale phase detection for live sessions."""

    def test_valid_phase(self, tmp_path: Path) -> None:
        _write_task(tmp_path, "task-01")
        registry = _make_registry(tmp_path)
        registry.load()

        assert registry.is_phase_valid("task-01", "phase_intro") is True
        assert registry.is_phase_valid("task-01", "phase_reveal") is True

    def test_invalid_phase(self, tmp_path: Path) -> None:
        _write_task(tmp_path, "task-01")
        registry = _make_registry(tmp_path)
        registry.load()

        assert registry.is_phase_valid("task-01", "nonexistent") is False

    def test_nonexistent_task(self, tmp_path: Path) -> None:
        registry = _make_registry(tmp_path)
        registry.load()

        assert registry.is_phase_valid("no-such-task", "any-phase") is False


# ---------------------------------------------------------------------------
# Dependency function (deps.py integration)
# ---------------------------------------------------------------------------


class TestGetTaskRegistryDep:
    """get_task_registry dependency in deps.py."""

    def test_returns_registry_when_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from backend.api import deps

        _write_task(tmp_path, "task-01")
        registry = _make_registry(tmp_path)
        registry.load()

        monkeypatch.setattr(deps, "_task_registry", registry)
        result = deps.get_task_registry()
        assert result is registry

    def test_raises_503_when_none(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from fastapi import HTTPException

        from backend.api import deps

        monkeypatch.setattr(deps, "_task_registry", None)
        with pytest.raises(HTTPException) as exc_info:
            deps.get_task_registry()
        assert exc_info.value.status_code == 503

    def test_503_detail_is_api_response(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from fastapi import HTTPException

        from backend.api import deps

        monkeypatch.setattr(deps, "_task_registry", None)
        with pytest.raises(HTTPException) as exc_info:
            deps.get_task_registry()
        detail = exc_info.value.detail
        assert detail["ok"] is False
        assert detail["error"]["code"] == "SERVICE_UNAVAILABLE"
