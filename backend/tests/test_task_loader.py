"""Tests for backend.tasks.loader — taxonomy loading, task loading, Path=Identity,
task ID validation, error handling, taxonomy warnings, and batch loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.tasks.loader import LoadError, LoadResult, LoadWarning, TaskLoader
from backend.tasks.schemas import TaskCartridge


# ---------------------------------------------------------------------------
# Test taxonomy
# ---------------------------------------------------------------------------

TAXONOMY: dict = {
    "triggers": {"urgency": "Skubumas"},
    "techniques": {"headline_manipulation": "Antraštės manipuliacija",
                   "manufactured_deadline": "Dirbtinis terminas"},
    "mediums": {"article": "Straipsnis"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_cartridge(task_id: str) -> dict:
    """Returns the smallest valid cartridge dict — matches test_task_schemas pattern."""
    return {
        "task_id": task_id,
        "task_type": "hybrid",
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
    """Creates ``content/tasks/{task_id}/task.json`` with minimal valid data.

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


# ---------------------------------------------------------------------------
# Taxonomy loading
# ---------------------------------------------------------------------------


class TestLoadTaxonomy:
    """TaskLoader.load_taxonomy — read and parse taxonomy.json."""

    def test_valid_taxonomy(self, tmp_path: Path) -> None:
        tax_file = tmp_path / "taxonomy.json"
        tax_data = {
            "triggers": {"urgency": "Skubumas"},
            "techniques": {"cherry_picking": "Selektyvus citavimas"},
            "mediums": {"article": "Straipsnis"},
        }
        tax_file.write_text(json.dumps(tax_data, ensure_ascii=False), encoding="utf-8")

        loader = TaskLoader()
        result = loader.load_taxonomy(tax_file)
        assert result == tax_data
        assert "urgency" in result["triggers"]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            TaskLoader().load_taxonomy(tmp_path / "nonexistent.json")

    def test_malformed_json_raises(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "taxonomy.json"
        bad_file.write_text("{broken json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            TaskLoader().load_taxonomy(bad_file)


# ---------------------------------------------------------------------------
# Single task loading — happy path
# ---------------------------------------------------------------------------


class TestLoadTaskSuccess:
    """TaskLoader.load_task — successful loading."""

    def test_valid_task(self, tmp_path: Path) -> None:
        task_dir = _write_task(tmp_path, "task-test-001")
        result = TaskLoader().load_task(task_dir, TAXONOMY)

        assert isinstance(result, LoadResult)
        assert isinstance(result.cartridge, TaskCartridge)
        assert result.cartridge.task_id == "task-test-001"
        assert result.cartridge.task_type == "hybrid"
        assert result.warnings == []

    def test_cartridge_is_frozen(self, tmp_path: Path) -> None:
        task_dir = _write_task(tmp_path, "task-test-001")
        result = TaskLoader().load_task(task_dir, TAXONOMY)
        with pytest.raises(Exception):
            result.cartridge.task_id = "changed"  # type: ignore[misc]

    def test_taxonomy_context_injected(self, tmp_path: Path) -> None:
        """Known taxonomy values produce no warnings."""
        task_dir = _write_task(tmp_path, "task-test-001")
        result = TaskLoader().load_task(task_dir, TAXONOMY)
        assert result.warnings == []


# ---------------------------------------------------------------------------
# Path=Identity enforcement
# ---------------------------------------------------------------------------


class TestPathIdentity:
    """Path=Identity — task_id must match directory name."""

    def test_mismatch_raises(self, tmp_path: Path) -> None:
        """task_id in JSON differs from directory name."""
        task_dir = tmp_path / "tasks" / "task-dir-001"
        task_dir.mkdir(parents=True)
        data = _minimal_cartridge("task-different-001")
        (task_dir / "task.json").write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8",
        )

        with pytest.raises(LoadError) as exc_info:
            TaskLoader().load_task(task_dir, TAXONOMY)
        assert exc_info.value.error_type == "path_mismatch"
        assert "task-different-001" in exc_info.value.message
        assert "task-dir-001" in exc_info.value.message

    def test_match_succeeds(self, tmp_path: Path) -> None:
        task_dir = _write_task(tmp_path, "task-example-01")
        result = TaskLoader().load_task(task_dir, TAXONOMY)
        assert result.cartridge.task_id == "task-example-01"


# ---------------------------------------------------------------------------
# Task ID character validation
# ---------------------------------------------------------------------------


class TestTaskIdValidation:
    """Task ID character rules: ^[a-z0-9][a-z0-9-]*[a-z0-9]$ (min 2 chars)."""

    @pytest.mark.parametrize("task_id", [
        "task-clickbait-001",
        "ab",
        "a0",
        "task-01",
        "x1y2z3",
    ])
    def test_valid_ids(self, tmp_path: Path, task_id: str) -> None:
        task_dir = _write_task(tmp_path, task_id)
        result = TaskLoader().load_task(task_dir, TAXONOMY)
        assert result.cartridge.task_id == task_id

    @pytest.mark.parametrize("task_id,reason", [
        ("Task-Bad", "uppercase"),
        ("task_bad", "underscore"),
        ("-leading", "leading hyphen"),
        ("trailing-", "trailing hyphen"),
        ("a", "too short (1 char)"),
        ("has spaces", "spaces"),
        ("ALLCAPS", "uppercase"),
    ])
    def test_invalid_ids(self, tmp_path: Path, task_id: str, reason: str) -> None:
        task_dir = tmp_path / "tasks" / task_id
        task_dir.mkdir(parents=True)
        data = _minimal_cartridge("placeholder")
        data["task_id"] = task_id
        (task_dir / "task.json").write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8",
        )

        with pytest.raises(LoadError) as exc_info:
            TaskLoader().load_task(task_dir, TAXONOMY)
        assert exc_info.value.error_type == "invalid_task_id"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestLoadTaskErrors:
    """LoadError cases — missing file, bad JSON, validation failure."""

    def test_missing_task_json(self, tmp_path: Path) -> None:
        task_dir = tmp_path / "tasks" / "task-empty-01"
        task_dir.mkdir(parents=True)
        # No task.json written

        with pytest.raises(LoadError) as exc_info:
            TaskLoader().load_task(task_dir, TAXONOMY)
        assert exc_info.value.error_type == "missing_file"

    def test_invalid_json(self, tmp_path: Path) -> None:
        task_dir = tmp_path / "tasks" / "task-bad-json-01"
        task_dir.mkdir(parents=True)
        (task_dir / "task.json").write_text("{not valid json", encoding="utf-8")

        with pytest.raises(LoadError) as exc_info:
            TaskLoader().load_task(task_dir, TAXONOMY)
        assert exc_info.value.error_type == "invalid_json"
        assert str(task_dir / "task.json") in exc_info.value.message

    def test_json_not_dict_reports_validation_error(self, tmp_path: Path) -> None:
        """Valid JSON but not a dict → validation_error (Pydantic rejects it)."""
        task_dir = tmp_path / "tasks" / "task-array-01"
        task_dir.mkdir(parents=True)
        (task_dir / "task.json").write_text("[1, 2, 3]", encoding="utf-8")

        with pytest.raises(LoadError) as exc_info:
            TaskLoader().load_task(task_dir, TAXONOMY)
        assert exc_info.value.error_type == "validation_error"

    def test_schema_validation_failure(self, tmp_path: Path) -> None:
        """Missing required fields → validation_error."""
        task_dir = tmp_path / "tasks" / "task-incomplete-01"
        task_dir.mkdir(parents=True)
        (task_dir / "task.json").write_text(
            json.dumps({"task_id": "task-incomplete-01"}),
            encoding="utf-8",
        )

        with pytest.raises(LoadError) as exc_info:
            TaskLoader().load_task(task_dir, TAXONOMY)
        assert exc_info.value.error_type == "validation_error"

    def test_missing_task_id_reports_validation_error(self, tmp_path: Path) -> None:
        """Missing task_id in JSON — Pydantic catches it as validation_error."""
        task_dir = tmp_path / "tasks" / "task-no-id-01"
        task_dir.mkdir(parents=True)
        data = _minimal_cartridge("task-no-id-01")
        del data["task_id"]
        (task_dir / "task.json").write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8",
        )

        with pytest.raises(LoadError) as exc_info:
            TaskLoader().load_task(task_dir, TAXONOMY)
        assert exc_info.value.error_type == "validation_error"


# ---------------------------------------------------------------------------
# Taxonomy warnings
# ---------------------------------------------------------------------------


class TestTaxonomyWarnings:
    """Taxonomy warnings captured as LoadWarning objects."""

    def test_unknown_trigger_produces_warning(self, tmp_path: Path) -> None:
        task_dir = _write_task(tmp_path, "task-warn-01", {"trigger": "panic"})
        result = TaskLoader().load_task(task_dir, TAXONOMY)

        assert len(result.warnings) >= 1
        trigger_warnings = [w for w in result.warnings if "panic" in w.message]
        assert len(trigger_warnings) == 1
        assert trigger_warnings[0].task_id == "task-warn-01"
        assert trigger_warnings[0].warning_type == "unknown_taxonomy"

    def test_known_values_no_warnings(self, tmp_path: Path) -> None:
        task_dir = _write_task(tmp_path, "task-clean-01")
        result = TaskLoader().load_task(task_dir, TAXONOMY)
        assert result.warnings == []

    def test_multiple_unknown_values(self, tmp_path: Path) -> None:
        """Unknown trigger + unknown medium → two warnings."""
        task_dir = _write_task(
            tmp_path, "task-multi-warn-01",
            {"trigger": "panic", "medium": "hologram"},
        )
        result = TaskLoader().load_task(task_dir, TAXONOMY)

        assert len(result.warnings) >= 2
        messages = [w.message for w in result.warnings]
        assert any("panic" in m for m in messages)
        assert any("hologram" in m for m in messages)

    def test_draft_in_progress_warning_captured(self, tmp_path: Path) -> None:
        """is_clean=False with empty patterns_embedded → draft warning captured."""
        overrides = {
            "is_clean": False,
            "evaluation": {
                "patterns_embedded": [],
                "checklist": [],
                "pass_conditions": {
                    "trickster_wins": "Mokinys pasidalino",
                    "partial": "Mokinys perskaitė",
                    "trickster_loses": "Mokinys atpažino",
                },
            },
        }
        task_dir = _write_task(tmp_path, "task-draft-01", overrides)
        result = TaskLoader().load_task(task_dir, TAXONOMY)

        assert len(result.warnings) >= 1
        draft_warnings = [w for w in result.warnings if "draft" in w.message]
        assert len(draft_warnings) == 1


# ---------------------------------------------------------------------------
# Batch loading (load_all_tasks)
# ---------------------------------------------------------------------------


class TestLoadAllTasks:
    """TaskLoader.load_all_tasks — batch loading and error collection."""

    def test_two_valid_tasks(self, tmp_path: Path) -> None:
        _write_task(tmp_path, "task-alpha-01")
        _write_task(tmp_path, "task-beta-01")

        successes, errors = TaskLoader().load_all_tasks(tmp_path, TAXONOMY)

        assert len(successes) == 2
        assert len(errors) == 0
        ids = {r.cartridge.task_id for r in successes}
        assert ids == {"task-alpha-01", "task-beta-01"}

    def test_one_valid_one_invalid(self, tmp_path: Path) -> None:
        _write_task(tmp_path, "task-good-01")
        # Create invalid task — bad JSON
        bad_dir = tmp_path / "tasks" / "task-bad-01"
        bad_dir.mkdir(parents=True)
        (bad_dir / "task.json").write_text("{broken", encoding="utf-8")

        successes, errors = TaskLoader().load_all_tasks(tmp_path, TAXONOMY)

        assert len(successes) == 1
        assert successes[0].cartridge.task_id == "task-good-01"
        assert len(errors) == 1
        assert errors[0].error_type == "invalid_json"

    def test_empty_tasks_directory(self, tmp_path: Path) -> None:
        (tmp_path / "tasks").mkdir()

        successes, errors = TaskLoader().load_all_tasks(tmp_path, TAXONOMY)

        assert successes == []
        assert errors == []

    def test_missing_tasks_directory(self, tmp_path: Path) -> None:
        """No tasks/ directory at all — returns empty, no error."""
        successes, errors = TaskLoader().load_all_tasks(tmp_path, TAXONOMY)

        assert successes == []
        assert errors == []

    def test_skips_non_directories(self, tmp_path: Path) -> None:
        """Files like README.md in tasks/ are skipped."""
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        (tasks_dir / "README.md").write_text("# Tasks", encoding="utf-8")
        _write_task(tmp_path, "task-only-01")

        successes, errors = TaskLoader().load_all_tasks(tmp_path, TAXONOMY)

        assert len(successes) == 1
        assert len(errors) == 0

    def test_skips_directories_without_task_json(self, tmp_path: Path) -> None:
        """Directories without task.json (e.g. TEMPLATE/) are skipped."""
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        (tasks_dir / "TEMPLATE").mkdir()
        _write_task(tmp_path, "task-real-01")

        successes, errors = TaskLoader().load_all_tasks(tmp_path, TAXONOMY)

        assert len(successes) == 1
        assert len(errors) == 0


# ---------------------------------------------------------------------------
# LoadError structured fields
# ---------------------------------------------------------------------------


class TestLoadErrorStructure:
    """LoadError carries structured fields for reporting."""

    def test_load_error_is_exception(self) -> None:
        err = LoadError("dir", "missing_file", "No task.json")
        assert isinstance(err, Exception)

    def test_load_error_fields(self) -> None:
        err = LoadError("content/tasks/task-bad", "path_mismatch", "IDs differ")
        assert err.task_dir == "content/tasks/task-bad"
        assert err.error_type == "path_mismatch"
        assert err.message == "IDs differ"
        assert str(err) == "IDs differ"

    def test_load_warning_frozen(self) -> None:
        w = LoadWarning("task-01", "unknown_taxonomy", "Unknown trigger")
        with pytest.raises(Exception):
            w.task_id = "changed"  # type: ignore[misc]

    def test_load_result_frozen(self) -> None:
        # Can't easily construct a full result without a cartridge,
        # but we can verify the dataclass is frozen
        assert LoadResult.__dataclass_params__.frozen  # type: ignore[attr-defined]
