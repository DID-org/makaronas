"""Task loader — reads cartridges from disk and produces validated TaskCartridge instances.

Reads ``content/taxonomy.json`` once at startup and injects it as Pydantic
validation context into every ``TaskCartridge.model_validate()`` call.
Enforces Path=Identity (directory name == task_id) and task ID character rules
at load time (Framework P20).

Tier 2 module: imports from ``backend.tasks.schemas`` (Tier 1) + stdlib.
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from backend.tasks.schemas import TaskCartridge, TaxonomyWarning

# Task ID character rule: lowercase alphanumeric + internal hyphens, min 2 chars.
# Appears in API URL paths, student profile references, and directory names.
_TASK_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*[a-z0-9]$")


# ---------------------------------------------------------------------------
# Data types — structured error/warning reporting
# ---------------------------------------------------------------------------


class LoadError(Exception):
    """Fatal loading failure for a single task.

    Raised by ``load_task`` and collected by ``load_all_tasks``.
    Extends ``Exception`` so it can be raised directly, while carrying
    structured fields for batch error reporting.

    Attributes:
        task_dir: The directory that was being loaded (as string).
        error_type: One of ``"missing_file"``, ``"invalid_json"``,
            ``"validation_error"``, ``"path_mismatch"``, ``"invalid_task_id"``.
        message: Human-readable error description.
    """

    def __init__(self, task_dir: str, error_type: str, message: str) -> None:
        self.task_dir = task_dir
        self.error_type = error_type
        self.message = message
        super().__init__(message)


@dataclass(frozen=True)
class LoadWarning:
    """Non-fatal issue encountered during task loading.

    Attributes:
        task_id: The task that produced the warning.
        warning_type: Category string (e.g. ``"unknown_taxonomy"``).
        message: Human-readable warning description.
    """

    task_id: str
    warning_type: str
    message: str


@dataclass(frozen=True)
class LoadResult:
    """Successful load outcome — a validated cartridge bundled with warnings.

    Attributes:
        cartridge: The frozen ``TaskCartridge`` instance.
        warnings: Any non-fatal warnings emitted during validation.
    """

    cartridge: TaskCartridge
    warnings: list[LoadWarning]


# ---------------------------------------------------------------------------
# TaskLoader
# ---------------------------------------------------------------------------


class TaskLoader:
    """Reads and validates task cartridges from disk.

    Stateless — each method is an independent operation.  Phase 3a's
    ``TaskRegistry`` holds the singleton; the loader is just a tool
    the registry uses.
    """

    def load_taxonomy(self, taxonomy_path: Path) -> dict:
        """Reads and parses ``taxonomy.json``.

        Args:
            taxonomy_path: Path to the taxonomy JSON file.

        Returns:
            Parsed dict with ``triggers``, ``techniques``, ``mediums`` keys.

        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        with open(taxonomy_path, encoding="utf-8") as f:
            return json.load(f)

    def load_task(self, task_dir: Path, taxonomy: dict) -> LoadResult:
        """Loads and validates a single task cartridge from disk.

        Sequence:
        1. Read ``task_dir / "task.json"`` and parse JSON.
        2. If ``task_id`` is present in parsed data — validate characters and
           Path=Identity (directory name must match).
        3. Call ``TaskCartridge.model_validate()`` with taxonomy context.
        4. Collect ``TaxonomyWarning`` emissions as ``LoadWarning`` objects.

        Args:
            task_dir: Directory containing ``task.json``.
            taxonomy: Parsed taxonomy dict for context injection.

        Returns:
            ``LoadResult`` with the frozen cartridge and any warnings.

        Raises:
            LoadError: On any fatal failure (missing file, bad JSON,
                validation error, path mismatch, bad task ID characters).
        """
        task_file = task_dir / "task.json"
        dir_str = str(task_dir)

        # Step 1: Read and parse JSON
        try:
            with open(task_file, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            raise LoadError(
                task_dir=dir_str,
                error_type="missing_file",
                message=f"No task.json found in {dir_str}",
            )
        except json.JSONDecodeError as exc:
            raise LoadError(
                task_dir=dir_str,
                error_type="invalid_json",
                message=f"Invalid JSON in {task_file}: {exc}",
            )

        # Step 2: Pre-validation checks on task_id (if present)
        # Guard: if JSON parsed as non-dict, skip pre-checks and let Pydantic report it.
        task_id = data.get("task_id") if isinstance(data, dict) else None
        if task_id is not None:
            if not isinstance(task_id, str) or not _TASK_ID_RE.match(task_id):
                raise LoadError(
                    task_dir=dir_str,
                    error_type="invalid_task_id",
                    message=(
                        f"task_id '{task_id}' contains invalid characters — "
                        f"must match {_TASK_ID_RE.pattern} "
                        f"(lowercase alphanumeric + internal hyphens, min 2 chars)"
                    ),
                )
            if task_id != task_dir.name:
                raise LoadError(
                    task_dir=dir_str,
                    error_type="path_mismatch",
                    message=(
                        f"task_id '{task_id}' does not match "
                        f"directory name '{task_dir.name}'"
                    ),
                )

        # Step 3: Schema validation with taxonomy context injection
        # Capture TaxonomyWarning emissions during model_validate()
        load_warnings: list[LoadWarning] = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                cartridge = TaskCartridge.model_validate(
                    data, context={"taxonomy": taxonomy},
                )
        except ValidationError as exc:
            raise LoadError(
                task_dir=dir_str,
                error_type="validation_error",
                message=f"Schema validation failed for {dir_str}: {exc}",
            )

        # Step 4: Convert captured TaxonomyWarning into LoadWarning objects
        for w in caught:
            if issubclass(w.category, TaxonomyWarning):
                load_warnings.append(
                    LoadWarning(
                        task_id=cartridge.task_id,
                        warning_type="unknown_taxonomy",
                        message=str(w.message),
                    )
                )

        return LoadResult(cartridge=cartridge, warnings=load_warnings)

    def load_all_tasks(
        self, content_dir: Path, taxonomy: dict,
    ) -> tuple[list[LoadResult], list[LoadError]]:
        """Scans ``content_dir / "tasks"`` and loads all task cartridges.

        Iterates child directories of the tasks folder, skipping non-directories
        and directories that don't contain ``task.json``.  Never raises — all
        errors are collected and returned.

        Args:
            content_dir: The content root (e.g. ``content/``).
            taxonomy: Parsed taxonomy dict for context injection.

        Returns:
            Tuple of (successes, errors) where each success is a ``LoadResult``
            and each error is a ``LoadError``.
        """
        tasks_dir = content_dir / "tasks"
        if not tasks_dir.is_dir():
            return ([], [])

        successes: list[LoadResult] = []
        errors: list[LoadError] = []

        for child in sorted(tasks_dir.iterdir()):
            if not child.is_dir():
                continue
            if not (child / "task.json").exists():
                continue
            try:
                result = self.load_task(child, taxonomy)
                successes.append(result)
            except LoadError as exc:
                errors.append(exc)

        return (successes, errors)
