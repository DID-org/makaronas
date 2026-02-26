"""Task registry — indexes validated cartridges for efficient querying.

Loads all cartridges from disk via ``TaskLoader``, builds dict-based indexes
for O(1) single-criterion lookups, and supports multi-criteria filtered queries.
``reload()`` atomically swaps the index so in-flight readers see a consistent
snapshot (Framework P17: hot-swappable content, P21: live session integrity).

Tier 2 module: imports from ``backend.tasks.loader`` (Tier 2) and
``backend.tasks.schemas`` (Tier 1).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

from backend.tasks.loader import LoadError, LoadWarning, TaskLoader
from backend.tasks.schemas import TaskCartridge

logger = logging.getLogger("makaronas.tasks.registry")


class TaskRegistry:
    """Indexes validated task cartridges for efficient querying.

    Args:
        content_dir: Path to the content directory (contains tasks/).
        taxonomy_path: Path to taxonomy.json (typically content_dir / "taxonomy.json").
    """

    def __init__(self, content_dir: Path, taxonomy_path: Path) -> None:
        self._content_dir = content_dir
        self._taxonomy_path = taxonomy_path
        self._loader = TaskLoader()

        # Indexes — populated by load()/reload()
        self._by_id: dict[str, TaskCartridge] = {}
        self._by_status: dict[str, set[str]] = {}
        self._by_trigger: dict[str, set[str]] = defaultdict(set)
        self._by_technique: dict[str, set[str]] = defaultdict(set)
        self._by_medium: dict[str, set[str]] = defaultdict(set)
        self._by_tag: dict[str, set[str]] = defaultdict(set)

        # Diagnostics — preserved from the most recent load/reload
        self._load_errors: list[LoadError] = []
        self._load_warnings: dict[str, list[LoadWarning]] = {}

    def load(self) -> None:
        """Scans content/tasks/, validates, and builds the index.

        Called once at startup. Logs errors for tasks that fail loading.
        Does not raise — a registry with zero valid tasks is still a
        valid (empty) registry. If taxonomy.json is missing or the
        content directory doesn't exist, logs the error and starts empty.
        """
        try:
            by_id, by_status, by_trigger, by_technique, by_medium, by_tag, errors, warnings = (
                self._build_indexes()
            )
        except Exception:
            logger.exception("Registry load failed — starting with empty index")
            return

        self._by_id = by_id
        self._by_status = by_status
        self._by_trigger = by_trigger
        self._by_technique = by_technique
        self._by_medium = by_medium
        self._by_tag = by_tag
        self._load_errors = errors
        self._load_warnings = warnings

        total = len(by_id)
        err_count = len(errors)
        logger.info(
            "Registry loaded: %d task(s), %d error(s)", total, err_count,
        )
        for err in errors:
            logger.error(
                "Load error [%s] in %s: %s",
                err.error_type, err.task_dir, err.message,
            )

    def reload(self) -> None:
        """Atomic reload: re-scan, re-validate, swap index.

        Builds a completely new index. On success (even partial — some
        cartridges valid, some errored), atomically replaces the old
        index. Only if the rebuild fails entirely (e.g., content_dir
        disappeared, unhandled exception), logs an error and keeps
        the old index — in-flight requests are never disrupted.
        """
        try:
            by_id, by_status, by_trigger, by_technique, by_medium, by_tag, errors, warnings = (
                self._build_indexes()
            )
        except Exception:
            logger.exception("Registry reload failed — keeping old index")
            return

        self._by_id = by_id
        self._by_status = by_status
        self._by_trigger = by_trigger
        self._by_technique = by_technique
        self._by_medium = by_medium
        self._by_tag = by_tag
        self._load_errors = errors
        self._load_warnings = warnings

        total = len(by_id)
        err_count = len(errors)
        logger.info(
            "Registry reloaded: %d task(s), %d error(s)", total, err_count,
        )
        for err in errors:
            logger.error(
                "Load error [%s] in %s: %s",
                err.error_type, err.task_dir, err.message,
            )

    def get_task(self, task_id: str) -> TaskCartridge | None:
        """Returns a single cartridge by ID, or None if not found.

        Searches all status partitions (active, deprecated, draft).
        """
        return self._by_id.get(task_id)

    def query(
        self,
        *,
        trigger: str | None = None,
        technique: str | None = None,
        medium: str | None = None,
        difficulty_min: int | None = None,
        difficulty_max: int | None = None,
        tags: list[str] | None = None,
        status: str = "active",
        limit: int = 50,
        offset: int = 0,
    ) -> list[TaskCartridge]:
        """Multi-criteria filtered query returning cartridges.

        All filter parameters are AND-combined. Tags use AND logic
        (all specified tags must be present). Status defaults to
        "active" — pass "all" to search across all statuses.

        Returns a list of cartridges sorted by task_id (stable ordering).
        Respects limit/offset for pagination.
        """
        # Start with the status partition
        if status == "all":
            candidates = set(self._by_id.keys())
        else:
            candidates = set(self._by_status.get(status, set()))

        # Intersect with each filter index
        if trigger is not None:
            candidates &= self._by_trigger.get(trigger, set())
        if technique is not None:
            candidates &= self._by_technique.get(technique, set())
        if medium is not None:
            candidates &= self._by_medium.get(medium, set())
        if tags:
            for tag in tags:
                candidates &= self._by_tag.get(tag, set())

        # Sort for deterministic output, then apply difficulty filter + pagination
        sorted_ids = sorted(candidates)

        # Difficulty range filter — applied over the sorted list
        if difficulty_min is not None or difficulty_max is not None:
            filtered: list[str] = []
            for tid in sorted_ids:
                d = self._by_id[tid].difficulty
                if difficulty_min is not None and d < difficulty_min:
                    continue
                if difficulty_max is not None and d > difficulty_max:
                    continue
                filtered.append(tid)
            sorted_ids = filtered

        # Pagination
        page = sorted_ids[offset:offset + limit]
        return [self._by_id[tid] for tid in page]

    def get_all_task_ids(self, status: str = "active") -> list[str]:
        """Returns all task IDs for the given status partition."""
        if status == "all":
            return sorted(self._by_id.keys())
        return sorted(self._by_status.get(status, set()))

    def count(self, status: str = "active") -> int:
        """Returns the count of tasks in the given status partition."""
        if status == "all":
            return len(self._by_id)
        return len(self._by_status.get(status, set()))

    def is_phase_valid(self, task_id: str, phase_id: str) -> bool:
        """Checks whether a phase ID exists in the current version of a task.

        Used by student endpoints for stale phase detection after reload.
        Returns False if the task doesn't exist OR the phase doesn't exist.
        """
        cartridge = self._by_id.get(task_id)
        if cartridge is None:
            return False
        return any(p.id == phase_id for p in cartridge.phases)

    @property
    def load_errors(self) -> list[LoadError]:
        """Returns errors from the most recent load/reload."""
        return list(self._load_errors)

    @property
    def load_warnings(self) -> dict[str, list[LoadWarning]]:
        """Returns warnings per task_id from the most recent load/reload.

        Keys are task_ids. Values are the LoadWarning lists from each
        LoadResult. Only includes tasks that had warnings.
        """
        return dict(self._load_warnings)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_indexes(
        self,
    ) -> tuple[
        dict[str, TaskCartridge],
        dict[str, set[str]],
        dict[str, set[str]],
        dict[str, set[str]],
        dict[str, set[str]],
        dict[str, set[str]],
        list[LoadError],
        dict[str, list[LoadWarning]],
    ]:
        """Loads all tasks and builds fresh index dicts.

        Returns:
            Tuple of (by_id, by_status, by_trigger, by_technique, by_medium,
            by_tag, errors, warnings).
        """
        taxonomy = self._loader.load_taxonomy(self._taxonomy_path)
        successes, errors = self._loader.load_all_tasks(
            self._content_dir, taxonomy,
        )

        by_id: dict[str, TaskCartridge] = {}
        by_status: dict[str, set[str]] = defaultdict(set)
        by_trigger: dict[str, set[str]] = defaultdict(set)
        by_technique: dict[str, set[str]] = defaultdict(set)
        by_medium: dict[str, set[str]] = defaultdict(set)
        by_tag: dict[str, set[str]] = defaultdict(set)
        warn_map: dict[str, list[LoadWarning]] = {}

        for result in successes:
            c = result.cartridge
            tid = c.task_id

            by_id[tid] = c
            by_status[c.status].add(tid)
            by_trigger[c.trigger].add(tid)
            by_technique[c.technique].add(tid)
            by_medium[c.medium].add(tid)
            for tag in c.tags:
                by_tag[tag].add(tid)

            if result.warnings:
                warn_map[tid] = list(result.warnings)

        return by_id, by_status, by_trigger, by_technique, by_medium, by_tag, errors, warn_map
