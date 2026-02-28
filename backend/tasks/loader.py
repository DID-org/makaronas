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

from backend.tasks.schemas import (
    AudioBlock,
    ButtonInteraction,
    ChatMessageBlock,
    ImageBlock,
    InvestigationInteraction,
    MemeBlock,
    Phase,
    SearchResultBlock,
    SocialPostBlock,
    TaskCartridge,
    TaxonomyWarning,
    TextBlock,
)

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
            ``"validation_error"``, ``"path_mismatch"``, ``"invalid_task_id"``,
            ``"path_traversal"``.
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
# Business validation — prompt injection patterns
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"<<SYS>>|<</SYS>>", re.IGNORECASE), "system prompt marker (<<SYS>>)"),
    (re.compile(r"\[INST\]|\[/INST\]", re.IGNORECASE), "instruction marker ([INST])"),
    (re.compile(r"\bYou are now\b", re.IGNORECASE), "role-switching (You are now)"),
    (re.compile(r"\bIgnore previous instructions\b", re.IGNORECASE), "instruction override"),
    (re.compile(r"\bSystem:\s", re.IGNORECASE), "role marker (System:)"),
    (re.compile(r"\bAssistant:\s", re.IGNORECASE), "role marker (Assistant:)"),
    (re.compile(r"\bDAN\b"), "DAN jailbreak pattern"),
]

# Warning types that trigger status demotion from active to draft.
_DEMOTION_WARNING_TYPES = frozenset({
    "orphan_phase",
    "no_terminal",
    "unbounded_cycle",
    "missing_asset",
    "type_mismatch",
    "evergreen_violation",
    "dangling_reference",
})


# ---------------------------------------------------------------------------
# Business validation — helpers
# ---------------------------------------------------------------------------


def _extract_transition_targets(phase: Phase) -> list[str]:
    """Extracts all transition target phase IDs from a phase.

    Checks button choices, investigation submit_target, and AI transitions.
    Generic/unknown interaction types are skipped — their transition schema
    is unknown (plan §7.2).
    """
    targets: list[str] = []

    if isinstance(phase.interaction, ButtonInteraction):
        for choice in phase.interaction.choices:
            targets.append(choice.target_phase)
    elif isinstance(phase.interaction, InvestigationInteraction):
        targets.append(phase.interaction.submit_target)

    if phase.ai_transitions is not None:
        targets.append(phase.ai_transitions.on_success)
        targets.append(phase.ai_transitions.on_max_exchanges)
        targets.append(phase.ai_transitions.on_partial)

    return targets


def _validate_graph(
    cartridge: TaskCartridge,
) -> tuple[list[LoadWarning], bool]:
    """Validates phase graph integrity: reachability, terminal, bounded cycles.

    Returns:
        Tuple of (warnings, should_demote).
    """
    warn: list[LoadWarning] = []
    demote = False
    task_id = cartridge.task_id
    phases = cartridge.phases

    # Empty phases list — initial_phase is dangling, no terminal
    if not phases:
        warn.append(LoadWarning(
            task_id=task_id,
            warning_type="dangling_reference",
            message=(
                f"initial_phase '{cartridge.initial_phase}' "
                f"references a nonexistent phase"
            ),
        ))
        warn.append(LoadWarning(
            task_id=task_id,
            warning_type="no_terminal",
            message="No terminal phase found in the phase graph",
        ))
        return warn, True

    # Build phase lookup and adjacency list
    phase_by_id = {p.id: p for p in phases}
    adjacency: dict[str, set[str]] = {p.id: set() for p in phases}

    # Collect edges, check for dangling references
    for phase in phases:
        for target in _extract_transition_targets(phase):
            if target in phase_by_id:
                adjacency[phase.id].add(target)
            else:
                warn.append(LoadWarning(
                    task_id=task_id,
                    warning_type="dangling_reference",
                    message=(
                        f"Phase '{phase.id}' references nonexistent "
                        f"target phase '{target}'"
                    ),
                ))
                demote = True

    # Check initial_phase exists
    if cartridge.initial_phase not in phase_by_id:
        warn.append(LoadWarning(
            task_id=task_id,
            warning_type="dangling_reference",
            message=(
                f"initial_phase '{cartridge.initial_phase}' "
                f"references a nonexistent phase"
            ),
        ))
        for phase in phases:
            warn.append(LoadWarning(
                task_id=task_id,
                warning_type="orphan_phase",
                message=f"Phase '{phase.id}' is unreachable from initial_phase",
            ))
        return warn, True

    # BFS from initial_phase
    visited: set[str] = {cartridge.initial_phase}
    queue = [cartridge.initial_phase]
    while queue:
        current = queue.pop(0)
        for neighbor in adjacency.get(current, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    # Orphan detection
    for phase in phases:
        if phase.id not in visited:
            warn.append(LoadWarning(
                task_id=task_id,
                warning_type="orphan_phase",
                message=f"Phase '{phase.id}' is unreachable from initial_phase",
            ))
            demote = True

    # Terminal phase check
    terminal_ids = {p.id for p in phases if p.is_terminal}
    if not terminal_ids:
        warn.append(LoadWarning(
            task_id=task_id,
            warning_type="no_terminal",
            message="No terminal phase found in the phase graph",
        ))
        demote = True

    # Bounded cycle check — every reachable non-terminal must reach a terminal.
    # Only meaningful when terminal phases exist.
    if terminal_ids:
        # Reverse BFS from terminals to find which phases can reach them
        reverse_adj: dict[str, set[str]] = {p.id: set() for p in phases}
        for src, targets in adjacency.items():
            for tgt in targets:
                reverse_adj[tgt].add(src)

        can_reach_terminal: set[str] = set(terminal_ids)
        queue = list(terminal_ids)
        while queue:
            current = queue.pop(0)
            for predecessor in reverse_adj.get(current, set()):
                if predecessor not in can_reach_terminal:
                    can_reach_terminal.add(predecessor)
                    queue.append(predecessor)

        for phase in phases:
            if (
                phase.id in visited
                and not phase.is_terminal
                and phase.id not in can_reach_terminal
            ):
                warn.append(LoadWarning(
                    task_id=task_id,
                    warning_type="unbounded_cycle",
                    message=(
                        f"Phase '{phase.id}' cannot reach any terminal phase "
                        f"— students may be trapped in an infinite loop"
                    ),
                ))
                demote = True

    return warn, demote


def _validate_assets(
    cartridge: TaskCartridge,
    task_dir: Path,
) -> tuple[list[LoadWarning], bool]:
    """Validates asset path safety and file existence.

    Raises LoadError for path traversal attempts (Framework P13).

    Returns:
        Tuple of (warnings, should_demote).
    """
    warn: list[LoadWarning] = []
    demote = False
    assets_dir = task_dir / "assets"
    dir_str = str(task_dir)

    for block in cartridge.presentation_blocks:
        # Extract (block_id, asset_path) pairs from known block types
        asset_refs: list[tuple[str, str]] = []
        if isinstance(block, ImageBlock):
            asset_refs.append((block.id, block.src))
        elif isinstance(block, AudioBlock):
            asset_refs.append((block.id, block.src))
        elif isinstance(block, MemeBlock):
            asset_refs.append((block.id, block.image_src))

        for block_id, asset_path in asset_refs:
            # Path safety — hard errors (Framework P13)
            if ".." in asset_path:
                raise LoadError(
                    task_dir=dir_str,
                    error_type="path_traversal",
                    message=(
                        f"Asset path '{asset_path}' in block '{block_id}' "
                        f"contains '..' — possible directory traversal"
                    ),
                )
            if asset_path.startswith("/"):
                raise LoadError(
                    task_dir=dir_str,
                    error_type="path_traversal",
                    message=(
                        f"Asset path '{asset_path}' in block '{block_id}' "
                        f"is absolute — must be relative to assets directory"
                    ),
                )

            # Defense in depth: resolve and check containment
            resolved = (assets_dir / asset_path).resolve()
            if not resolved.is_relative_to(assets_dir.resolve()):
                raise LoadError(
                    task_dir=dir_str,
                    error_type="path_traversal",
                    message=(
                        f"Asset path '{asset_path}' in block '{block_id}' "
                        f"resolves outside the assets directory"
                    ),
                )

            # Existence check — demotion warning
            if not resolved.exists():
                warn.append(LoadWarning(
                    task_id=cartridge.task_id,
                    warning_type="missing_asset",
                    message=(
                        f"Asset '{asset_path}' referenced by block "
                        f"'{block_id}' not found at {resolved}"
                    ),
                ))
                demote = True

    return warn, demote


def _validate_prompt_dir(
    cartridge: TaskCartridge,
    project_root: Path,
) -> list[LoadWarning]:
    """Checks prompt directory existence for AI/hybrid tasks.

    Non-demotion warning — prompt files are authored separately and their
    absence during early development is expected.
    """
    if cartridge.task_type not in ("ai_driven", "hybrid"):
        return []
    if cartridge.ai_config is None:
        return []  # type_mismatch catches this separately

    prompt_path = project_root / cartridge.ai_config.prompt_directory
    if not prompt_path.is_dir():
        return [LoadWarning(
            task_id=cartridge.task_id,
            warning_type="missing_prompt_dir",
            message=(
                f"Prompt directory '{cartridge.ai_config.prompt_directory}' "
                f"not found at {prompt_path}"
            ),
        )]
    return []


def _validate_type_completeness(
    cartridge: TaskCartridge,
) -> tuple[list[LoadWarning], bool]:
    """Checks that task_type matches phase composition and AI configuration."""
    warn: list[LoadWarning] = []
    demote = False
    task_id = cartridge.task_id

    ai_phases = [p for p in cartridge.phases if p.is_ai_phase]
    static_phases = [p for p in cartridge.phases if not p.is_ai_phase]

    if cartridge.task_type == "hybrid":
        if not ai_phases:
            warn.append(LoadWarning(
                task_id=task_id,
                warning_type="type_mismatch",
                message="Hybrid task has no AI phases — needs at least one",
            ))
            demote = True
        if not static_phases:
            warn.append(LoadWarning(
                task_id=task_id,
                warning_type="type_mismatch",
                message="Hybrid task has no static phases — needs at least one",
            ))
            demote = True
    elif cartridge.task_type == "ai_driven":
        if not ai_phases:
            warn.append(LoadWarning(
                task_id=task_id,
                warning_type="type_mismatch",
                message="AI-driven task has no AI phases — needs at least one",
            ))
            demote = True
    elif cartridge.task_type == "static":
        if ai_phases:
            warn.append(LoadWarning(
                task_id=task_id,
                warning_type="type_mismatch",
                message=(
                    f"Static task has {len(ai_phases)} AI phase(s) — "
                    f"static tasks must not have AI phases"
                ),
            ))
            demote = True

    # ai_config required for ai_driven and hybrid
    if cartridge.task_type in ("ai_driven", "hybrid") and cartridge.ai_config is None:
        warn.append(LoadWarning(
            task_id=task_id,
            warning_type="type_mismatch",
            message=f"{cartridge.task_type} task is missing ai_config",
        ))
        demote = True

    return warn, demote


def _validate_evergreen(
    cartridge: TaskCartridge,
) -> tuple[list[LoadWarning], bool]:
    """Checks evergreen flag — active tasks must have is_evergreen=True."""
    if cartridge.status == "active" and not cartridge.is_evergreen:
        return ([LoadWarning(
            task_id=cartridge.task_id,
            warning_type="evergreen_violation",
            message="Active task must have is_evergreen=True",
        )], True)
    return ([], False)


def _scan_prompt_injection(
    cartridge: TaskCartridge,
) -> list[LoadWarning]:
    """Scans text content for common prompt injection patterns.

    Advisory only — warns content authors about accidental injection vectors.
    Does not trigger demotion (plan §4.5).
    """
    warn: list[LoadWarning] = []
    task_id = cartridge.task_id

    # Collect (field_description, text) pairs from presentation blocks
    texts: list[tuple[str, str]] = []
    for block in cartridge.presentation_blocks:
        if isinstance(block, TextBlock):
            texts.append((f"block '{block.id}' text", block.text))
        elif isinstance(block, ChatMessageBlock):
            texts.append((f"block '{block.id}' text", block.text))
        elif isinstance(block, SocialPostBlock):
            texts.append((f"block '{block.id}' text", block.text))
        elif isinstance(block, SearchResultBlock):
            texts.append((f"block '{block.id}' title", block.title))
            texts.append((f"block '{block.id}' snippet", block.snippet))
        elif isinstance(block, MemeBlock):
            if block.top_text:
                texts.append((f"block '{block.id}' top_text", block.top_text))
            if block.bottom_text:
                texts.append((f"block '{block.id}' bottom_text", block.bottom_text))

    # Scan phase trickster_content fields
    for phase in cartridge.phases:
        if phase.trickster_content is not None:
            texts.append((
                f"phase '{phase.id}' trickster_content",
                phase.trickster_content,
            ))

    # Check all collected text against injection patterns
    for field_desc, text in texts:
        for pattern, desc in _INJECTION_PATTERNS:
            if pattern.search(text):
                warn.append(LoadWarning(
                    task_id=task_id,
                    warning_type="prompt_injection_suspect",
                    message=f"Possible prompt injection in {field_desc}: {desc}",
                ))

    return warn


# ---------------------------------------------------------------------------
# Business validation — public entry point
# ---------------------------------------------------------------------------


def validate_business_rules(
    cartridge: TaskCartridge,
    task_dir: Path,
    project_root: Path,
) -> tuple[TaskCartridge, list[LoadWarning]]:
    """Runs business validation on a schema-valid cartridge.

    Checks phase graph integrity, asset existence, prompt directory,
    type-specific completeness, evergreen flag, and prompt injection patterns.
    Tasks that fail business rules are demoted to draft status.

    Args:
        cartridge: The frozen, schema-valid TaskCartridge.
        task_dir: The task's content directory (for asset checks).
        project_root: The project root (for prompt directory checks).

    Returns:
        Tuple of (cartridge, warnings). If any business rule triggered
        demotion, the returned cartridge has status="draft".

    Raises:
        LoadError: With error_type="path_traversal" for malicious asset
            paths. This is a security violation, not a content quality issue.
    """
    biz_warnings: list[LoadWarning] = []
    should_demote = False

    # 1. Phase graph integrity
    graph_warn, graph_demote = _validate_graph(cartridge)
    biz_warnings.extend(graph_warn)
    should_demote = should_demote or graph_demote

    # 2. Asset existence (may raise LoadError for path traversal)
    asset_warn, asset_demote = _validate_assets(cartridge, task_dir)
    biz_warnings.extend(asset_warn)
    should_demote = should_demote or asset_demote

    # 3. Prompt directory (non-demotion warning)
    biz_warnings.extend(_validate_prompt_dir(cartridge, project_root))

    # 4. Type-specific completeness
    type_warn, type_demote = _validate_type_completeness(cartridge)
    biz_warnings.extend(type_warn)
    should_demote = should_demote or type_demote

    # 5. Evergreen flag
    eg_warn, eg_demote = _validate_evergreen(cartridge)
    biz_warnings.extend(eg_warn)
    should_demote = should_demote or eg_demote

    # 6. Prompt injection scanning (non-demotion warning)
    biz_warnings.extend(_scan_prompt_injection(cartridge))

    # 7. Status demotion — only demote active tasks
    if should_demote and cartridge.status == "active":
        cartridge = cartridge.model_copy(update={"status": "draft"})
        reasons = sorted({
            w.warning_type for w in biz_warnings
            if w.warning_type in _DEMOTION_WARNING_TYPES
        })
        biz_warnings.append(LoadWarning(
            task_id=cartridge.task_id,
            warning_type="status_demoted",
            message=(
                f"Task '{cartridge.task_id}' demoted to draft due to: "
                + ", ".join(reasons)
            ),
        ))

    return cartridge, biz_warnings


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

    def load_task(
        self, task_dir: Path, taxonomy: dict, content_dir: Path,
    ) -> LoadResult:
        """Loads and validates a single task cartridge from disk.

        Sequence:
        1. Read ``task_dir / "task.json"`` and parse JSON.
        2. If ``task_id`` is present in parsed data — validate characters and
           Path=Identity (directory name must match).
        3. Call ``TaskCartridge.model_validate()`` with taxonomy context.
        4. Collect ``TaxonomyWarning`` emissions as ``LoadWarning`` objects.
        5. Run business validation (graph, assets, type, evergreen, injection).

        Args:
            task_dir: Directory containing ``task.json``.
            taxonomy: Parsed taxonomy dict for context injection.
            content_dir: The content root directory (e.g. ``content/``).
                Used to derive ``project_root`` for business validation
                (Framework P17 — no .parent chain derivation).

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

        # Step 5: Business validation (graph, assets, type, evergreen, injection)
        project_root = content_dir.parent
        cartridge, biz_warnings = validate_business_rules(
            cartridge, task_dir, project_root,
        )
        load_warnings.extend(biz_warnings)

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
                result = self.load_task(child, taxonomy, content_dir)
                successes.append(result)
            except LoadError as exc:
                errors.append(exc)

        return (successes, errors)
