"""Prompt loading from disk with model-specific fallback chain and caching.

Loads Trickster prompt files from the prompts/ directory tree. Each prompt
type (persona, behaviour, safety) has a base version and optional
model-specific overrides. The loader tries the model-specific file first,
falls back to base, and caches the result keyed by (provider, task_id).

Consumed by:
- ContextManager (Phase 4a) — assembles system prompts from loaded layers
- Startup checks (Phase 6a) — validates all AI tasks have required prompts
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from backend.tasks.schemas import TaskCartridge

logger = logging.getLogger(__name__)

# Provider name → file suffix mapping.
# Unknown providers fall back to base files only.
_PROVIDER_SUFFIX: dict[str, str] = {
    "anthropic": "claude",
    "gemini": "gemini",
}

# Prompt types that live in prompts/trickster/ (layers 1-3).
_BASE_PROMPT_TYPES = ("persona", "behaviour", "safety")


@dataclass(frozen=True)
class TricksterPrompts:
    """Loaded Trickster prompt layers for a single (provider, task_id) combo.

    Each field is the raw Markdown content of the corresponding prompt file,
    or None if the file doesn't exist (or is empty/whitespace-only).
    """

    persona: str | None
    behaviour: str | None
    safety: str | None
    task_override: str | None


class PromptLoader:
    """Loads and caches Trickster prompt files from disk.

    Args:
        prompts_dir: Base prompts directory (e.g. PROJECT_ROOT / "prompts").
    """

    def __init__(self, prompts_dir: Path) -> None:
        self._prompts_dir = prompts_dir
        self._cache: dict[tuple[str, str | None], TricksterPrompts] = {}

    def load_trickster_prompts(
        self, provider: str, task_id: str | None = None
    ) -> TricksterPrompts:
        """Loads Trickster prompt layers with model-specific fallback.

        For each prompt type, tries the model-specific variant first
        (e.g. persona_gemini.md), then falls back to base (persona_base.md).
        Empty or whitespace-only files are treated as absent.

        Args:
            provider: Provider name (e.g. "gemini", "anthropic").
            task_id: Optional task ID for loading task-specific overrides.

        Returns:
            TricksterPrompts with loaded content (or None per field).
        """
        cache_key = (provider, task_id)
        if cache_key in self._cache:
            logger.debug("Cache hit for prompts: provider=%s task_id=%s", provider, task_id)
            return self._cache[cache_key]

        logger.debug("Cache miss for prompts: provider=%s task_id=%s", provider, task_id)
        suffix = _PROVIDER_SUFFIX.get(provider)
        trickster_dir = self._prompts_dir / "trickster"

        persona = self._load_with_fallback(trickster_dir, "persona", suffix)
        behaviour = self._load_with_fallback(trickster_dir, "behaviour", suffix)
        safety = self._load_with_fallback(trickster_dir, "safety", suffix)

        task_override: str | None = None
        if task_id is not None:
            task_dir = self._prompts_dir / "tasks" / task_id
            task_override = self._load_with_fallback(task_dir, "trickster", suffix)

        result = TricksterPrompts(
            persona=persona,
            behaviour=behaviour,
            safety=safety,
            task_override=task_override,
        )
        self._cache[cache_key] = result
        return result

    def validate_task_prompts(self, cartridge: TaskCartridge) -> list[str]:
        """Validates that required base prompt files exist for AI-phase tasks.

        Checks persona_base.md, behaviour_base.md, and safety_base.md under
        prompts/trickster/. Only validates tasks that have AI phases
        (task_type is ai_driven or hybrid, ai_config is present, and at least
        one phase has is_ai_phase=True).

        Args:
            cartridge: The task cartridge to validate.

        Returns:
            List of error strings (empty means valid).
        """
        if cartridge.task_type not in ("ai_driven", "hybrid"):
            return []
        if cartridge.ai_config is None:
            return []
        if not any(phase.is_ai_phase for phase in cartridge.phases):
            return []

        errors: list[str] = []
        trickster_dir = self._prompts_dir / "trickster"

        for prompt_type in _BASE_PROMPT_TYPES:
            filename = f"{prompt_type}_base.md"
            filepath = trickster_dir / filename
            if not filepath.exists():
                errors.append(
                    f"Task '{cartridge.task_id}': missing required prompt file "
                    f"prompts/trickster/{filename}"
                )
            elif not filepath.read_text(encoding="utf-8").strip():
                errors.append(
                    f"Task '{cartridge.task_id}': prompt file "
                    f"prompts/trickster/{filename} is empty"
                )

        return errors

    def invalidate(self) -> None:
        """Clears the in-memory prompt cache.

        Called on registry reload to force re-reading from disk.
        """
        logger.debug("Prompt cache invalidated (%d entries cleared)", len(self._cache))
        self._cache.clear()

    def _load_with_fallback(
        self, directory: Path, type_name: str, suffix: str | None
    ) -> str | None:
        """Loads a prompt file with model-specific fallback to base.

        Tries {type_name}_{suffix}.md first (if suffix is not None),
        then {type_name}_base.md. Returns None if neither exists or
        if the found file is empty/whitespace-only.

        Args:
            directory: Directory containing the prompt files.
            type_name: Prompt type (e.g. "persona", "behaviour", "trickster").
            suffix: Model-specific suffix (e.g. "gemini", "claude"), or None.

        Returns:
            File content as a string, or None.
        """
        if suffix is not None:
            content = self._read_prompt_file(directory / f"{type_name}_{suffix}.md")
            if content is not None:
                return content

        return self._read_prompt_file(directory / f"{type_name}_base.md")

    @staticmethod
    def _read_prompt_file(path: Path) -> str | None:
        """Reads a single prompt file, returning None if absent or empty.

        Args:
            path: Full path to the prompt file.

        Returns:
            Stripped file content, or None if file doesn't exist or is
            empty/whitespace-only.
        """
        try:
            content = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None

        stripped = content.strip()
        if not stripped:
            return None

        return stripped
