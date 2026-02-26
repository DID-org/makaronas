"""Composer AI persona — teacher assistant for curriculum design.

Provides streaming AI assistance for teachers: task recommendations,
curriculum planning, and pedagogical explanations. Uses the same
provider + prompt + context infrastructure as the Trickster.

V8 implements the full Composer; V3 defines the interface shape so
the provider and prompt infrastructure can accommodate it without
changes.

Tier 2 service: imports from providers/base (T1).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from backend.ai.providers.base import AIProvider


class Composer:
    """Teacher assistant AI — curriculum design and task recommendations.

    Consumes the same AIProvider abstraction as the Trickster, with
    Composer-specific prompts (authored in V8, loaded via PromptLoader).
    Streaming interface for real-time teacher interaction.

    V8 implements; V3 defines the interface shape.

    Args:
        provider: The AI provider instance for generating responses.
    """

    def __init__(self, provider: AIProvider) -> None:
        self._provider = provider

    async def suggest(
        self,
        *,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """Suggests tasks or curriculum sequences based on teacher query.

        Streams recommendations token-by-token for responsive UI.

        Args:
            query: The teacher's curriculum question or request.
            context: Optional context (class profile, completed tasks, etc.).

        Yields:
            Text tokens of the recommendation response.

        Raises:
            NotImplementedError: V8 implements this method.
        """
        raise NotImplementedError("Composer.suggest() is a V8 deliverable.")
        # Make this an async generator so the return type is valid
        yield  # pragma: no cover

    async def explain(
        self,
        *,
        task_id: str,
        aspect: str = "overview",
    ) -> AsyncIterator[str]:
        """Explains a task's pedagogical design or manipulation techniques.

        Helps teachers understand what a task teaches and why, supporting
        informed curriculum decisions.

        Args:
            task_id: The cartridge task_id to explain.
            aspect: What to explain ("overview", "techniques", "difficulty").

        Yields:
            Text tokens of the explanation response.

        Raises:
            NotImplementedError: V8 implements this method.
        """
        raise NotImplementedError("Composer.explain() is a V8 deliverable.")
        yield  # pragma: no cover
