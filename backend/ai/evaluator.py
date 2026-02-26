"""AI evaluator — student response assessment against EvaluationContract.

Uses provider.complete() (non-streaming) for rubric analysis. Given a
student's exchanges and the cartridge's EvaluationContract, produces a
structured assessment of what the student recognized and how well.

V6 implements the full Evaluator; V3 defines the interface shape so
the provider infrastructure can accommodate it without changes.

Tier 2 service: imports from providers/base (T1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.ai.providers.base import AIProvider


@dataclass
class EvaluationResult:
    """Structured output from an AI evaluation pass.

    V6 will define the full shape. This placeholder captures the
    minimum fields the platform needs to render evaluation feedback.
    """

    outcome: str  # "trickster_wins" | "partial" | "trickster_loses"
    summary: str  # Human-readable evaluation summary (Lithuanian)
    details: dict[str, Any]  # Pattern-level breakdown, checklist results


class Evaluator:
    """Student response assessment against EvaluationContract.

    Consumes the same AIProvider abstraction as the Trickster, but uses
    provider.complete() (non-streaming) because rubric analysis needs
    the full response before structured parsing.

    V6 implements; V3 defines the interface shape.

    Args:
        provider: The AI provider instance for generating evaluations.
    """

    def __init__(self, provider: AIProvider) -> None:
        self._provider = provider

    async def evaluate(
        self,
        *,
        exchanges: list[dict[str, str]],
        evaluation_contract: dict[str, Any],
        task_id: str,
    ) -> EvaluationResult:
        """Evaluates student responses against the cartridge's rubric.

        Analyses the full exchange history using the EvaluationContract's
        patterns_embedded, checklist, and pass_conditions to produce a
        structured assessment.

        Args:
            exchanges: The student-trickster exchange history.
            evaluation_contract: The cartridge's EvaluationContract as dict.
            task_id: The cartridge task_id for prompt loading.

        Returns:
            Structured evaluation result with outcome, summary, and details.

        Raises:
            NotImplementedError: V6 implements this method.
        """
        raise NotImplementedError("Evaluator.evaluate() is a V6 deliverable.")
