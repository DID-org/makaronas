"""Trickster AI persona \u2014 adversarial dialogue engine (V3).

Orchestrates the conversation loop between student and Trickster:
provider streaming, exchange accumulation, transition signal extraction,
safety checking, and malformed response retry.

This is the first component that wires all prior V3 modules together.
The engine is the ONLY code path that modifies session.exchanges.

Consumed by:
- Student endpoint (Phase 6b) \u2014 calls respond() and iterates token_iterator
- Phase 5b \u2014 adds debrief() method on top of this core

Tier 2 service: imports from providers/base (T1), context (T2), safety (T2),
schemas (T1), tasks/schemas (T1), models (T1).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from backend.ai import safety
from backend.ai.context import ContextManager
from backend.ai.providers.base import (
    AIProvider,
    TextChunk,
    ToolCallEvent,
    UsageInfo,
)
from backend.models import resolve_tier
from backend.schemas import Exchange, GameSession
from backend.tasks.schemas import FreeformInteraction, Phase, TaskCartridge

logger = logging.getLogger("makaronas.ai.trickster")

# Minimum response length before triggering malformed-response retry.
_MIN_RESPONSE_LENGTH = 10

# Signal name -> AiTransitions attribute name mapping.
_SIGNAL_MAP: dict[str, str] = {
    "understood": "on_success",
    "partial": "on_partial",
    "max_reached": "on_max_exchanges",
}


@dataclass
class TricksterResult:
    """Streaming response plus post-completion metadata.

    The consumer iterates token_iterator to yield SSE token events.
    After exhaustion, done_data/redaction_data/usage are populated
    by the engine's internal generator.

    Attributes:
        token_iterator: Async iterator of text tokens for SSE streaming.
        done_data: Set after iteration if response is safe. Contains
            phase_transition, next_phase, exchanges_count.
        redaction_data: Set after iteration if safety violation detected.
            Contains fallback_text, boundary.
        usage: Token usage from the provider (may be None for MockProvider).
    """

    token_iterator: AsyncIterator[str]
    done_data: dict[str, Any] | None = field(default=None, init=False)
    redaction_data: dict[str, Any] | None = field(default=None, init=False)
    usage: UsageInfo | None = field(default=None, init=False)


class TricksterEngine:
    """Orchestrates Trickster dialogue \u2014 streaming, safety, transitions.

    Wires the AI provider, context manager, and safety pipeline into a
    single conversational flow. The engine wraps a provider call and
    yields text tokens downstream while managing exchange accumulation,
    tool call interception, and post-completion safety checks.

    Args:
        provider: AI provider instance (MockProvider in tests,
            GeminiProvider or AnthropicProvider in production).
        context_manager: Context assembly service.
    """

    def __init__(
        self,
        provider: AIProvider,
        context_manager: ContextManager,
    ) -> None:
        self._provider = provider
        self._context_manager = context_manager

    async def respond(
        self,
        session: GameSession,
        cartridge: TaskCartridge,
        phase: Phase,
        student_input: str,
    ) -> TricksterResult:
        """Processes a student message and returns streaming Trickster response.

        Orchestrates: exchange accumulation, input validation, context
        assembly, provider streaming with tool call interception,
        post-completion safety check, and transition signal resolution.

        Args:
            session: Mutable game session (exchanges will be appended).
            cartridge: Task cartridge with AI config and safety settings.
            phase: Current phase (must have FreeformInteraction + AiTransitions).
            student_input: The student's message text.

        Returns:
            TricksterResult with token iterator and post-completion fields.

        Raises:
            ValueError: If phase lacks FreeformInteraction or AiTransitions.
        """
        # 1. Validate phase
        interaction = phase.interaction
        if not isinstance(interaction, FreeformInteraction):
            raise ValueError(
                f"Phase '{phase.id}' does not have a FreeformInteraction "
                f"(got {type(interaction).__name__})"
            )
        ai_transitions = phase.ai_transitions
        if ai_transitions is None:
            raise ValueError(
                f"Phase '{phase.id}' does not have ai_transitions"
            )

        # 2. Save student exchange (before AI call \u2014 never lose student input)
        session.exchanges.append(
            Exchange(role="student", content=student_input)
        )

        # 3. Input validation (warn-and-log only, never blocks)
        safety.validate_input(student_input, cartridge.task_id)

        # 4. Resolve model tier
        model_config = resolve_tier(cartridge.ai_config.model_preference)

        # 5. Assemble context
        exchange_count = sum(
            1 for e in session.exchanges if e.role == "student"
        )
        ctx = self._context_manager.assemble_trickster_call(
            session,
            cartridge,
            model_config.provider,
            exchange_count,
            interaction.min_exchanges,
        )

        logger.info(
            "Trickster respond: task=%s phase=%s exchange=%d/%d",
            cartridge.task_id,
            phase.id,
            exchange_count,
            interaction.max_exchanges,
        )

        # Build result with internal generator as token_iterator.
        # The generator populates done_data/redaction_data/usage on exhaustion.
        # Python closures are late-binding: 'result' is captured by reference.
        # When the generator body executes (during consumer iteration),
        # 'result' is already bound to the TricksterResult instance.
        provider = self._provider
        max_exchanges = interaction.max_exchanges

        async def _stream() -> AsyncIterator[str]:
            accumulated = ""
            transition_signal: str | None = None

            # 6-7. Call provider and stream + accumulate
            async for event in provider.stream(
                system_prompt=ctx.system_prompt,
                messages=ctx.messages,
                model_config=model_config,
                tools=ctx.tools,
            ):
                if isinstance(event, TextChunk):
                    accumulated += event.text
                    yield event.text
                elif isinstance(event, ToolCallEvent):
                    if event.function_name == "transition_phase":
                        sig = event.arguments.get("signal")
                        if sig in _SIGNAL_MAP:
                            transition_signal = sig
                        else:
                            logger.warning(
                                "Unknown transition signal: %s", sig,
                            )
                    else:
                        logger.warning(
                            "Unexpected tool call: %s",
                            event.function_name,
                        )

            # 8. Malformed response check \u2014 retry once if < 10 chars
            if (
                len(accumulated) < _MIN_RESPONSE_LENGTH
                and transition_signal is None
            ):
                logger.warning(
                    "Malformed response (<%d chars), retrying",
                    _MIN_RESPONSE_LENGTH,
                )
                retry_signal: str | None = None

                async for event in provider.stream(
                    system_prompt=ctx.system_prompt,
                    messages=ctx.messages,
                    model_config=model_config,
                    tools=ctx.tools,
                ):
                    if isinstance(event, TextChunk):
                        accumulated += event.text
                        yield event.text
                    elif isinstance(event, ToolCallEvent):
                        if event.function_name == "transition_phase":
                            sig = event.arguments.get("signal")
                            if sig in _SIGNAL_MAP:
                                retry_signal = sig

                if retry_signal is not None:
                    transition_signal = retry_signal

                if len(accumulated) < _MIN_RESPONSE_LENGTH:
                    logger.error(
                        "Both attempts produced malformed response "
                        "(<%d chars)",
                        _MIN_RESPONSE_LENGTH,
                    )
                    result.done_data = {
                        "error": "malformed_response",
                        "phase_transition": None,
                        "next_phase": None,
                        "exchanges_count": exchange_count,
                    }
                    result.usage = getattr(
                        provider, "_last_usage", None,
                    )
                    return

            # 9. Post-completion safety check
            safety_result = safety.check_output(
                accumulated, cartridge.safety, is_debrief=False,
            )

            # 10. Handle violation (takes priority over any transition)
            if not safety_result.is_safe:
                violation = safety_result.violation
                session.exchanges.append(
                    Exchange(
                        role="trickster",
                        content=violation.fallback_text,
                    )
                )
                session.last_redaction_reason = violation.boundary
                result.redaction_data = {
                    "fallback_text": violation.fallback_text,
                    "boundary": violation.boundary,
                }
                result.done_data = None
                logger.info(
                    "Safety violation: boundary=%s", violation.boundary,
                )
            else:
                # 11. Safe \u2014 store response and resolve transition
                session.exchanges.append(
                    Exchange(role="trickster", content=accumulated)
                )

                transition_name: str | None = None
                next_phase: str | None = None

                if transition_signal is not None:
                    attr_name = _SIGNAL_MAP[transition_signal]
                    transition_name = attr_name
                    next_phase = getattr(ai_transitions, attr_name)
                elif exchange_count >= max_exchanges:
                    transition_name = "on_max_exchanges"
                    next_phase = ai_transitions.on_max_exchanges

                result.done_data = {
                    "phase_transition": transition_name,
                    "next_phase": next_phase,
                    "exchanges_count": exchange_count,
                }
                result.redaction_data = None

                if transition_name is not None:
                    logger.info(
                        "Transition: %s -> %s",
                        transition_name,
                        next_phase,
                    )

            # 12. Capture usage
            result.usage = getattr(provider, "_last_usage", None)

        result = TricksterResult(token_iterator=_stream())
        return result
