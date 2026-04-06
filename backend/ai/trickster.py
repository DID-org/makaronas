"""Trickster AI persona \u2014 adversarial dialogue engine (V3).

Orchestrates the conversation loop between student and Trickster:
provider streaming, exchange accumulation, transition signal extraction,
safety checking, malformed response retry, debrief generation, and
prompt snapshotting for live session integrity.

This is the first component that wires all prior V3 modules together.
The engine is the ONLY code path that modifies session.exchanges.

Consumed by:
- Student endpoint (Phase 6b) \u2014 calls respond()/debrief() and iterates
  token_iterator

Tier 2 service: imports from providers/base (T1), context (T2), safety (T2),
schemas (T1), tasks/schemas (T1), models (T1).
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from backend.ai import safety
from backend.ai.context import ContextManager
from backend.ai.intensity import score_intensity
from backend.ai.safety import FALLBACK_INTENSITY
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

# Regex to extract transition signals leaked as text instead of tool calls.
# Some models (especially lighter ones) emit the tool call as JSON text
# instead of using the function calling mechanism. This catches common
# patterns like: { "action": "transition_phase", ... "signal": "understood" }
_TEXT_SIGNAL_RE = re.compile(
    r"""(?:transition_phase|publish_article).*?['"]?(understood|partial|max_reached)['"]?""",
    re.DOTALL,
)

# Regex to strip leaked tool call text from displayed output.
# Catches multiple formats models may emit:
#   { "action": "transition_phase", ... }  (including nested braces)
#   befp_transition_phase: {signal: 'understood'}
#   transition_phase({"signal": "understood"})
_LEAKED_TOOL_CALL_RE = re.compile(
    r'\s*(?:\{.*(?:transition_phase|publish_article).*\}|befp_(?:transition_phase|publish_article)\s*:.+|(?:transition_phase|publish_article)\s*\(.+\))\s*$',
    re.DOTALL,
)

# Regex to extract article_text from leaked publish_article JSON.
_TEXT_ARTICLE_RE = re.compile(
    r"""article_text['":\s]+['"](.+?)['"]""",
    re.DOTALL,
)

# Regex to extract response_text from leaked transition_phase JSON.
# Matches: "response_text": "..." (greedy — grab everything up to the last quote-brace)
_TEXT_RESPONSE_RE = re.compile(
    r"""response_text['":\s]+['"](.*?)['"]\s*[}'"]\s*[}'"]?\s*$""",
    re.DOTALL,
)

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


@dataclass
class DebriefResult:
    """Streaming debrief response plus post-completion metadata.

    Same consumer pattern as TricksterResult: iterate token_iterator
    to exhaustion, then read done_data/redaction_data/usage.

    Debrief has no phase transitions \u2014 done_data contains
    ``{"debrief_complete": True}`` on success.

    Attributes:
        token_iterator: Async iterator of text tokens for SSE streaming.
        done_data: Set after iteration if response is safe. Contains
            debrief_complete=True.
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
        intensity_indicators: Loaded indicator data for intensity scoring.
            None disables intensity enforcement (graceful degradation).
    """

    def __init__(
        self,
        provider: AIProvider,
        context_manager: ContextManager,
        intensity_indicators: dict | None = None,
    ) -> None:
        self._provider = provider
        self._context_manager = context_manager
        self._intensity_indicators = intensity_indicators

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

        # 2. Snapshot prompts on first AI call for this task (Principle 21)
        if session.prompt_snapshots is None:
            snap_config = resolve_tier(cartridge.ai_config.model_preference)
            prompts = self._context_manager._loader.load_trickster_prompts(
                snap_config.provider, cartridge.task_id,
                persona_mode=cartridge.ai_config.persona_mode,
            )
            fourth_wall = self._context_manager._loader.load_fourth_wall_prompt(
                snap_config.provider,
            )
            self._context_manager.snapshot_prompts(
                session, prompts, fourth_wall=fourth_wall,
            )

        # 3. Save student exchange (before AI call \u2014 never lose student input)
        session.exchanges.append(
            Exchange(role="student", content=student_input)
        )

        # 4. Input validation (warn-and-log only, never blocks)
        safety.validate_input(student_input, cartridge.task_id)

        # 5. Resolve model tier and provider
        model_config = resolve_tier(cartridge.ai_config.model_preference)
        # Use per-task provider if tier differs from default
        from backend.api.deps import create_provider
        from backend.config import get_settings
        provider = create_provider(model_config, get_settings())

        # 6. Assemble context
        exchange_count = sum(
            1 for e in session.exchanges if e.role == "student"
        )
        ctx = self._context_manager.assemble_trickster_call(
            session,
            cartridge,
            model_config.provider,
            exchange_count,
            interaction.min_exchanges,
            phase_id=phase.id,
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
        max_exchanges = interaction.max_exchanges

        async def _stream() -> AsyncIterator[str]:
            accumulated = ""
            transition_signal: str | None = None

            # 6. Stream Flash response — NO TOOLS
            # Flash handles conversation only. Phase transitions are decided
            # by the Flash Lite evaluator after the response completes.
            async for event in provider.stream(
                system_prompt=ctx.system_prompt,
                messages=ctx.messages,
                model_config=model_config,
                tools=None,
                force_tool=False,
            ):
                if isinstance(event, TextChunk):
                    accumulated += event.text
                    yield event.text

            # 7. Strip any leaked tool call text Flash may still produce
            # (from residual prompt instructions).
            m = _TEXT_SIGNAL_RE.search(accumulated)
            if m:
                accumulated = _LEAKED_TOOL_CALL_RE.sub('', accumulated).rstrip()
                logger.info("Stripped leaked tool call text from Flash response")

            # 8. Empty response recovery — nudge and retry
            if len(accumulated) < _MIN_RESPONSE_LENGTH:
                logger.warning(
                    "Empty response (<%d chars), nudging Flash to retry",
                    _MIN_RESPONSE_LENGTH,
                )
                nudge_messages = list(ctx.messages) + [
                    {"role": "assistant", "content": accumulated or ""},
                    {"role": "user", "content": (
                        "[SYSTEM: Your last response was empty \u2014 the student "
                        "saw nothing. Please respond to the student\u2019s last "
                        "message in Lithuanian.]"
                    )},
                ]
                async for event in provider.stream(
                    system_prompt=ctx.system_prompt,
                    messages=nudge_messages,
                    model_config=model_config,
                    tools=None,
                    force_tool=False,
                ):
                    if isinstance(event, TextChunk):
                        accumulated += event.text
                        yield event.text

            if len(accumulated) < _MIN_RESPONSE_LENGTH:
                fallback = "Hmm... pabandykite dar kart\u0105."
                accumulated = fallback
                yield fallback
                result.done_data = {
                    "error": "malformed_response",
                    "phase_transition": None,
                    "next_phase": None,
                    "exchanges_count": exchange_count,
                }
                result.usage = getattr(provider, "_last_usage", None)
                return

            # 9a. Phase transition evaluation — Flash Lite decides
            logger.info(
                "Evaluator gate: exchange_count=%d min_exchanges=%d phase=%s",
                exchange_count, interaction.min_exchanges, phase.id,
            )
            if exchange_count >= interaction.min_exchanges:
                from backend.ai.phase_evaluator import (
                    evaluate_exchange_with_tool,
                    format_checklist,
                )
                from backend.api.deps import create_provider as _create_provider
                from backend.models import ModelConfig as _MC, GEMINI_FLASH_LITE

                eval_model = _MC(provider="gemini", model_id=GEMINI_FLASH_LITE)
                from backend.config import get_settings as _get_settings
                eval_provider = _create_provider(eval_model, _get_settings())

                # Phase-level checklist takes priority over task-level
                checklist_items = []
                if phase.evaluator_checklist:
                    checklist_items = [
                        {
                            "id": item.id,
                            "description": item.description,
                            "is_mandatory": item.is_mandatory,
                        }
                        for item in phase.evaluator_checklist
                    ]
                elif cartridge.evaluation and cartridge.evaluation.checklist:
                    checklist_items = [
                        {
                            "id": item.id,
                            "description": item.description,
                            "is_mandatory": item.is_mandatory,
                        }
                        for item in cartridge.evaluation.checklist
                    ]
                checklist_text = format_checklist(checklist_items)

                eval_result = await evaluate_exchange_with_tool(
                    provider=eval_provider,
                    model_config=eval_model,
                    student_message=student_input,
                    assistant_response=accumulated,
                    checklist_text=checklist_text,
                )

                if eval_result.should_transition:
                    transition_signal = eval_result.signal or "understood"
                    logger.info(
                        "Evaluator: transition signal=%s items=%s",
                        transition_signal, eval_result.satisfied_items,
                    )
                    # Handle write_article phase — save student's article
                    if phase.id == "write_article":
                        student_msgs = [
                            e.content for e in session.exchanges
                            if e.role == "student"
                        ]
                        final_article = student_msgs[-1] if student_msgs else ""
                        if final_article:
                            session.generated_artifacts.append({
                                "type": "student_article",
                                "text": final_article,
                                "phase": phase.id,
                                "task_id": cartridge.task_id,
                            })
                            try:
                                from pathlib import Path
                                Path("/tmp/student_article.txt").write_text(
                                    final_article, encoding="utf-8"
                                )
                            except Exception:
                                pass
                            logger.info("Student article saved (%d chars)", len(final_article))
                else:
                    logger.debug("Evaluator: continue")

            # 9b. Force transition at max exchanges
            if transition_signal is None and exchange_count >= max_exchanges:
                transition_signal = "max_reached"
                logger.info("Max exchanges reached, forcing transition")

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
                # 10b. Intensity scoring (after safety, before storage)
                intensity_score: float | None = None
                intensity_deescalation = False

                if self._intensity_indicators is not None:
                    ceiling = cartridge.safety.intensity_ceiling
                    intensity_score = score_intensity(
                        accumulated,
                        exchange_count,
                        max_exchanges,
                        self._intensity_indicators,
                    )
                    session.turn_intensities.append(intensity_score)

                    if intensity_score > ceiling * 1.5:
                        # Hard redaction \u2014 intensity way too high
                        logger.error(
                            "Intensity hard redaction: task=%s exchange=%d "
                            "score=%.2f ceiling=%d",
                            cartridge.task_id,
                            exchange_count,
                            intensity_score,
                            ceiling,
                        )
                        session.exchanges.append(
                            Exchange(
                                role="trickster",
                                content=FALLBACK_INTENSITY,
                            )
                        )
                        session.last_redaction_reason = "intensity"
                        result.redaction_data = {
                            "fallback_text": FALLBACK_INTENSITY,
                            "boundary": "intensity",
                        }
                        result.done_data = None
                        result.usage = getattr(
                            provider, "_last_usage", None,
                        )
                        return
                    elif intensity_score > ceiling:
                        # De-escalation flag for Phase 4c
                        intensity_deescalation = True
                        logger.warning(
                            "Intensity exceeded ceiling: task=%s exchange=%d "
                            "score=%.2f ceiling=%d",
                            cartridge.task_id,
                            exchange_count,
                            intensity_score,
                            ceiling,
                        )

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

                if intensity_score is not None:
                    result.done_data["intensity_score"] = intensity_score
                    result.done_data["intensity_deescalation"] = (
                        intensity_deescalation
                    )

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

    async def debrief(
        self,
        session: GameSession,
        cartridge: TaskCartridge,
    ) -> DebriefResult:
        """Generates the Trickster's debrief (honest reveal) for the student.

        Assembles debrief-specific context (EvaluationContract data + full
        exchange history), streams through the provider, and runs safety
        with pedagogical exemption (is_debrief=True).

        No phase parameter \u2014 debrief is task-level. No transitions, no
        exchange counting, no min/max gates.

        Args:
            session: Game session with exchanges and prompt snapshot.
            cartridge: Task cartridge with evaluation contract and safety.

        Returns:
            DebriefResult with token iterator and post-completion fields.
        """
        # 1. Resolve model tier and provider
        model_config = resolve_tier(cartridge.ai_config.model_preference)
        from backend.api.deps import create_provider
        from backend.config import get_settings
        provider = create_provider(model_config, get_settings())

        # 2. Assemble debrief context
        ctx = self._context_manager.assemble_debrief_call(
            session, cartridge, model_config.provider,
        )

        logger.info(
            "Trickster debrief: task=%s exchanges=%d",
            cartridge.task_id,
            len(session.exchanges),
        )

        async def _stream() -> AsyncIterator[str]:
            accumulated = ""

            # 3. Call provider and stream + accumulate
            async for event in provider.stream(
                system_prompt=ctx.system_prompt,
                messages=ctx.messages,
                model_config=model_config,
                tools=None,
            ):
                if isinstance(event, TextChunk):
                    accumulated += event.text
                    yield event.text
                elif isinstance(event, ToolCallEvent):
                    logger.warning(
                        "Unexpected tool call in debrief: %s",
                        event.function_name,
                    )

            # 4. Malformed response check \u2014 retry once if < 10 chars
            if len(accumulated) < _MIN_RESPONSE_LENGTH:
                logger.warning(
                    "Malformed debrief (<%d chars), retrying",
                    _MIN_RESPONSE_LENGTH,
                )
                async for event in provider.stream(
                    system_prompt=ctx.system_prompt,
                    messages=ctx.messages,
                    model_config=model_config,
                    tools=None,
                ):
                    if isinstance(event, TextChunk):
                        accumulated += event.text
                        yield event.text

                if len(accumulated) < _MIN_RESPONSE_LENGTH:
                    logger.error(
                        "Both debrief attempts produced malformed response "
                        "(<%d chars)",
                        _MIN_RESPONSE_LENGTH,
                    )
                    result.done_data = {"error": "malformed_response"}
                    result.usage = getattr(
                        provider, "_last_usage", None,
                    )
                    return

            # 5. Post-completion safety check (pedagogical exemption)
            safety_result = safety.check_output(
                accumulated, cartridge.safety, is_debrief=True,
            )

            # 6. Handle violation
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
                    "Debrief safety violation: boundary=%s",
                    violation.boundary,
                )
            else:
                # 7. Safe \u2014 store debrief exchange
                session.exchanges.append(
                    Exchange(role="trickster", content=accumulated)
                )
                result.done_data = {"debrief_complete": True}
                result.redaction_data = None

            # 8. Capture usage
            result.usage = getattr(provider, "_last_usage", None)

        result = DebriefResult(token_iterator=_stream())
        return result
