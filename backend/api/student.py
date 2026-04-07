"""Student-facing API routes — sessions, responses, profiles, GDPR.

Seven endpoints that form the student's entry point to Makaronas:
- Session lifecycle: create, next task, respond (SSE), debrief (SSE)
- Profile: radar view
- GDPR: deletion, export

All responses use the ApiResponse envelope. Streaming endpoints use the
Phase 3b SSE infrastructure. Auth is enforced on every endpoint via
get_current_user dependency.

Tier 3 orchestration module: imports from deps (Tier 2-3), schemas (Tier 1),
streaming (Tier 2), config (Tier 2), tasks/registry (Tier 2), tasks/schemas (Tier 1).

Created: Phase 4a
Updated: Phase 4b — replaced next_task stub with registry-backed implementation,
    evolved RespondRequest.action to open string
Updated: Phase 6b — replaced stub token generators with real TricksterEngine
    calls, added custom SSE generator for post-completion safety/transition handling
"""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from backend.ai.trickster import DebriefResult, TricksterEngine, TricksterResult
from backend.ai.usage import log_ai_call
from backend.ai.context import ContextManager
from backend.ai.safety import check_output
from backend.api.deps import (
    _get_api_key_for_provider,
    check_ai_readiness,
    create_provider,
    get_context_manager,
    get_current_user,
    get_database,
    get_session_store,
    get_task_registry,
    get_trickster_engine,
)
from backend.config import Settings, get_settings
from backend.hooks.interfaces import DatabaseAdapter, SessionStore
from backend.models import resolve_tier
from backend.schemas import (
    ApiError,
    ApiResponse,
    DoneEvent,
    ErrorEvent,
    GameSession,
    RedactEvent,
    TokenEvent,
    User,
)
from backend.streaming import create_sse_response, format_sse_event
from backend.tasks.registry import TaskRegistry
from backend.tasks.schemas import (
    ButtonInteraction,
    FreeformInteraction,
    GenericInteraction,
    InvestigationInteraction,
    Phase,
    TaskCartridge,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request bodies (API-boundary types, local to this module)
# ---------------------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    """Request body for POST /session."""

    roadmap_id: str | None = None
    language: str | None = None


class RespondRequest(BaseModel):
    """Request body for POST /session/{session_id}/respond."""

    action: str
    payload: str


class GenerateRequest(BaseModel):
    """Request body for POST /session/{session_id}/generate."""

    source_content: str
    student_prompt: str


class ChoiceRequest(BaseModel):
    """Request body for POST /session/{session_id}/choice."""

    target_phase: str
    context_label: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get_session_or_404(
    session_id: str,
    session_store: SessionStore,
) -> GameSession:
    """Retrieves a session or raises 404 with SESSION_NOT_FOUND."""
    session = await session_store.get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=ApiResponse(
                ok=False,
                error=ApiError(
                    code="SESSION_NOT_FOUND",
                    message="Session not found or expired.",
                ),
            ).model_dump(),
        )
    return session


def _check_ownership(session: GameSession, user: User) -> None:
    """Verifies the authenticated user owns the session. Raises 403 if not."""
    if session.student_id != user.id:
        raise HTTPException(
            status_code=403,
            detail=ApiResponse(
                ok=False,
                error=ApiError(
                    code="FORBIDDEN",
                    message="You do not have access to this session.",
                ),
            ).model_dump(),
        )


def _check_profile_access(student_id: str, user: User) -> None:
    """Checks if the user can access the given student's profile.

    Students can only access their own profile. Teachers can access any
    student's profile within their school (school_id scoping handled by
    the database layer).
    """
    if user.role == "student" and user.id != student_id:
        raise HTTPException(
            status_code=403,
            detail=ApiResponse(
                ok=False,
                error=ApiError(
                    code="FORBIDDEN",
                    message="You can only access your own profile.",
                ),
            ).model_dump(),
        )


# ---------------------------------------------------------------------------
# Cartridge → student response helpers
# ---------------------------------------------------------------------------


def _find_initial_phase(cartridge: TaskCartridge) -> Phase:
    """Finds the initial phase in the cartridge's phase list.

    Returns:
        The Phase object matching cartridge.initial_phase.

    Raises:
        ValueError: If initial_phase ID doesn't match any phase (should
            never happen with validated cartridges, but defensive).
    """
    for phase in cartridge.phases:
        if phase.id == cartridge.initial_phase:
            return phase
    raise ValueError(
        f"initial_phase '{cartridge.initial_phase}' not found in phases"
    )


def _find_phase_by_id(cartridge: TaskCartridge, phase_id: str) -> Phase | None:
    """Finds a phase by ID in the cartridge's phase list.

    Returns None if not found (defensive — should not happen with
    validated cartridges, but the done event enrichment should degrade
    gracefully rather than crash the stream).
    """
    for phase in cartridge.phases:
        if phase.id == phase_id:
            return phase
    return None


def _derive_content_blocks(cartridge: TaskCartridge, phase: Phase) -> list[dict]:
    """Resolves visible_blocks IDs to serialized presentation block dicts.

    Preserves the visible_blocks ordering (display order for the frontend).
    Skips unresolved block IDs with a warning — a missing reference is a
    loader validation gap, not a student-facing error.
    """
    block_lookup = {block.id: block for block in cartridge.presentation_blocks}
    result = []
    for block_id in phase.visible_blocks:
        block = block_lookup.get(block_id)
        if block is None:
            logger.warning(
                "Phase '%s' references block '%s' not found in cartridge '%s'",
                phase.id,
                block_id,
                cartridge.task_id,
            )
            continue
        result.append(block.model_dump())
    return result


def _derive_available_actions(phase: Phase) -> list[str]:
    """Derives available action types from the phase's interaction config."""
    interaction = phase.interaction
    if interaction is None:
        return []
    if isinstance(interaction, ButtonInteraction):
        return ["button_click"]
    if isinstance(interaction, FreeformInteraction):
        return ["freeform"]
    if isinstance(interaction, InvestigationInteraction):
        return ["investigate"]
    if isinstance(interaction, GenericInteraction):
        return [interaction.type]
    return []


def _derive_trickster_intro(phase: Phase) -> str | None:
    """Derives the trickster intro from the initial phase.

    Priority: phase.trickster_content > FreeformInteraction.trickster_opening > None.
    """
    if phase.trickster_content is not None:
        return phase.trickster_content
    if isinstance(phase.interaction, FreeformInteraction):
        return phase.interaction.trickster_opening
    return None


def _derive_phase_response(
    cartridge: TaskCartridge,
    phase: Phase,
    session: "GameSession | None" = None,
) -> dict:
    """Assembles a phase content response dict for the student API.

    Combines existing derivation helpers with terminal phase fields and
    reveal gating. Reused by /current, /next, /choice, and done event
    enrichment to eliminate duplication.

    When session is provided, student-generated artifacts (e.g., published
    articles from prior tasks) are injected into the content blocks.

    Returns:
        Dict with keys: task_id, task_type, medium, title, content,
        available_actions, trickster_intro, current_phase, is_terminal,
        evaluation_outcome, reveal, interaction.
    """
    content = _derive_content_blocks(cartridge, phase)

    # Inject student's published article — only for the comments task
    if session and cartridge.task_id == "task-petryla-comments-001" and phase.id == "dialogue":
        article_text = None

        # Primary: check generated_artifacts
        if session.generated_artifacts:
            for artifact in session.generated_artifacts:
                if artifact.get("type") == "student_article" and artifact.get("text"):
                    article_text = artifact["text"]
                    break

        # Fallback: read from file (written by respond endpoint or trickster engine)
        if not article_text:
            try:
                from pathlib import Path
                p = Path("/tmp/student_article.txt")
                if p.exists():
                    article_text = p.read_text(encoding="utf-8").strip()
            except Exception:
                pass

        if article_text:
            student_block = {
                "id": "student-published-article",
                "type": "text",
                "data": {"text": article_text, "style": "student-article"},
                "text": article_text,
                "style": "student-article",
            }
            insert_idx = 1 if content else 0
            content.insert(insert_idx, student_block)

    return {
        "task_id": cartridge.task_id,
        "task_type": cartridge.task_type,
        "medium": cartridge.medium,
        "title": cartridge.title,
        "content": content,
        "available_actions": _derive_available_actions(phase),
        "trickster_intro": _derive_trickster_intro(phase),
        "current_phase": phase.id,
        "is_terminal": phase.is_terminal,
        "evaluation_outcome": phase.evaluation_outcome,
        "reveal": cartridge.reveal.model_dump() if phase.is_terminal else None,
        "interaction": phase.interaction.model_dump() if phase.interaction else None,
        "ai_transitions": phase.ai_transitions.model_dump() if phase.ai_transitions else None,
    }


def _get_legal_choice_targets(phase: Phase) -> set[str]:
    """Returns the set of phase IDs reachable via /choice from this phase.

    ButtonInteraction phases allow transitions to each choice's target_phase.
    InvestigationInteraction phases allow a single submit_target transition.
    AI phases also allow manual advance to ai_transitions targets (failsafe
    for when the model fails to call the transition tool).
    """
    targets: set[str] = set()
    interaction = phase.interaction
    if isinstance(interaction, ButtonInteraction):
        targets = {c.target_phase for c in interaction.choices}
    elif isinstance(interaction, InvestigationInteraction):
        targets = {interaction.submit_target}
    # AI phases: allow manual advance to any ai_transitions target
    if phase.ai_transitions:
        for target in (phase.ai_transitions.on_success,
                       phase.ai_transitions.on_partial,
                       phase.ai_transitions.on_max_exchanges):
            if target:
                targets.add(target)
    return targets


# ---------------------------------------------------------------------------
# AI integration helpers
# ---------------------------------------------------------------------------


def _resolve_ai_phase(
    session: GameSession,
    cartridge: TaskCartridge,
) -> Phase:
    """Resolves the current phase from session state and validates for AI use.

    Args:
        session: Active game session with current_phase set.
        cartridge: Task cartridge loaded from the registry.

    Returns:
        The validated Phase object, ready for TricksterEngine.respond().

    Raises:
        HTTPException: 422 if no active phase, 409 if phase is stale,
            422 if phase is not an AI phase with FreeformInteraction.
    """
    if session.current_phase is None:
        raise HTTPException(
            status_code=422,
            detail=ApiResponse(
                ok=False,
                error=ApiError(
                    code="NO_ACTIVE_PHASE",
                    message="No active phase in this session.",
                ),
            ).model_dump(),
        )

    # Find the phase in the cartridge
    phase: Phase | None = None
    for p in cartridge.phases:
        if p.id == session.current_phase:
            phase = p
            break

    if phase is None:
        raise HTTPException(
            status_code=409,
            detail=ApiResponse(
                ok=False,
                error=ApiError(
                    code="TASK_CONTENT_UPDATED",
                    message="Task content has been updated since your last interaction.",
                ),
                data={"initial_phase": cartridge.initial_phase},
            ).model_dump(),
        )

    if not phase.is_ai_phase or not isinstance(phase.interaction, FreeformInteraction):
        raise HTTPException(
            status_code=422,
            detail=ApiResponse(
                ok=False,
                error=ApiError(
                    code="NOT_AI_PHASE",
                    message="Current phase does not support freeform AI interaction.",
                ),
            ).model_dump(),
        )

    if phase.ai_transitions is None:
        raise HTTPException(
            status_code=422,
            detail=ApiResponse(
                ok=False,
                error=ApiError(
                    code="NOT_AI_PHASE",
                    message="Current phase has no AI transition configuration.",
                ),
            ).model_dump(),
        )

    return phase


def _check_generation_readiness(settings: Settings) -> list[str]:
    """Checks whether the generation path (fast tier) is ready.

    Returns a list of issues. Empty list means ready.
    Simpler than check_ai_readiness() — no prompt validation needed
    since generation uses a hardcoded system prompt.
    """
    issues: list[str] = []
    try:
        model_config = resolve_tier("fast")
    except KeyError:
        issues.append("Unknown model tier: 'fast'")
        return issues

    api_key = _get_api_key_for_provider(model_config.provider, settings)
    if not api_key:
        issues.append(
            f"Missing API key for provider {model_config.provider!r} "
            f"(tier: 'fast')"
        )
    return issues


async def _stream_trickster_response(
    result: TricksterResult | DebriefResult,
    session: GameSession,
    session_store: SessionStore,
    cartridge: TaskCartridge,
    call_type: str,
    timeout_seconds: float = 60.0,
) -> AsyncGenerator[str, None]:
    """Turns a TricksterResult/DebriefResult into a full SSE event stream.

    Mirrors stream_ai_response() but adds post-completion result inspection:
    checks redaction_data and done_data after iterator exhaustion, emits
    the appropriate final event, logs usage, and saves the session.

    Args:
        result: Engine result with token_iterator and post-completion fields.
        session: Game session to save after stream completion.
        session_store: Session persistence interface.
        cartridge: Task cartridge for usage logging (model tier resolution).
        call_type: "trickster" or "debrief" for usage logging.
        timeout_seconds: Maximum wall-clock time for the stream.

    Yields:
        SSE-formatted strings (token events, then one done/redact/error event).
    """
    accumulated: list[str] = []
    start_time = time.monotonic()

    try:
        async with asyncio.timeout(timeout_seconds):
            async for token in result.token_iterator:
                accumulated.append(token)
                yield format_sse_event("token", TokenEvent(text=token))

    except TimeoutError:
        partial = "".join(accumulated)
        logger.warning(
            "AI stream timed out after %.1fs, partial_text length=%d",
            timeout_seconds,
            len(partial),
        )
        yield format_sse_event(
            "error",
            ErrorEvent(
                code="AI_TIMEOUT",
                message="AI atsakymas u\u017etruko per ilgai. Bandykite dar kart\u0105.",
                partial_text=partial,
            ),
        )
        return

    except Exception as exc:
        partial = "".join(accumulated)
        logger.warning(
            "AI stream error: %s, partial_text length=%d",
            exc,
            len(partial),
        )
        yield format_sse_event(
            "error",
            ErrorEvent(
                code="STREAM_ERROR",
                message="AI atsakyme \u012fvyko klaida. Bandykite dar kart\u0105.",
                partial_text=partial,
            ),
        )
        return

    # --- Post-completion: result fields now populated ---
    elapsed_ms = (time.monotonic() - start_time) * 1000
    full_text = "".join(accumulated)

    # Check for redaction (safety violation takes priority)
    if result.redaction_data is not None:
        yield format_sse_event(
            "redact",
            RedactEvent(fallback_text=result.redaction_data["fallback_text"]),
        )
    else:
        # Update session phase on transition (respond only)
        done_data = result.done_data or {}
        if done_data.get("next_phase") is not None:
            session.current_phase = done_data["next_phase"]

        # Record task history on terminal transition
        if done_data.get("phase_transition") is not None:
            session.task_history.append({
                "task_id": cartridge.task_id,
                "evaluation_outcome": done_data["phase_transition"],
                "exchange_count": done_data["exchanges_count"],
                "intensity_score": done_data.get("intensity_score"),
                "is_clean": cartridge.is_clean,
            })

            # Telemetry: save task completion data incrementally
            try:
                from backend.telemetry import save_task_completion
                save_task_completion(
                    session=session,
                    task_id=cartridge.task_id,
                    phase_exchanges=[
                        {
                            "role": e.role,
                            "content": e.content,
                            "timestamp": e.timestamp.isoformat(),
                        }
                        for e in session.exchanges
                    ],
                )
            except Exception as exc:
                logger.warning("Telemetry save failed: %s", exc)

        # Enrich done event with next phase content for seamless transitions
        if done_data.get("next_phase") is not None:
            target_phase = _find_phase_by_id(cartridge, done_data["next_phase"])
            if target_phase is not None:
                done_data["next_phase_content"] = _derive_phase_response(
                    cartridge, target_phase
                )

        yield format_sse_event(
            "done",
            DoneEvent(full_text=full_text, data=done_data),
        )

    # --- Usage logging ---
    if result.usage is not None and cartridge.ai_config is not None:
        try:
            model_config = resolve_tier(cartridge.ai_config.model_preference)
            log_ai_call(
                model_id=model_config.model_id,
                prompt_tokens=result.usage.prompt_tokens,
                completion_tokens=result.usage.completion_tokens,
                latency_ms=elapsed_ms,
                task_id=cartridge.task_id,
                session_id=session.session_id,
                call_type=call_type,
            )
        except Exception:
            logger.warning("Failed to log AI usage", exc_info=True)

    # --- Session persistence (last step) ---
    try:
        await session_store.save_session(session)
    except Exception:
        logger.error("Failed to save session after AI stream", exc_info=True)


async def _static_fallback_stream() -> AsyncGenerator[str, None]:
    """Emits a single DoneEvent with fallback data when AI is unavailable."""
    yield format_sse_event(
        "done",
        DoneEvent(
            full_text="",
            data={"fallback": True, "reason": "AI temporarily unavailable"},
        ),
    )


# ---------------------------------------------------------------------------
# Session endpoints
# ---------------------------------------------------------------------------


@router.post("/session")
async def create_session(
    body: CreateSessionRequest,
    request: Request,
    user: User = Depends(get_current_user),
    session_store: SessionStore = Depends(get_session_store),
) -> dict:
    """Creates a new game session.

    Returns the session_id and a stub intro message. The language defaults
    to the platform's configured default if not specified.
    """
    settings = get_settings()
    language = body.language or settings.default_language

    user_agent = request.headers.get("user-agent")

    session = GameSession(
        session_id=str(uuid4()),
        student_id=user.id,
        school_id=user.school_id,
        language=language,
        roadmap_id=body.roadmap_id,
        user_agent=user_agent,
    )
    await session_store.save_session(session)

    return ApiResponse(
        ok=True,
        data={
            "session_id": session.session_id,
            "language": session.language,
            "intro": "Welcome to the Trickster's arena. Stay sharp.",
        },
    ).model_dump()


@router.get("/session/{session_id}/next")
async def next_task(
    session_id: str,
    task_id: str | None = None,
    user: User = Depends(get_current_user),
    session_store: SessionStore = Depends(get_session_store),
    registry: TaskRegistry = Depends(get_task_registry),
) -> dict:
    """Returns the next task content for the session.

    Loads real cartridge data from the registry. The ``task_id`` query param
    overrides the session's current_task — useful for demo flows before V9
    (Roadmap Engine) provides automatic task assignment.
    """
    session = await _get_session_or_404(session_id, session_store)
    _check_ownership(session, user)

    # --- Task ID resolution ---
    resolved_task_id = task_id or session.current_task
    if resolved_task_id is None:
        raise HTTPException(
            status_code=422,
            detail=ApiResponse(
                ok=False,
                error=ApiError(
                    code="NO_TASK_ASSIGNED",
                    message="No task assigned to this session. Provide task_id query param.",
                ),
            ).model_dump(),
        )

    # --- Load cartridge ---
    cartridge = registry.get_task(resolved_task_id)
    if cartridge is None or cartridge.status == "draft":
        raise HTTPException(
            status_code=404,
            detail=ApiResponse(
                ok=False,
                error=ApiError(
                    code="TASK_NOT_FOUND",
                    message=f"Task '{resolved_task_id}' not found.",
                ),
            ).model_dump(),
        )

    # --- Stale phase detection (Framework P21) ---
    # Only when returning to the SAME task with an existing phase
    if (
        resolved_task_id == session.current_task
        and session.current_phase is not None
        and not registry.is_phase_valid(resolved_task_id, session.current_phase)
    ):
        raise HTTPException(
            status_code=409,
            detail=ApiResponse(
                ok=False,
                error=ApiError(
                    code="TASK_CONTENT_UPDATED",
                    message="Task content has been updated since your last interaction.",
                ),
                data={"initial_phase": cartridge.initial_phase},
            ).model_dump(),
        )

    # --- Reset per-task state when switching tasks ---
    is_task_switch = (
        session.current_task is not None
        and session.current_task != resolved_task_id
    )
    if is_task_switch:
        # Telemetry: save completion for the OUTGOING task if it isn't already
        # in task_history. AI tasks save themselves via /respond when the
        # evaluator transitions. Static tasks (no AI exchanges) need this
        # safety net so they're captured too.
        outgoing_task_id = session.current_task
        already_recorded = any(
            entry.get("task_id") == outgoing_task_id
            for entry in session.task_history
        )
        if not already_recorded:
            outgoing_cartridge = registry.get_task(outgoing_task_id)
            session.task_history.append({
                "task_id": outgoing_task_id,
                "evaluation_outcome": "static_complete",
                "exchange_count": 0,
                "intensity_score": None,
                "is_clean": (
                    outgoing_cartridge.is_clean
                    if outgoing_cartridge is not None
                    else False
                ),
            })
            try:
                from backend.telemetry import save_task_completion
                save_task_completion(
                    session=session,
                    task_id=outgoing_task_id,
                    phase_exchanges=[
                        {
                            "role": e.role,
                            "content": e.content,
                            "timestamp": e.timestamp.isoformat(),
                        }
                        for e in session.exchanges
                    ],
                )
            except Exception as exc:
                logger.warning(
                    "Static task telemetry save failed for %s: %s",
                    outgoing_task_id,
                    exc,
                )

        session.exchanges = []
        session.choices = []
        session.turn_intensities = []
        session.generated_artifacts = []
        session.prompt_snapshots = None
        session.checklist_progress = {}
        session.investigation_paths = []
        session.raw_performance = {}
        session.last_redaction_reason = None

    # --- Derive response from initial phase ---
    initial_phase = _find_initial_phase(cartridge)

    # --- Update session ---
    session.current_task = resolved_task_id
    session.current_phase = cartridge.initial_phase
    await session_store.save_session(session)

    return ApiResponse(
        ok=True,
        data=_derive_phase_response(cartridge, initial_phase, session=session),
    ).model_dump()


@router.get("/session/{session_id}/current")
async def current_session(
    session_id: str,
    user: User = Depends(get_current_user),
    session_store: SessionStore = Depends(get_session_store),
    registry: TaskRegistry = Depends(get_task_registry),
) -> dict:
    """Returns the session's current phase content without mutating state.

    Read-only recovery endpoint for page refresh. Returns phase content
    plus accumulated dialogue history so the frontend can re-render the
    student's current position. Does NOT modify the session.
    """
    session = await _get_session_or_404(session_id, session_store)
    _check_ownership(session, user)

    # --- No active task ---
    if session.current_task is None:
        return ApiResponse(ok=True, data={"current_task": None}).model_dump()

    # --- Load cartridge ---
    cartridge = registry.get_task(session.current_task)
    if cartridge is None:
        raise HTTPException(
            status_code=404,
            detail=ApiResponse(
                ok=False,
                error=ApiError(
                    code="TASK_NOT_FOUND",
                    message=f"Task '{session.current_task}' not found.",
                ),
            ).model_dump(),
        )

    # --- Stale phase detection (Framework P21) ---
    if (
        session.current_phase is not None
        and not registry.is_phase_valid(session.current_task, session.current_phase)
    ):
        raise HTTPException(
            status_code=409,
            detail=ApiResponse(
                ok=False,
                error=ApiError(
                    code="TASK_CONTENT_UPDATED",
                    message="Task content has been updated since your last interaction.",
                ),
                data={"initial_phase": cartridge.initial_phase},
            ).model_dump(),
        )

    # --- Find current phase ---
    current_phase = None
    for phase in cartridge.phases:
        if phase.id == session.current_phase:
            current_phase = phase
            break

    if current_phase is None:
        raise HTTPException(
            status_code=404,
            detail=ApiResponse(
                ok=False,
                error=ApiError(
                    code="TASK_NOT_FOUND",
                    message=f"Phase '{session.current_phase}' not found in task.",
                ),
            ).model_dump(),
        )

    # --- Build response ---
    data = _derive_phase_response(cartridge, current_phase, session=session)
    data["dialogue_history"] = [
        exchange.model_dump() for exchange in session.exchanges
    ]

    return ApiResponse(ok=True, data=data).model_dump()


@router.post("/session/{session_id}/choice")
async def choose(
    session_id: str,
    body: ChoiceRequest,
    user: User = Depends(get_current_user),
    session_store: SessionStore = Depends(get_session_store),
    registry: TaskRegistry = Depends(get_task_registry),
) -> dict:
    """Validates and executes a student's phase transition choice.

    Records the choice context for AI continuity, validates the target
    phase is a legal outbound edge from the current phase, advances
    the session, and returns the new phase content. Prevents DevTools
    phase-skipping (Framework P13).
    """
    # 1. Session exists
    session = await _get_session_or_404(session_id, session_store)

    # 2. Ownership
    _check_ownership(session, user)

    # 3. Active task
    if session.current_task is None:
        raise HTTPException(
            status_code=422,
            detail=ApiResponse(
                ok=False,
                error=ApiError(
                    code="NO_TASK_ASSIGNED",
                    message="No task assigned to this session.",
                ),
            ).model_dump(),
        )

    # 4. Load cartridge
    cartridge = registry.get_task(session.current_task)
    if cartridge is None:
        raise HTTPException(
            status_code=404,
            detail=ApiResponse(
                ok=False,
                error=ApiError(
                    code="TASK_NOT_FOUND",
                    message=f"Task '{session.current_task}' not found.",
                ),
            ).model_dump(),
        )

    # 5. Active phase
    if session.current_phase is None:
        raise HTTPException(
            status_code=422,
            detail=ApiResponse(
                ok=False,
                error=ApiError(
                    code="NO_ACTIVE_PHASE",
                    message="No active phase in this session.",
                ),
            ).model_dump(),
        )

    # 6. Stale phase detection (Framework P21) — must precede graph validation
    current_phase: Phase | None = None
    for p in cartridge.phases:
        if p.id == session.current_phase:
            current_phase = p
            break

    if current_phase is None:
        raise HTTPException(
            status_code=409,
            detail=ApiResponse(
                ok=False,
                error=ApiError(
                    code="TASK_CONTENT_UPDATED",
                    message="Task content has been updated since your last interaction.",
                ),
                data={"initial_phase": cartridge.initial_phase},
            ).model_dump(),
        )

    # 7. Phase-graph validation — the security gate (Framework P13)
    legal_targets = _get_legal_choice_targets(current_phase)
    if body.target_phase not in legal_targets:
        raise HTTPException(
            status_code=422,
            detail=ApiResponse(
                ok=False,
                error=ApiError(
                    code="INVALID_PHASE_TRANSITION",
                    message="The requested phase transition is not allowed from the current phase.",
                ),
            ).model_dump(),
        )

    # 8. Resolve target phase (defense in depth — cartridge authoring bug check)
    target_phase: Phase | None = None
    for p in cartridge.phases:
        if p.id == body.target_phase:
            target_phase = p
            break

    if target_phase is None:
        raise HTTPException(
            status_code=404,
            detail=ApiResponse(
                ok=False,
                error=ApiError(
                    code="TASK_NOT_FOUND",
                    message=f"Target phase '{body.target_phase}' not found in task.",
                ),
            ).model_dump(),
        )

    # 9. Record choice context + advance session
    session.choices.append({
        "phase": session.current_phase,
        "target_phase": body.target_phase,
        "context_label": body.context_label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    session.current_phase = body.target_phase
    await session_store.save_session(session)

    # 10. Build response
    return ApiResponse(
        ok=True,
        data=_derive_phase_response(cartridge, target_phase, session=session),
    ).model_dump()


@router.post("/session/{session_id}/respond")
async def respond(
    session_id: str,
    body: RespondRequest,
    user: User = Depends(get_current_user),
    session_store: SessionStore = Depends(get_session_store),
    registry: TaskRegistry = Depends(get_task_registry),
    engine: TricksterEngine = Depends(get_trickster_engine),
):
    """Accepts a student response and streams a Trickster reply via SSE.

    Validation and ownership checks happen before streaming begins —
    once SSE starts, HTTP status is locked at 200. The response is
    powered by real TricksterEngine calls with post-completion safety
    checking, transition signal extraction, and usage logging.
    """
    session = await _get_session_or_404(session_id, session_store)
    _check_ownership(session, user)

    # Load cartridge for this session's current task
    if session.current_task is None:
        raise HTTPException(
            status_code=422,
            detail=ApiResponse(
                ok=False,
                error=ApiError(
                    code="NO_TASK_ASSIGNED",
                    message="No task assigned to this session.",
                ),
            ).model_dump(),
        )

    cartridge = registry.get_task(session.current_task)
    if cartridge is None:
        raise HTTPException(
            status_code=404,
            detail=ApiResponse(
                ok=False,
                error=ApiError(
                    code="TASK_NOT_FOUND",
                    message=f"Task '{session.current_task}' not found.",
                ),
            ).model_dump(),
        )

    # Resolve and validate AI phase
    phase = _resolve_ai_phase(session, cartridge)

    # Check AI readiness (pre-stream validation)
    settings = get_settings()
    issues = check_ai_readiness(cartridge, settings)
    if issues:
        if cartridge.ai_config and cartridge.ai_config.has_static_fallback:
            # Serve static fallback as a single DoneEvent
            generator = _static_fallback_stream()
            return create_sse_response(generator)
        else:
            raise HTTPException(
                status_code=503,
                detail=ApiResponse(
                    ok=False,
                    error=ApiError(
                        code="AI_UNAVAILABLE",
                        message="AI paslauga laikinai neprieinama.",
                    ),
                ).model_dump(),
            )

    # Call engine
    result = await engine.respond(session, cartridge, phase, body.payload)

    # Build SSE stream from result
    generator = _stream_trickster_response(
        result, session, session_store, cartridge, call_type="trickster",
    )
    return create_sse_response(generator)


@router.get("/session/{session_id}/debrief")
async def debrief(
    session_id: str,
    user: User = Depends(get_current_user),
    session_store: SessionStore = Depends(get_session_store),
    registry: TaskRegistry = Depends(get_task_registry),
    engine: TricksterEngine = Depends(get_trickster_engine),
):
    """Streams a debrief summary via SSE powered by TricksterEngine.

    Validation and ownership checks happen before streaming begins.
    Debrief requires AI — there is no meaningful static fallback.
    """
    session = await _get_session_or_404(session_id, session_store)
    _check_ownership(session, user)

    # Load cartridge
    if session.current_task is None:
        raise HTTPException(
            status_code=422,
            detail=ApiResponse(
                ok=False,
                error=ApiError(
                    code="NO_TASK_ASSIGNED",
                    message="No task assigned to this session.",
                ),
            ).model_dump(),
        )

    cartridge = registry.get_task(session.current_task)
    if cartridge is None:
        raise HTTPException(
            status_code=404,
            detail=ApiResponse(
                ok=False,
                error=ApiError(
                    code="TASK_NOT_FOUND",
                    message=f"Task '{session.current_task}' not found.",
                ),
            ).model_dump(),
        )

    # Debrief requires ai_config
    if cartridge.ai_config is None:
        raise HTTPException(
            status_code=422,
            detail=ApiResponse(
                ok=False,
                error=ApiError(
                    code="NOT_AI_TASK",
                    message="This task does not support AI debrief.",
                ),
            ).model_dump(),
        )

    # Check AI readiness
    settings = get_settings()
    issues = check_ai_readiness(cartridge, settings)
    if issues:
        raise HTTPException(
            status_code=503,
            detail=ApiResponse(
                ok=False,
                error=ApiError(
                    code="AI_UNAVAILABLE",
                    message="AI paslauga laikinai neprieinama.",
                ),
            ).model_dump(),
        )

    # Call engine
    result = await engine.debrief(session, cartridge)

    # Build SSE stream from result
    generator = _stream_trickster_response(
        result, session, session_store, cartridge, call_type="debrief",
    )
    return create_sse_response(generator)


# ---------------------------------------------------------------------------
# Generation endpoint (Context-Isolated Tool AI)
# ---------------------------------------------------------------------------


@router.post("/session/{session_id}/generate")
async def generate(
    session_id: str,
    body: GenerateRequest,
    user: User = Depends(get_current_user),
    session_store: SessionStore = Depends(get_session_store),
    registry: TaskRegistry = Depends(get_task_registry),
    context_manager: ContextManager = Depends(get_context_manager),
) -> dict:
    """Generates content via the context-isolated Tool AI.

    The student provides a prompt and source content; the Tool AI produces
    output using the ``fast`` tier model. Generated text is safety-checked
    against the current task's safety config, stored as an artifact in the
    session, and returned as a standard JSON response (not SSE).
    """
    # --- Validation (same pattern as /respond) ---
    session = await _get_session_or_404(session_id, session_store)
    _check_ownership(session, user)

    # Input validation: non-empty strings
    if not body.source_content.strip():
        raise HTTPException(
            status_code=422,
            detail=ApiResponse(
                ok=False,
                error=ApiError(
                    code="INVALID_REQUEST",
                    message="source_content negali b\u016bti tu\u0161\u010dias.",
                ),
            ).model_dump(),
        )
    if not body.student_prompt.strip():
        raise HTTPException(
            status_code=422,
            detail=ApiResponse(
                ok=False,
                error=ApiError(
                    code="INVALID_REQUEST",
                    message="student_prompt negali b\u016bti tu\u0161\u010dias.",
                ),
            ).model_dump(),
        )

    # Load cartridge (needed for safety config)
    if session.current_task is None:
        raise HTTPException(
            status_code=422,
            detail=ApiResponse(
                ok=False,
                error=ApiError(
                    code="NO_TASK_ASSIGNED",
                    message="Sesijai n\u0117ra priskirta u\u017eduotis.",
                ),
            ).model_dump(),
        )

    cartridge = registry.get_task(session.current_task)
    if cartridge is None:
        raise HTTPException(
            status_code=404,
            detail=ApiResponse(
                ok=False,
                error=ApiError(
                    code="TASK_NOT_FOUND",
                    message=f"U\u017eduotis '{session.current_task}' nerasta.",
                ),
            ).model_dump(),
        )

    # Check generation readiness (fast tier API key)
    settings = get_settings()
    issues = _check_generation_readiness(settings)
    if issues:
        raise HTTPException(
            status_code=503,
            detail=ApiResponse(
                ok=False,
                error=ApiError(
                    code="AI_UNAVAILABLE",
                    message="AI paslauga laikinai neprieinama.",
                ),
            ).model_dump(),
        )

    # --- Assemble context and call provider ---
    assembled = context_manager.assemble_generation_call(
        body.source_content, body.student_prompt,
    )

    model_config = resolve_tier("fast")
    provider = create_provider(model_config, settings)

    start_time = time.monotonic()
    generated_text, usage = await provider.complete(
        system_prompt=assembled.system_prompt,
        messages=assembled.messages,
        model_config=model_config,
        tools=None,
    )
    elapsed_ms = (time.monotonic() - start_time) * 1000

    # --- Safety check (cartridge.safety is always present) ---
    safety_redacted = False
    safety_result = check_output(generated_text, cartridge.safety, is_debrief=False)
    if not safety_result.is_safe and safety_result.violation is not None:
        generated_text = safety_result.violation.fallback_text
        safety_redacted = True

    # --- Store artifact ---
    artifact = {
        "student_prompt": body.student_prompt,
        "generated_text": generated_text,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "safety_redacted": safety_redacted,
    }
    session.generated_artifacts.append(artifact)
    artifact_index = len(session.generated_artifacts) - 1

    # --- Usage logging ---
    try:
        log_ai_call(
            model_id=model_config.model_id,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            latency_ms=elapsed_ms,
            task_id=cartridge.task_id,
            session_id=session.session_id,
            call_type="generate",
        )
    except Exception:
        logger.warning("Failed to log AI usage for generate", exc_info=True)

    # --- Persist session ---
    await session_store.save_session(session)

    # --- Build response ---
    response_data: dict = {
        "generated_text": generated_text,
        "artifact_index": artifact_index,
    }
    if safety_redacted:
        response_data["safety_redacted"] = True

    return ApiResponse(ok=True, data=response_data).model_dump()


# ---------------------------------------------------------------------------
# Profile endpoints
# ---------------------------------------------------------------------------


@router.get("/profile/{student_id}/radar")
async def radar_profile(
    student_id: str,
    user: User = Depends(get_current_user),
    database: DatabaseAdapter = Depends(get_database),
) -> dict:
    """Returns the student's radar profile data (stub).

    Accessible by the student themselves or by a teacher.
    """
    _check_profile_access(student_id, user)

    profile = await database.get_student_profile(student_id, user.school_id)

    if profile is not None:
        return ApiResponse(
            ok=True,
            data={
                "student_id": profile.student_id,
                "trigger_vulnerability": profile.trigger_vulnerability,
                "technique_recognition": {
                    k: {"caught": v.caught, "total": v.total}
                    for k, v in profile.technique_recognition.items()
                },
                "sessions_completed": profile.sessions_completed,
                "last_active": profile.last_active.isoformat() if profile.last_active else None,
            },
        ).model_dump()

    # No profile yet — return stub defaults
    return ApiResponse(
        ok=True,
        data={
            "student_id": student_id,
            "trigger_vulnerability": {},
            "technique_recognition": {},
            "sessions_completed": 0,
            "last_active": None,
        },
    ).model_dump()


# ---------------------------------------------------------------------------
# GDPR endpoints (Framework Principle 3 — Sacred Trust)
# ---------------------------------------------------------------------------


@router.delete("/profile/{student_id}")
async def delete_profile(
    student_id: str,
    user: User = Depends(get_current_user),
    database: DatabaseAdapter = Depends(get_database),
) -> dict:
    """Deletes all stored data for the student (GDPR right to deletion).

    Students can delete their own profile. Teachers can delete any student's
    profile within their school.
    """
    _check_profile_access(student_id, user)

    await database.delete_student_profile(student_id, user.school_id)

    return ApiResponse(
        ok=True,
        data={"deleted": True, "student_id": student_id},
    ).model_dump()


@router.get("/profile/{student_id}/export")
async def export_profile(
    student_id: str,
    user: User = Depends(get_current_user),
    database: DatabaseAdapter = Depends(get_database),
) -> dict:
    """Exports all stored data for the student (GDPR right to access).

    Students can export their own data. Teachers can export any student's
    data within their school.
    """
    _check_profile_access(student_id, user)

    export_data = await database.export_student_data(student_id, user.school_id)

    return ApiResponse(
        ok=True,
        data=export_data,
    ).model_dump()


# ---------------------------------------------------------------------------
# GET /session/{session_id}/report — session completion report
# ---------------------------------------------------------------------------


@router.get("/session/{session_id}/report")
async def session_report(
    session_id: str,
    user: User = Depends(get_current_user),
    session_store: SessionStore = Depends(get_session_store),
) -> dict:
    """Generates a personalized end-of-session report.

    Reviews the student's performance across all completed tasks and
    generates an encouraging summary (100-160 words, Lithuanian).
    """
    session = await _get_session_or_404(session_id, session_store)
    _check_ownership(session, user)

    if not session.task_history:
        return ApiResponse(
            ok=True,
            data={"report": "Sveikiname baigus sesiją!"},
        ).model_dump()

    # Human-readable Lithuanian descriptions for each task — anchors the AI
    # to real content so it can't hallucinate tasks that weren't done.
    task_descriptions_lt = {
        "task-petryla-001": (
            "straipsnių apie mokytoją Petrylą tyrimas (perskaitė du skirtingus "
            "straipsnius, ieškojo šaltinių ir parašė žinutę draugams)"
        ),
        "task-petryla-comments-001": (
            "komentarų skilties bei nuotraukos analizė (atpažino botus, trolius "
            "ir tikrus dalintojus, ištyrė nuotraukos kilmę)"
        ),
        "task-petryla-network-001": (
            "botų tinklo vizualizacija ir trumpas straipsnis apie tinklus"
        ),
        "task-petryla-video-001": (
            "deepfake vaizdo įrašo analizė (ieškojo požymių, kad asmuo nerealus)"
        ),
    }

    # Build the task list from actual session data — no hardcoded sequence.
    task_lines = []
    for entry in session.task_history:
        task_id = entry.get("task_id", "")
        desc = task_descriptions_lt.get(task_id, task_id)
        outcome = entry.get("evaluation_outcome", "unknown")
        outcome_lt = {
            "on_success": "puikiai atliko",
            "on_partial": "iš dalies atliko",
            "on_max_exchanges": "atliko su pagalba",
        }.get(outcome, "atliko")
        task_lines.append(f"- {desc} — {outcome_lt}")

    summary_text = "\n".join(task_lines)

    # Personal touch: pull the student's actual article from generated_artifacts.
    # This is THEIR words — the strongest signal for a unique, personal report.
    student_article_text = None
    for artifact in session.generated_artifacts:
        if artifact.get("type") == "student_article" and artifact.get("text"):
            student_article_text = artifact["text"].strip()
            break

    personal_section = ""
    if student_article_text:
        personal_section = (
            f"\n\nMokinio žinutė draugams (jo paties žodžiais):\n"
            f'"{student_article_text}"\n\n'
            f"PASTABA: Pakomentuok šią žinutę konkrečiai vienu sakiniu — "
            f"pagirk už tai, kas joje stipru, arba paminėk konkrečią detalę, "
            f"kurią pastebėjai. Nepakartok jos visos."
        )

    system_prompt = (
        "Tu esi Makaronas — DI asistentas, kuris moko paauglius atpažinti "
        "dezinformaciją. Mokinys ką tik baigė visas užduotis. Parašyk ASMENINĘ "
        "ataskaitą lietuvių kalba (120-180 žodžių). Naudok 'jūs' formą.\n\n"
        "KRITIŠKAI SVARBU: minėk TIK tas užduotis, kurios išvardintos žemiau. "
        "NESUGALVOK papildomų užduočių, veiklų ar detalių (pvz., banko išrašų, "
        "naujienų laidų ar kitų dalykų, kurių sąraše nėra). Jeigu nesi tikras — "
        "geriau apskritai apie tai nekalbėk.\n\n"
        "Ataskaita turi:\n"
        "- Pagirti mokinį už pastangas ir kantrybę\n"
        "- Paminėti konkrečiai, ką jie darė šiose užduotyse (pagal sąrašą)\n"
        "- Jeigu yra mokinio žinutė — pakomentuoti ją asmeniškai\n"
        "- Trumpai paskatinti būti budriems ateityje\n"
        "- Padėkoti už dalyvavimą\n\n"
        "Nerašyk pavadinimo ar antraštės. Tik tekstą."
    )

    messages = [
        {"role": "user", "content": (
            f"Mokinys atliko šias užduotis:\n{summary_text}"
            f"{personal_section}\n\n"
            f"Parašyk asmeninę ataskaitą."
        )},
    ]

    model_config = resolve_tier("fast")
    settings = get_settings()
    provider = create_provider(model_config, settings)

    start_time = time.monotonic()
    report_text, usage = await provider.complete(
        system_prompt=system_prompt,
        messages=messages,
        model_config=model_config,
        tools=None,
    )
    elapsed_ms = (time.monotonic() - start_time) * 1000

    log_ai_call(
        call_type="session_report",
        model_id=model_config.model_id,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        latency_ms=elapsed_ms,
        task_id="session_report",
        session_id=session_id,
    )

    # Telemetry: save session completion with report
    try:
        from backend.telemetry import save_session_end
        save_session_end(session, report_text.strip())
    except Exception as exc:
        logger.warning("Telemetry session end save failed: %s", exc)

    return ApiResponse(
        ok=True,
        data={"report": report_text.strip()},
    ).model_dump()


# ---------------------------------------------------------------------------
# GET /dump-sessions — admin: dump all active sessions to disk
# ---------------------------------------------------------------------------


@router.get("/dump-sessions")
async def dump_sessions(
    session_store: SessionStore = Depends(get_session_store),
) -> dict:
    """Dumps all active sessions to data/sessions/ for telemetry.

    Call this after each class to capture data from students who
    didn't finish. No auth required — this is a dev/admin tool.
    """
    from backend.telemetry import save_active_session

    sessions = session_store.get_all_sessions()
    count = 0
    for session in sessions:
        try:
            save_active_session(session)
            count += 1
        except Exception as exc:
            logger.warning("Failed to dump session %s: %s", session.session_id, exc)

    return ApiResponse(
        ok=True,
        data={"dumped": count, "message": f"Dumped {count} active sessions to data/sessions/"},
    ).model_dump()


# ---------------------------------------------------------------------------
# GET /download-sessions — admin: download all session telemetry as JSON
# ---------------------------------------------------------------------------


@router.get("/download-sessions")
async def download_sessions(
    session_store: SessionStore = Depends(get_session_store),
) -> dict:
    """Downloads all session telemetry as a single JSON payload.

    First dumps all active in-memory sessions to disk, then reads
    every JSON file from data/sessions/ and returns them. Hit this
    endpoint before redeploying Railway to preserve student data.
    """
    import json
    from pathlib import Path
    from backend.telemetry import save_active_session, DATA_DIR

    # Step 1: dump active sessions so nothing is lost
    for session in session_store.get_all_sessions():
        try:
            save_active_session(session)
        except Exception:
            pass

    # Step 2: read all session files
    sessions = []
    if DATA_DIR.exists():
        for path in sorted(DATA_DIR.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                data["_filename"] = path.name
                sessions.append(data)
            except (json.JSONDecodeError, OSError):
                pass

    return ApiResponse(
        ok=True,
        data={"session_count": len(sessions), "sessions": sessions},
    ).model_dump()
