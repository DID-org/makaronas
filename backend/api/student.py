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
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.ai.trickster import DebriefResult, TricksterEngine, TricksterResult
from backend.ai.usage import log_ai_call
from backend.api.deps import (
    check_ai_readiness,
    get_current_user,
    get_database,
    get_session_store,
    get_task_registry,
    get_trickster_engine,
)
from backend.config import get_settings
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


async def _stream_trickster_response(
    result: TricksterResult | DebriefResult,
    session: GameSession,
    session_store: SessionStore,
    cartridge: TaskCartridge,
    call_type: str,
    timeout_seconds: float = 30.0,
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
    user: User = Depends(get_current_user),
    session_store: SessionStore = Depends(get_session_store),
) -> dict:
    """Creates a new game session.

    Returns the session_id and a stub intro message. The language defaults
    to the platform's configured default if not specified.
    """
    settings = get_settings()
    language = body.language or settings.default_language

    session = GameSession(
        session_id=str(uuid4()),
        student_id=user.id,
        school_id=user.school_id,
        language=language,
        roadmap_id=body.roadmap_id,
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

    # --- Derive response from initial phase ---
    initial_phase = _find_initial_phase(cartridge)

    # --- Update session ---
    session.current_task = resolved_task_id
    session.current_phase = cartridge.initial_phase
    await session_store.save_session(session)

    return ApiResponse(
        ok=True,
        data={
            "task_id": cartridge.task_id,
            "task_type": cartridge.task_type,
            "medium": cartridge.medium,
            "title": cartridge.title,
            "content": _derive_content_blocks(cartridge, initial_phase),
            "available_actions": _derive_available_actions(initial_phase),
            "trickster_intro": _derive_trickster_intro(initial_phase),
            "current_phase": cartridge.initial_phase,
        },
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
