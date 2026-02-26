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
"""

import logging
from collections.abc import AsyncIterator
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.api.deps import get_current_user, get_database, get_session_store, get_task_registry
from backend.config import get_settings
from backend.hooks.interfaces import DatabaseAdapter, SessionStore
from backend.schemas import ApiError, ApiResponse, GameSession, User
from backend.streaming import create_sse_response, stream_ai_response
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
# Stub token generators
# ---------------------------------------------------------------------------


async def _trickster_tokens() -> AsyncIterator[str]:
    """Stub token generator simulating a Trickster response."""
    for token in ["The ", "Trickster ", "is ", "watching... "]:
        yield token


async def _debrief_tokens() -> AsyncIterator[str]:
    """Stub token generator simulating a debrief response."""
    for token in ["You ", "did ", "well ", "today. "]:
        yield token


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
):
    """Accepts a student response and streams a Trickster reply via SSE.

    Validation and ownership checks happen before streaming begins —
    once SSE starts, HTTP status is locked at 200.
    """
    session = await _get_session_or_404(session_id, session_store)
    _check_ownership(session, user)

    generator = stream_ai_response(
        _trickster_tokens(),
        done_data={"action_received": body.action},
    )
    return create_sse_response(generator)


@router.get("/session/{session_id}/debrief")
async def debrief(
    session_id: str,
    user: User = Depends(get_current_user),
    session_store: SessionStore = Depends(get_session_store),
):
    """Streams a debrief summary via SSE (stub).

    Validation and ownership checks happen before streaming begins.
    """
    session = await _get_session_or_404(session_id, session_store)
    _check_ownership(session, user)

    generator = stream_ai_response(
        _debrief_tokens(),
        done_data={"tasks_completed": 1},
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
