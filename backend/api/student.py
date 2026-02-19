"""Student-facing API routes — sessions, responses, profiles, GDPR.

Seven endpoints that form the student's entry point to Makaronas:
- Session lifecycle: create, next task, respond (SSE), debrief (SSE)
- Profile: radar view
- GDPR: deletion, export

All responses use the ApiResponse envelope. Streaming endpoints use the
Phase 3b SSE infrastructure. Auth is enforced on every endpoint via
get_current_user dependency.

Tier 2 service module: imports from deps (Tier 2), schemas (Tier 1),
streaming (Tier 2), config (Tier 2).

Created: Phase 4a
"""

from collections.abc import AsyncIterator
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.api.deps import get_current_user, get_database, get_session_store
from backend.config import get_settings
from backend.hooks.interfaces import DatabaseAdapter, SessionStore
from backend.schemas import ApiError, ApiResponse, GameSession, User
from backend.streaming import create_sse_response, stream_ai_response

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

    action: Literal["button_click", "freeform", "investigate"]
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
    user: User = Depends(get_current_user),
    session_store: SessionStore = Depends(get_session_store),
) -> dict:
    """Returns the next task content for the session (stub)."""
    session = await _get_session_or_404(session_id, session_store)
    _check_ownership(session, user)

    return ApiResponse(
        ok=True,
        data={
            "task_id": "stub-task-001",
            "task_type": "stub",
            "medium": "article",
            "content": {
                "headline": "Scientists Confirm: Coffee Grants Immortality",
                "body": "A groundbreaking study by the Institute of Wishful Thinking...",
            },
            "available_actions": ["freeform", "button_click", "investigate"],
            "trickster_intro": "Interesting article, isn't it? What do you think?",
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
