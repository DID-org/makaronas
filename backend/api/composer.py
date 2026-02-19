"""Composer AI routes and asset serving — teacher curriculum collaboration.

Four endpoints split across two routers:
- Composer (teacher/admin only): chat (SSE), roadmap generate, roadmap refine
- Assets (any authenticated user): static file serving for task content

The Composer is the teacher's AI collaborator — it helps build curriculum by
suggesting tasks, generating roadmaps, and refining them through dialogue.
In V1 these are stubs returning hardcoded data; the real AI integration
comes in V8.

The asset route serves images, audio, and other task content files from disk,
backing the URLs that LocalFileStorage.get_asset_url() produces.

Tier 2 service module: imports from deps (Tier 2), schemas (Tier 1),
streaming (Tier 2).

Created: Phase 4c
"""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from starlette.responses import FileResponse

from backend.api.deps import get_current_user, get_file_storage
from backend.hooks.interfaces import FileStorage
from backend.schemas import ApiError, ApiResponse, User
from backend.streaming import create_sse_response, stream_ai_response

router = APIRouter()
asset_router = APIRouter()


# ---------------------------------------------------------------------------
# Request bodies (API-boundary types, local to this module)
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """Request body for POST /chat."""

    message: str
    conversation_id: str | None = None


class GenerateRoadmapRequest(BaseModel):
    """Request body for POST /roadmap/generate."""

    description: str
    constraints: dict[str, Any] | None = None


class RefineRoadmapRequest(BaseModel):
    """Request body for POST /roadmap/refine."""

    roadmap_id: str
    instruction: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_teacher(user: User) -> None:
    """Verifies the authenticated user has teacher or admin role.

    Raises 403 with FORBIDDEN error code if the user is a student.
    Students have zero access to composer endpoints.
    """
    if user.role not in ("teacher", "admin"):
        raise HTTPException(
            status_code=403,
            detail=ApiResponse(
                ok=False,
                error=ApiError(
                    code="FORBIDDEN",
                    message="Teacher or admin role required.",
                ),
            ).model_dump(),
        )


def _validate_asset_path(task_id: str, filename: str, base_path: Path) -> Path:
    """Validates and resolves an asset path, guarding against traversal attacks.

    Framework Principle 13 (Security by Design): reject any path component
    that could escape the asset root. Defense in depth: check for suspicious
    characters AND verify the resolved path is within bounds.

    Args:
        task_id: The task identifier from the URL path.
        filename: The asset filename from the URL path.
        base_path: The root directory for task assets.

    Returns:
        The resolved, validated filesystem path.

    Raises:
        HTTPException: 400 with BAD_REQUEST if the path is invalid.
    """
    for component in (task_id, filename):
        if ".." in component or "/" in component or "\\" in component:
            raise HTTPException(
                status_code=400,
                detail=ApiResponse(
                    ok=False,
                    error=ApiError(
                        code="BAD_REQUEST",
                        message="Invalid path component.",
                    ),
                ).model_dump(),
            )

    file_path = (base_path / task_id / filename).resolve()

    if not file_path.is_relative_to(base_path.resolve()):
        raise HTTPException(
            status_code=400,
            detail=ApiResponse(
                ok=False,
                error=ApiError(
                    code="BAD_REQUEST",
                    message="Invalid path component.",
                ),
            ).model_dump(),
        )

    return file_path


# ---------------------------------------------------------------------------
# Stub token generator
# ---------------------------------------------------------------------------


async def _composer_tokens() -> AsyncIterator[str]:
    """Stub token generator simulating a Composer AI response."""
    for token in [
        "I'd ",
        "suggest ",
        "a ",
        "three-task ",
        "sequence ",
        "on ",
        "urgency ",
        "triggers. ",
    ]:
        yield token


# ---------------------------------------------------------------------------
# Composer endpoints (teacher/admin only)
# ---------------------------------------------------------------------------


@router.post("/chat")
async def composer_chat(
    body: ChatRequest,
    user: User = Depends(get_current_user),
):
    """Streams a Composer AI reply via SSE (stub).

    Validates auth and role before streaming begins — once SSE starts,
    HTTP status is locked at 200. The conversation_id is echoed back
    (or generated if not provided) in the done event.
    """
    _require_teacher(user)

    conversation_id = body.conversation_id or str(uuid4())

    generator = stream_ai_response(
        _composer_tokens(),
        done_data={"conversation_id": conversation_id},
    )
    return create_sse_response(generator)


@router.post("/roadmap/generate")
async def generate_roadmap(
    body: GenerateRoadmapRequest,
    user: User = Depends(get_current_user),
) -> dict:
    """Generates a proposed roadmap from a description (stub).

    Returns a hardcoded proposed roadmap regardless of input.
    Real AI-powered generation comes in V8.
    """
    _require_teacher(user)

    return ApiResponse(
        ok=True,
        data={
            "roadmap_id": str(uuid4()),
            "title": "Urgency & Social Pressure",
            "tasks": [
                {
                    "task_id": "task-urgency-article-001",
                    "title": "The Midnight Deadline",
                    "time_minutes": 10,
                },
                {
                    "task_id": "task-belonging-chat-002",
                    "title": "Everyone Already Knows",
                    "time_minutes": 15,
                },
                {
                    "task_id": "task-injustice-meme-003",
                    "title": "They Don't Want You to See This",
                    "time_minutes": 12,
                },
            ],
            "reasoning": (
                "This sequence builds from recognising urgency in news "
                "to identifying social proof in peer conversations, "
                "then challenges students with visual manipulation."
            ),
        },
    ).model_dump()


@router.post("/roadmap/refine")
async def refine_roadmap(
    body: RefineRoadmapRequest,
    user: User = Depends(get_current_user),
) -> dict:
    """Refines an existing roadmap based on teacher instruction (stub).

    Returns a hardcoded refined roadmap regardless of input.
    Real AI-powered refinement comes in V8.
    """
    _require_teacher(user)

    return ApiResponse(
        ok=True,
        data={
            "roadmap_id": body.roadmap_id,
            "title": "Urgency & Social Pressure (Shortened)",
            "tasks": [
                {
                    "task_id": "task-urgency-article-001",
                    "title": "The Midnight Deadline",
                    "time_minutes": 10,
                },
                {
                    "task_id": "task-belonging-chat-002",
                    "title": "Everyone Already Knows",
                    "time_minutes": 15,
                },
            ],
            "changes": (
                "Removed the advanced meme task to fit within "
                "the requested 25-minute window."
            ),
        },
    ).model_dump()


# ---------------------------------------------------------------------------
# Asset endpoint (any authenticated user)
# ---------------------------------------------------------------------------


@asset_router.get("/{task_id}/{filename}")
async def serve_asset(
    task_id: str,
    filename: str,
    user: User = Depends(get_current_user),
    storage: FileStorage = Depends(get_file_storage),
) -> FileResponse:
    """Serves a static asset file for a task (image, audio, etc.).

    Accessible by any authenticated user — students need task images/audio,
    teachers need preview assets.

    Path traversal protection is enforced before any filesystem access
    (Framework Principle 13).
    """
    # V1 stub coupling: access _base_path from LocalFileStorage.
    # When the team swaps to CDN storage, this route becomes unnecessary.
    base_path = Path(storage._base_path)

    file_path = _validate_asset_path(task_id, filename, base_path)

    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=ApiResponse(
                ok=False,
                error=ApiError(
                    code="ASSET_NOT_FOUND",
                    message=f"Asset '{filename}' not found for task '{task_id}'.",
                ),
            ).model_dump(),
        )

    return FileResponse(file_path)
