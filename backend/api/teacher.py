"""Teacher-facing API routes — task library, roadmaps, class insights.

Five endpoints that form the teacher's window into Makaronas:
- Library: browse task summaries with filters, view full task detail
- Roadmaps: list available roadmaps, create custom roadmaps
- Insights: anonymous class-level aggregated patterns

All responses use the ApiResponse envelope. No SSE streaming — all JSON.
Auth is enforced on every endpoint via get_current_user dependency.
Teacher/admin role required on all endpoints.

Tier 3 orchestration module: imports from deps (Tier 2-3), schemas (Tier 1).

Created: Phase 4b
Updated: Phase 4a — replaced library stubs with registry-backed queries
"""

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.api.deps import get_current_user, get_database, get_task_registry
from backend.hooks.interfaces import DatabaseAdapter
from backend.schemas import ApiError, ApiResponse, User
from backend.tasks.registry import TaskRegistry
from backend.tasks.schemas import (
    ChatMessageBlock,
    MemeBlock,
    SearchResultBlock,
    SocialPostBlock,
    TaskCartridge,
    TextBlock,
    VideoTranscriptBlock,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request bodies (API-boundary types, local to this module)
# ---------------------------------------------------------------------------


class CreateRoadmapRequest(BaseModel):
    """Request body for POST /roadmaps."""

    title: str
    task_ids: list[str]
    notes: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_teacher(user: User) -> None:
    """Verifies the authenticated user has teacher or admin role.

    Raises 403 with FORBIDDEN error code if the user is a student.
    Students have zero access to teacher endpoints.
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


# ---------------------------------------------------------------------------
# Cartridge → API response helpers
# ---------------------------------------------------------------------------


def _derive_content_preview(cartridge: TaskCartridge) -> str:
    """Derives a text preview from the first text-bearing presentation block.

    Iterates blocks in order, extracts text from known text-bearing types.
    Skips ImageBlock, AudioBlock, GenericBlock (no useful text content).
    Truncates to 200 characters with "..." suffix when needed.

    Returns:
        Preview text, or empty string if no text-bearing block exists.
    """
    for block in cartridge.presentation_blocks:
        text: str | None = None
        if isinstance(block, (TextBlock, SocialPostBlock, ChatMessageBlock)):
            text = block.text
        elif isinstance(block, VideoTranscriptBlock):
            text = block.transcript
        elif isinstance(block, SearchResultBlock):
            text = block.snippet
        elif isinstance(block, MemeBlock):
            parts = []
            if block.top_text:
                parts.append(block.top_text)
            if block.bottom_text:
                parts.append(block.bottom_text)
            text = " ".join(parts) if parts else None

        if text:
            if len(text) > 200:
                return text[:200] + "..."
            return text
    return ""


def _cartridge_to_summary(cartridge: TaskCartridge) -> dict:
    """Converts a cartridge to the summary dict for list responses."""
    return {
        "task_id": cartridge.task_id,
        "title": cartridge.title,
        "trigger": cartridge.trigger,
        "technique": cartridge.technique,
        "medium": cartridge.medium,
        "difficulty": cartridge.difficulty,
        "time_minutes": cartridge.time_minutes,
        "tags": list(cartridge.tags),
        "status": cartridge.status,
        "task_type": cartridge.task_type,
    }


def _cartridge_to_detail(cartridge: TaskCartridge) -> dict:
    """Converts a cartridge to the detail dict for single-task responses."""
    detail = _cartridge_to_summary(cartridge)
    detail.update({
        "description": cartridge.description,
        "content_preview": _derive_content_preview(cartridge),
        "learning_objectives": list(cartridge.learning_objectives),
        "is_clean": cartridge.is_clean,
        "version": cartridge.version,
        "phase_count": len(cartridge.phases),
        "is_evergreen": cartridge.is_evergreen,
    })
    return detail


# ---------------------------------------------------------------------------
# Stub data (roadmaps only — V9 scope)
# ---------------------------------------------------------------------------

_STUB_ROADMAPS = [
    {
        "roadmap_id": "roadmap-intro-sequence",
        "title": "Introduction to Media Literacy",
        "source": "prebuilt",
        "task_count": 5,
        "description": "A gentle introduction covering the three core triggers.",
    },
    {
        "roadmap_id": "roadmap-advanced-tactics",
        "title": "Advanced Manipulation Tactics",
        "source": "prebuilt",
        "task_count": 8,
        "description": "Deep dive into cherry-picked data, fabricated quotes, and structural bias.",
    },
]


# ---------------------------------------------------------------------------
# Library endpoints
# ---------------------------------------------------------------------------


@router.get("/library")
async def list_library(
    trigger: str | None = None,
    technique: str | None = None,
    medium: str | None = None,
    difficulty: int | None = None,
    time_max: int | None = None,
    tags: str | None = None,
    search: str | None = None,  # Accepted but ignored — full-text search is V10 scope
    status: str = "active",
    limit: int = 50,
    offset: int = 0,
    user: User = Depends(get_current_user),
    registry: TaskRegistry = Depends(get_task_registry),
) -> dict:
    """Browses the task library with optional filters.

    Returns real cartridge data from the registry. All query parameters
    filter results using AND logic. ``difficulty`` is int (1-5), breaking
    from V1 stub's string representation.

    When ``time_max`` is specified, the page may contain fewer than
    ``limit`` results because time_max is a post-filter applied after
    the registry's indexed query.
    """
    _require_teacher(user)

    # Draft access: explicit check documenting intent for future student endpoints
    if status in ("draft", "all") and user.role not in ("teacher", "admin"):
        raise HTTPException(
            status_code=403,
            detail=ApiResponse(
                ok=False,
                error=ApiError(
                    code="FORBIDDEN",
                    message="Draft access requires teacher or admin role.",
                ),
            ).model_dump(),
        )

    # Parse tags from comma-separated string
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None

    # Build query — difficulty as exact match (both min and max)
    query_kwargs: dict = {
        "trigger": trigger,
        "technique": technique,
        "medium": medium,
        "tags": tag_list,
        "status": status,
    }
    if difficulty is not None:
        query_kwargs["difficulty_min"] = difficulty
        query_kwargs["difficulty_max"] = difficulty

    # Fetch ALL matching results (no pagination) for total count + time_max
    all_results = registry.query(**query_kwargs, limit=999_999, offset=0)

    # Post-filter: time_max (not indexed by registry)
    if time_max is not None:
        all_results = [c for c in all_results if c.time_minutes <= time_max]

    total = len(all_results)

    # Apply pagination
    page = all_results[offset:offset + limit]

    return ApiResponse(
        ok=True,
        data={
            "tasks": [_cartridge_to_summary(c) for c in page],
            "total": total,
        },
    ).model_dump()


@router.get("/library/{task_id}")
async def get_task_detail(
    task_id: str,
    include_drafts: bool = False,
    user: User = Depends(get_current_user),
    registry: TaskRegistry = Depends(get_task_registry),
) -> dict:
    """Returns full detail for a single task from the registry.

    Returns 404 for unknown tasks and for draft tasks when
    ``include_drafts`` is False (does not leak draft existence).
    """
    _require_teacher(user)

    cartridge = registry.get_task(task_id)

    # 404 for missing tasks AND for draft tasks when include_drafts is False
    if cartridge is None or (cartridge.status == "draft" and not include_drafts):
        raise HTTPException(
            status_code=404,
            detail=ApiResponse(
                ok=False,
                error=ApiError(
                    code="TASK_NOT_FOUND",
                    message=f"Task '{task_id}' not found.",
                ),
            ).model_dump(),
        )

    return ApiResponse(
        ok=True,
        data=_cartridge_to_detail(cartridge),
    ).model_dump()


# ---------------------------------------------------------------------------
# Roadmap endpoints
# ---------------------------------------------------------------------------


@router.get("/roadmaps")
async def list_roadmaps(
    user: User = Depends(get_current_user),
) -> dict:
    """Lists available roadmaps — pre-built and custom (stub).

    Returns hardcoded stub roadmaps. Real roadmap persistence
    comes in V9.
    """
    _require_teacher(user)

    return ApiResponse(
        ok=True,
        data={"roadmaps": _STUB_ROADMAPS},
    ).model_dump()


@router.post("/roadmaps")
async def create_roadmap(
    body: CreateRoadmapRequest,
    user: User = Depends(get_current_user),
) -> dict:
    """Creates a custom roadmap (stub — no persistence).

    Validates the request body and returns a confirmation with a
    generated UUID. Real roadmap storage comes in V9.
    """
    _require_teacher(user)

    return ApiResponse(
        ok=True,
        data={
            "roadmap_id": str(uuid4()),
            "title": body.title,
            "task_count": len(body.task_ids),
            "source": "custom",
        },
    ).model_dump()


# ---------------------------------------------------------------------------
# Class insights endpoint
# ---------------------------------------------------------------------------


@router.get("/class/{class_id}/insights")
async def class_insights(
    class_id: str,
    user: User = Depends(get_current_user),
    database: DatabaseAdapter = Depends(get_database),
) -> dict:
    """Returns anonymous class-level aggregated patterns.

    Uses the real DatabaseAdapter path — school_id comes from the
    authenticated teacher's profile for multi-tenant scoping.
    Returns 404 if no insights exist for the given class.
    """
    _require_teacher(user)

    insights = await database.get_class_insights(class_id, user.school_id)
    if insights is None:
        raise HTTPException(
            status_code=404,
            detail=ApiResponse(
                ok=False,
                error=ApiError(
                    code="CLASS_NOT_FOUND",
                    message=f"No insights found for class '{class_id}'.",
                ),
            ).model_dump(),
        )

    return ApiResponse(
        ok=True,
        data={
            "class_id": insights.class_id,
            "school_id": insights.school_id,
            "trigger_distribution": insights.trigger_distribution,
            "common_failure_points": insights.common_failure_points,
            "growth_trends": insights.growth_trends,
        },
    ).model_dump()
