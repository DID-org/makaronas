"""Teacher-facing API routes — task library, roadmaps, class insights.

Five endpoints that form the teacher's window into Makaronas:
- Library: browse task summaries with filters, view full task detail
- Roadmaps: list available roadmaps, create custom roadmaps
- Insights: anonymous class-level aggregated patterns

All responses use the ApiResponse envelope. No SSE streaming — all JSON.
Auth is enforced on every endpoint via get_current_user dependency.
Teacher/admin role required on all endpoints.

Tier 2 service module: imports from deps (Tier 2), schemas (Tier 1).

Created: Phase 4b
"""

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.api.deps import get_current_user, get_database
from backend.hooks.interfaces import DatabaseAdapter
from backend.schemas import ApiError, ApiResponse, User

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
# Stub data
# ---------------------------------------------------------------------------

_STUB_TASKS = [
    {
        "task_id": "task-urgency-article-001",
        "title": "The Midnight Deadline",
        "trigger": "urgency",
        "technique": "manufactured_deadline",
        "medium": "article",
        "difficulty": "beginner",
        "time_minutes": 10,
        "tags": ["urgency", "news", "pressure"],
    },
    {
        "task_id": "task-belonging-chat-002",
        "title": "Everyone Already Knows",
        "trigger": "belonging",
        "technique": "bandwagon",
        "medium": "group_chat",
        "difficulty": "intermediate",
        "time_minutes": 15,
        "tags": ["belonging", "social_proof", "peer_pressure"],
    },
    {
        "task_id": "task-injustice-meme-003",
        "title": "They Don't Want You to See This",
        "trigger": "injustice",
        "technique": "cherry_picked_data",
        "medium": "meme",
        "difficulty": "advanced",
        "time_minutes": 12,
        "tags": ["injustice", "visual", "statistics"],
    },
]

_STUB_TASK_DETAIL = {
    **_STUB_TASKS[0],
    "description": (
        "A news article uses a fabricated deadline to pressure readers "
        "into sharing before verifying. Students must identify the "
        "urgency trigger and explain why the deadline is manufactured."
    ),
    "content_preview": (
        "BREAKING: New regulation takes effect at midnight — "
        "share this before it's too late..."
    ),
    "learning_objectives": [
        "Recognise manufactured urgency in news headlines",
        "Distinguish real deadlines from artificial pressure",
        "Practice pausing before sharing time-sensitive content",
    ],
}

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
    difficulty: str | None = None,
    time_max: int | None = None,
    tags: str | None = None,
    search: str | None = None,
    user: User = Depends(get_current_user),
) -> dict:
    """Browses the task library with optional filters (stub).

    All query parameters are accepted and documented in OpenAPI but
    currently ignored — the stub returns the same hardcoded list
    regardless. Real filtering comes in V2/V10.
    """
    _require_teacher(user)

    return ApiResponse(
        ok=True,
        data={"tasks": _STUB_TASKS},
    ).model_dump()


@router.get("/library/{task_id}")
async def get_task_detail(
    task_id: str,
    user: User = Depends(get_current_user),
) -> dict:
    """Returns full detail for a single task (stub).

    Returns the same hardcoded task detail regardless of task_id.
    Real task lookup comes in V2/V10.
    """
    _require_teacher(user)

    return ApiResponse(
        ok=True,
        data=_STUB_TASK_DETAIL,
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
