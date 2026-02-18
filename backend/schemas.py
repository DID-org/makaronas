"""Core data models — shared Pydantic types for the Makaronas platform.

Every conversation, learning profile, API response, and SSE event flows through
these types. They are the shared vocabulary that lets dozens of components talk
without ambiguity.

This is a Tier 1 leaf module: it imports only from pydantic and the stdlib.
No project imports allowed — everything else imports from here.

Usage:
    from backend.schemas import User, StudentProfile, GameSession, ApiResponse
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Helper models
# ---------------------------------------------------------------------------


class TechniqueStats(BaseModel):
    """Recognition statistics for a single manipulation technique."""

    caught: int = 0
    total: int = 0


class Exchange(BaseModel):
    """One conversation turn between student and Trickster."""

    model_config = ConfigDict(frozen=True)

    role: Literal["student", "trickster"]
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


class User(BaseModel):
    """Identity model returned by the auth layer.

    Frozen — users are identity objects, no mutation after creation.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    role: Literal["student", "teacher", "admin"]
    name: str
    school_id: str


# ---------------------------------------------------------------------------
# Student data (Framework Principle 3 — Sacred Trust)
# ---------------------------------------------------------------------------


class StudentProfile(BaseModel):
    """Persistent learning profile — statistical summaries only, never raw text.

    Stored across sessions via DatabaseAdapter. Contains trigger vulnerability
    scores, technique recognition rates, and engagement/growth data. The
    dict[str, Any] fields are deliberately unstructured in V1 — V6 (Evaluation)
    will define their internal shapes.

    Mutable: the evaluation engine updates fields after each session.
    """

    student_id: str
    school_id: str
    trigger_vulnerability: dict[str, float] = Field(default_factory=dict)
    technique_recognition: dict[str, TechniqueStats] = Field(default_factory=dict)
    engagement_signals: dict[str, Any] = Field(default_factory=dict)
    growth_trajectory: dict[str, Any] = Field(default_factory=dict)
    sessions_completed: int = 0
    last_active: datetime | None = None
    tasks_completed: list[str] = Field(default_factory=list)


class GameSession(BaseModel):
    """Ephemeral game state for one playthrough (24h TTL).

    Lives in SessionStore, separate from DatabaseAdapter. Contains raw student
    responses — this is why it has a short lifetime and is kept apart from the
    persistent learning profile.

    Mutable: updated on every student interaction.
    """

    session_id: str
    student_id: str
    school_id: str
    language: str = "lt"
    roadmap_id: str | None = None
    current_task: str | None = None
    exchanges: list[Exchange] = Field(default_factory=list)
    choices: list[dict[str, Any]] = Field(default_factory=list)
    checklist_progress: dict[str, Any] = Field(default_factory=dict)
    investigation_paths: list[str] = Field(default_factory=list)
    raw_performance: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc) + timedelta(hours=24),
    )


# ---------------------------------------------------------------------------
# Teacher insights
# ---------------------------------------------------------------------------


class ClassInsights(BaseModel):
    """Aggregated, anonymous class-level patterns.

    Returned by DatabaseAdapter.get_class_insights(). Never contains
    individual student data. Frozen — immutable snapshot.
    """

    model_config = ConfigDict(frozen=True)

    class_id: str
    school_id: str
    trigger_distribution: dict[str, float] = Field(default_factory=dict)
    common_failure_points: list[str] = Field(default_factory=list)
    growth_trends: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# API envelope
# ---------------------------------------------------------------------------


class ApiError(BaseModel):
    """Error detail inside ApiResponse.error.

    code is an uppercase string like "TASK_NOT_FOUND", "SESSION_EXPIRED",
    "AI_TIMEOUT". Not an enum — error codes grow across visions.
    """

    model_config = ConfigDict(frozen=True)

    code: str
    message: str


class ApiResponse(BaseModel):
    """Universal response envelope — every API endpoint returns this shape."""

    model_config = ConfigDict(frozen=True)

    ok: bool
    data: Any | None = None
    error: ApiError | None = None


# ---------------------------------------------------------------------------
# SSE events (Phase 3b streaming infrastructure)
# ---------------------------------------------------------------------------


class TokenEvent(BaseModel):
    """Incremental text token from AI streaming."""

    model_config = ConfigDict(frozen=True)

    text: str


class DoneEvent(BaseModel):
    """Stream completion with full text and optional structured data."""

    model_config = ConfigDict(frozen=True)

    full_text: str
    data: dict[str, Any] = Field(default_factory=dict)


class ErrorEvent(BaseModel):
    """Stream error with partial text recovery."""

    model_config = ConfigDict(frozen=True)

    code: str
    message: str
    partial_text: str = ""


# ---------------------------------------------------------------------------
# Content markers
# ---------------------------------------------------------------------------


class ContentBlock(BaseModel):
    """Content source marker — identifies whether content came from AI or static.

    model_family is None for static content, populated for AI-generated content
    (e.g. "claude", "gemini").
    """

    model_config = ConfigDict(frozen=True)

    source: Literal["ai", "static"]
    content: str
    model_family: str | None = None
