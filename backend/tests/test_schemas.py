"""Tests for backend.schemas — Core Pydantic data models."""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from backend.schemas import (
    ApiError,
    ApiResponse,
    ClassInsights,
    ContentBlock,
    DoneEvent,
    ErrorEvent,
    Exchange,
    GameSession,
    StudentProfile,
    TechniqueStats,
    TokenEvent,
    User,
)


# ---------------------------------------------------------------------------
# User
# ---------------------------------------------------------------------------


class TestUser:
    """User identity model — frozen, Literal role validation."""

    def test_valid_student(self) -> None:
        u = User(id="s1", role="student", name="Jonas", school_id="school-1")
        assert u.id == "s1"
        assert u.role == "student"
        assert u.name == "Jonas"
        assert u.school_id == "school-1"

    def test_valid_teacher(self) -> None:
        u = User(id="t1", role="teacher", name="Mokytojas", school_id="school-1")
        assert u.role == "teacher"

    def test_valid_admin(self) -> None:
        u = User(id="a1", role="admin", name="Admin", school_id="school-1")
        assert u.role == "admin"

    def test_invalid_role_rejected(self) -> None:
        with pytest.raises(ValidationError, match="role"):
            User(id="s1", role="superadmin", name="Test", school_id="school-1")

    def test_missing_required_field(self) -> None:
        with pytest.raises(ValidationError):
            User(id="s1", role="student", name="Test")  # type: ignore[call-arg]

    def test_frozen(self) -> None:
        u = User(id="s1", role="student", name="Test", school_id="school-1")
        with pytest.raises(ValidationError):
            u.name = "Changed"  # type: ignore[misc]

    def test_serialization_roundtrip(self) -> None:
        u = User(id="s1", role="student", name="Jonas", school_id="school-1")
        assert User.model_validate(u.model_dump()) == u


# ---------------------------------------------------------------------------
# TechniqueStats
# ---------------------------------------------------------------------------


class TestTechniqueStats:
    """Helper model for technique recognition tracking."""

    def test_defaults(self) -> None:
        ts = TechniqueStats()
        assert ts.caught == 0
        assert ts.total == 0

    def test_custom_values(self) -> None:
        ts = TechniqueStats(caught=5, total=10)
        assert ts.caught == 5
        assert ts.total == 10

    def test_serialization_roundtrip(self) -> None:
        ts = TechniqueStats(caught=3, total=7)
        assert TechniqueStats.model_validate(ts.model_dump()) == ts


# ---------------------------------------------------------------------------
# Exchange
# ---------------------------------------------------------------------------


class TestExchange:
    """Conversation turn model — frozen, Literal role, auto-timestamp."""

    def test_student_role(self) -> None:
        e = Exchange(role="student", content="I think this is fake")
        assert e.role == "student"
        assert e.content == "I think this is fake"

    def test_trickster_role(self) -> None:
        e = Exchange(role="trickster", content="But the source is real...")
        assert e.role == "trickster"

    def test_invalid_role_rejected(self) -> None:
        with pytest.raises(ValidationError, match="role"):
            Exchange(role="teacher", content="test")

    def test_auto_timestamp(self) -> None:
        before = datetime.now(timezone.utc)
        e = Exchange(role="student", content="test")
        after = datetime.now(timezone.utc)
        assert before <= e.timestamp <= after

    def test_explicit_timestamp(self) -> None:
        ts = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        e = Exchange(role="student", content="test", timestamp=ts)
        assert e.timestamp == ts

    def test_frozen(self) -> None:
        e = Exchange(role="student", content="test")
        with pytest.raises(ValidationError):
            e.content = "changed"  # type: ignore[misc]

    def test_serialization_roundtrip(self) -> None:
        e = Exchange(role="trickster", content="Try again")
        assert Exchange.model_validate(e.model_dump()) == e


# ---------------------------------------------------------------------------
# StudentProfile
# ---------------------------------------------------------------------------


class TestStudentProfile:
    """Persistent learning profile — mutable, statistical summaries only."""

    def test_minimal_construction(self) -> None:
        p = StudentProfile(student_id="s1", school_id="school-1")
        assert p.student_id == "s1"
        assert p.school_id == "school-1"
        assert p.trigger_vulnerability == {}
        assert p.technique_recognition == {}
        assert p.engagement_signals == {}
        assert p.growth_trajectory == {}
        assert p.sessions_completed == 0
        assert p.last_active is None
        assert p.tasks_completed == []

    def test_populated_profile(self) -> None:
        p = StudentProfile(
            student_id="s1",
            school_id="school-1",
            trigger_vulnerability={"urgency": 0.7, "belonging": 0.3},
            technique_recognition={
                "cherry_picking": TechniqueStats(caught=3, total=5),
            },
            sessions_completed=4,
            last_active=datetime(2026, 2, 1, tzinfo=timezone.utc),
            tasks_completed=["task-1", "task-2"],
        )
        assert p.trigger_vulnerability["urgency"] == 0.7
        assert p.technique_recognition["cherry_picking"].caught == 3
        assert p.sessions_completed == 4
        assert len(p.tasks_completed) == 2

    def test_mutable(self) -> None:
        p = StudentProfile(student_id="s1", school_id="school-1")
        p.sessions_completed = 5
        assert p.sessions_completed == 5

    def test_serialization_roundtrip(self) -> None:
        p = StudentProfile(
            student_id="s1",
            school_id="school-1",
            trigger_vulnerability={"urgency": 0.7},
            technique_recognition={
                "cherry_picking": TechniqueStats(caught=3, total=5),
            },
            engagement_signals={"avg_response_time": 12.5},
            growth_trajectory={"week_1": [0.3, 0.5]},
            sessions_completed=4,
            last_active=datetime(2026, 2, 1, tzinfo=timezone.utc),
            tasks_completed=["task-1"],
        )
        dumped = p.model_dump()
        restored = StudentProfile.model_validate(dumped)
        assert restored == p

    def test_json_roundtrip_datetime(self) -> None:
        """Datetime fields serialize to ISO-8601 in JSON mode."""
        p = StudentProfile(
            student_id="s1",
            school_id="school-1",
            last_active=datetime(2026, 2, 1, 10, 30, 0, tzinfo=timezone.utc),
        )
        json_dict = p.model_dump(mode="json")
        assert isinstance(json_dict["last_active"], str)
        restored = StudentProfile.model_validate(json_dict)
        assert restored.last_active == p.last_active

    def test_none_last_active_roundtrip(self) -> None:
        p = StudentProfile(student_id="s1", school_id="school-1")
        dumped = p.model_dump(mode="json")
        assert dumped["last_active"] is None
        restored = StudentProfile.model_validate(dumped)
        assert restored.last_active is None

    def test_nested_technique_stats_roundtrip(self) -> None:
        """TechniqueStats inside technique_recognition dict roundtrips cleanly."""
        p = StudentProfile(
            student_id="s1",
            school_id="school-1",
            technique_recognition={
                "fake_quote": TechniqueStats(caught=2, total=8),
                "misleading_graph": TechniqueStats(caught=0, total=3),
            },
        )
        dumped = p.model_dump()
        restored = StudentProfile.model_validate(dumped)
        assert restored.technique_recognition["fake_quote"].caught == 2
        assert restored.technique_recognition["misleading_graph"].total == 3


# ---------------------------------------------------------------------------
# GameSession
# ---------------------------------------------------------------------------


class TestGameSession:
    """Ephemeral session state — mutable, 24h TTL, raw student data."""

    def test_minimal_construction(self) -> None:
        s = GameSession(session_id="sess-1", student_id="s1", school_id="school-1")
        assert s.session_id == "sess-1"
        assert s.language == "lt"
        assert s.roadmap_id is None
        assert s.current_task is None
        assert s.exchanges == []
        assert s.choices == []
        assert s.checklist_progress == {}
        assert s.investigation_paths == []
        assert s.raw_performance == {}

    def test_default_timestamps(self) -> None:
        before = datetime.now(timezone.utc)
        s = GameSession(session_id="sess-1", student_id="s1", school_id="school-1")
        after = datetime.now(timezone.utc)
        assert before <= s.created_at <= after
        # expires_at is ~24h after created_at
        expected_expiry = s.created_at + timedelta(hours=24)
        delta = abs((s.expires_at - expected_expiry).total_seconds())
        assert delta < 1.0  # within 1 second tolerance

    def test_with_exchanges(self) -> None:
        ex = Exchange(role="student", content="Is this real?")
        s = GameSession(
            session_id="sess-1",
            student_id="s1",
            school_id="school-1",
            exchanges=[ex],
        )
        assert len(s.exchanges) == 1
        assert s.exchanges[0].role == "student"

    def test_mutable(self) -> None:
        s = GameSession(session_id="sess-1", student_id="s1", school_id="school-1")
        s.current_task = "task-42"
        assert s.current_task == "task-42"

    def test_serialization_roundtrip(self) -> None:
        ts = datetime(2026, 2, 1, 10, 0, 0, tzinfo=timezone.utc)
        s = GameSession(
            session_id="sess-1",
            student_id="s1",
            school_id="school-1",
            language="en",
            roadmap_id="roadmap-1",
            current_task="task-3",
            exchanges=[
                Exchange(role="trickster", content="Hello!", timestamp=ts),
                Exchange(role="student", content="Hi", timestamp=ts),
            ],
            choices=[{"action": "click", "target": "option-a"}],
            checklist_progress={"item-1": True},
            investigation_paths=["path-a", "path-b"],
            raw_performance={"score": 0.85},
            created_at=ts,
            expires_at=ts + timedelta(hours=24),
        )
        dumped = s.model_dump()
        restored = GameSession.model_validate(dumped)
        assert restored == s

    def test_json_roundtrip_datetimes(self) -> None:
        """Datetime fields serialize to ISO-8601 in JSON mode and parse back."""
        s = GameSession(
            session_id="sess-1",
            student_id="s1",
            school_id="school-1",
            created_at=datetime(2026, 2, 1, 10, 0, 0, tzinfo=timezone.utc),
            expires_at=datetime(2026, 2, 2, 10, 0, 0, tzinfo=timezone.utc),
        )
        json_dict = s.model_dump(mode="json")
        assert isinstance(json_dict["created_at"], str)
        assert isinstance(json_dict["expires_at"], str)
        restored = GameSession.model_validate(json_dict)
        assert restored.created_at == s.created_at
        assert restored.expires_at == s.expires_at

    def test_nested_exchanges_json_roundtrip(self) -> None:
        ts = datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc)
        s = GameSession(
            session_id="sess-1",
            student_id="s1",
            school_id="school-1",
            exchanges=[Exchange(role="trickster", content="Think again", timestamp=ts)],
            created_at=ts,
            expires_at=ts + timedelta(hours=24),
        )
        json_dict = s.model_dump(mode="json")
        restored = GameSession.model_validate(json_dict)
        assert restored.exchanges[0].content == "Think again"
        assert restored.exchanges[0].timestamp == ts


# ---------------------------------------------------------------------------
# ClassInsights
# ---------------------------------------------------------------------------


class TestClassInsights:
    """Aggregated class-level patterns — frozen, no individual student data."""

    def test_construction(self) -> None:
        ci = ClassInsights(
            class_id="class-A",
            school_id="school-1",
            trigger_distribution={"urgency": 0.6, "belonging": 0.4},
            common_failure_points=["cherry_picking", "fake_quote"],
            growth_trends={"week_1": 0.3, "week_2": 0.5},
        )
        assert ci.class_id == "class-A"
        assert ci.trigger_distribution["urgency"] == 0.6
        assert len(ci.common_failure_points) == 2

    def test_defaults(self) -> None:
        ci = ClassInsights(class_id="class-A", school_id="school-1")
        assert ci.trigger_distribution == {}
        assert ci.common_failure_points == []
        assert ci.growth_trends == {}

    def test_frozen(self) -> None:
        ci = ClassInsights(class_id="class-A", school_id="school-1")
        with pytest.raises(ValidationError):
            ci.class_id = "changed"  # type: ignore[misc]

    def test_serialization_roundtrip(self) -> None:
        ci = ClassInsights(
            class_id="class-A",
            school_id="school-1",
            trigger_distribution={"urgency": 0.6},
            common_failure_points=["fake_quote"],
        )
        assert ClassInsights.model_validate(ci.model_dump()) == ci


# ---------------------------------------------------------------------------
# ApiError & ApiResponse
# ---------------------------------------------------------------------------


class TestApiError:
    """Error detail model — frozen."""

    def test_construction(self) -> None:
        e = ApiError(code="TASK_NOT_FOUND", message="No task with that ID")
        assert e.code == "TASK_NOT_FOUND"
        assert e.message == "No task with that ID"

    def test_frozen(self) -> None:
        e = ApiError(code="AI_TIMEOUT", message="Timed out")
        with pytest.raises(ValidationError):
            e.code = "CHANGED"  # type: ignore[misc]

    def test_serialization_roundtrip(self) -> None:
        e = ApiError(code="SESSION_EXPIRED", message="Session has expired")
        assert ApiError.model_validate(e.model_dump()) == e


class TestApiResponse:
    """Universal response envelope — frozen."""

    def test_success_response(self) -> None:
        r = ApiResponse(ok=True, data={"session_id": "abc"})
        assert r.ok is True
        assert r.data == {"session_id": "abc"}
        assert r.error is None

    def test_error_response(self) -> None:
        err = ApiError(code="TASK_NOT_FOUND", message="Not found")
        r = ApiResponse(ok=False, error=err)
        assert r.ok is False
        assert r.data is None
        assert r.error.code == "TASK_NOT_FOUND"

    def test_defaults(self) -> None:
        r = ApiResponse(ok=True)
        assert r.data is None
        assert r.error is None

    def test_frozen(self) -> None:
        r = ApiResponse(ok=True)
        with pytest.raises(ValidationError):
            r.ok = False  # type: ignore[misc]

    def test_serialization_roundtrip(self) -> None:
        err = ApiError(code="AI_TIMEOUT", message="Timed out")
        r = ApiResponse(ok=False, error=err)
        assert ApiResponse.model_validate(r.model_dump()) == r

    def test_nested_error_roundtrip(self) -> None:
        """ApiError inside ApiResponse survives model_dump -> model_validate."""
        err = ApiError(code="SESSION_EXPIRED", message="Gone")
        r = ApiResponse(ok=False, error=err)
        restored = ApiResponse.model_validate(r.model_dump())
        assert restored.error.code == "SESSION_EXPIRED"


# ---------------------------------------------------------------------------
# SSE Events
# ---------------------------------------------------------------------------


class TestTokenEvent:
    """Incremental text token — frozen."""

    def test_construction(self) -> None:
        e = TokenEvent(text="Hello")
        assert e.text == "Hello"

    def test_frozen(self) -> None:
        e = TokenEvent(text="Hello")
        with pytest.raises(ValidationError):
            e.text = "Changed"  # type: ignore[misc]

    def test_serialization_roundtrip(self) -> None:
        e = TokenEvent(text="partial response")
        assert TokenEvent.model_validate(e.model_dump()) == e


class TestDoneEvent:
    """Stream completion event — frozen."""

    def test_construction(self) -> None:
        e = DoneEvent(full_text="Complete response here")
        assert e.full_text == "Complete response here"
        assert e.data == {}

    def test_with_data(self) -> None:
        e = DoneEvent(
            full_text="Done",
            data={"task_complete": True, "score": 0.8},
        )
        assert e.data["task_complete"] is True

    def test_frozen(self) -> None:
        e = DoneEvent(full_text="Done")
        with pytest.raises(ValidationError):
            e.full_text = "Changed"  # type: ignore[misc]

    def test_serialization_roundtrip(self) -> None:
        e = DoneEvent(full_text="Complete", data={"key": "value"})
        assert DoneEvent.model_validate(e.model_dump()) == e


class TestErrorEvent:
    """Stream error event — frozen, with partial text recovery."""

    def test_construction(self) -> None:
        e = ErrorEvent(code="AI_TIMEOUT", message="Model timed out")
        assert e.code == "AI_TIMEOUT"
        assert e.partial_text == ""

    def test_with_partial_text(self) -> None:
        e = ErrorEvent(
            code="AI_TIMEOUT",
            message="Timed out",
            partial_text="The article claims",
        )
        assert e.partial_text == "The article claims"

    def test_frozen(self) -> None:
        e = ErrorEvent(code="AI_TIMEOUT", message="Timed out")
        with pytest.raises(ValidationError):
            e.code = "CHANGED"  # type: ignore[misc]

    def test_serialization_roundtrip(self) -> None:
        e = ErrorEvent(code="RATE_LIMIT", message="Too fast", partial_text="Partial")
        assert ErrorEvent.model_validate(e.model_dump()) == e


# ---------------------------------------------------------------------------
# ContentBlock
# ---------------------------------------------------------------------------


class TestContentBlock:
    """Content source marker — frozen, Literal source validation."""

    def test_ai_content(self) -> None:
        cb = ContentBlock(source="ai", content="AI response", model_family="claude")
        assert cb.source == "ai"
        assert cb.model_family == "claude"

    def test_static_content(self) -> None:
        cb = ContentBlock(source="static", content="Static text")
        assert cb.source == "static"
        assert cb.model_family is None

    def test_invalid_source_rejected(self) -> None:
        with pytest.raises(ValidationError, match="source"):
            ContentBlock(source="manual", content="test")

    def test_frozen(self) -> None:
        cb = ContentBlock(source="static", content="test")
        with pytest.raises(ValidationError):
            cb.content = "changed"  # type: ignore[misc]

    def test_serialization_roundtrip(self) -> None:
        cb = ContentBlock(source="ai", content="Response", model_family="gemini")
        assert ContentBlock.model_validate(cb.model_dump()) == cb

    def test_none_model_family_roundtrip(self) -> None:
        cb = ContentBlock(source="static", content="Text")
        dumped = cb.model_dump()
        assert dumped["model_family"] is None
        restored = ContentBlock.model_validate(dumped)
        assert restored.model_family is None
