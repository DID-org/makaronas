"""Hook interfaces — abstract base classes for all swappable services.

These ABCs define the contracts between the AI/platform logic and the
infrastructure layer. Each one has a stub implementation (Phase 2b) that
lets the platform run end-to-end without real infrastructure, and a
production implementation that the team wires in when ready.

Tier 1 leaf module: imports only from abc, typing (stdlib) and
backend.schemas (also Tier 1). No project services, no orchestration.

TEAM: To implement a real service, subclass the relevant ABC and implement
every abstract method. Python will raise TypeError at instantiation if
any method is missing — you'll know immediately what's left to do.

Usage:
    from backend.hooks.interfaces import AuthService, DatabaseAdapter
    from backend.hooks.interfaces import SessionStore, FileStorage, RateLimiter
"""

from abc import ABC, abstractmethod
from typing import Any

from backend.schemas import ClassInsights, GameSession, StudentProfile, User


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


class AuthService(ABC):
    """Validates auth tokens and resolves users.

    The auth provider (OAuth, JWT, SAML — team's choice) lives behind this
    interface. The platform never touches tokens directly; it asks the
    AuthService and gets a User back.

    TEAM: Replace the stub (FakeAuthService) with your auth provider.
    The stub accepts any token and returns a configurable test user.
    """

    @abstractmethod
    async def validate_token(self, token: str) -> User | None:
        """Validates an auth token and returns the associated user.

        Args:
            token: Auth token from the request (format determined by
                the team's auth provider — JWT, session cookie, etc.).

        Returns:
            The User if the token is valid and not expired, None otherwise.
        """
        ...

    @abstractmethod
    async def get_user(self, user_id: str) -> User | None:
        """Looks up a user by their ID.

        Used for internal service calls that already know the user ID
        (e.g., background jobs, admin tools) and don't need token validation.

        Args:
            user_id: The opaque user identifier.

        Returns:
            The User if found, None if the user doesn't exist.
        """
        ...


# ---------------------------------------------------------------------------
# Database (persistent learning profiles)
# ---------------------------------------------------------------------------


class DatabaseAdapter(ABC):
    """Persistent storage for student learning profiles and class insights.

    Holds statistical summaries only — never raw conversation text. That
    lives in SessionStore (24h TTL). This separation is a GDPR design
    decision: raw student responses expire, learning profiles persist.

    Multi-tenancy: every method that reads or writes student data takes
    school_id explicitly. This is the isolation boundary — a school_id
    mismatch must never return another school's data.

    TEAM: Replace the stub (InMemoryStore) with your database.
    The stub uses Python dicts and loses data on restart.
    """

    @abstractmethod
    async def get_student_profile(
        self, student_id: str, school_id: str
    ) -> StudentProfile | None:
        """Retrieves a student's persistent learning profile.

        Args:
            student_id: The opaque student identifier.
            school_id: Multi-tenant isolation key. Must match the profile's
                school_id — never return a profile from a different school.

        Returns:
            The StudentProfile if found within the school, None otherwise.
        """
        ...

    @abstractmethod
    async def save_student_profile(self, profile: StudentProfile) -> None:
        """Creates or updates a student's learning profile.

        The profile's school_id field is the source of truth for
        multi-tenant placement — no separate school_id parameter needed.

        Args:
            profile: The StudentProfile to persist. If a profile already
                exists for this student_id + school_id, it is overwritten.
        """
        ...

    @abstractmethod
    async def delete_student_profile(
        self, student_id: str, school_id: str
    ) -> None:
        """Deletes ALL stored data for a student within a school (GDPR).

        This is the structural enforcement of the right to deletion.
        After this call, get_student_profile must return None for this
        student_id + school_id combination.

        Args:
            student_id: The opaque student identifier.
            school_id: Multi-tenant isolation key.
        """
        ...

    @abstractmethod
    async def export_student_data(
        self, student_id: str, school_id: str
    ) -> dict[str, Any]:
        """Exports all stored data for a student in a human-readable format (GDPR).

        This is the structural enforcement of the right to access. The
        returned dict must contain everything stored for this student —
        the exact shape is the implementer's choice, but it must be
        complete and readable.

        Args:
            student_id: The opaque student identifier.
            school_id: Multi-tenant isolation key.

        Returns:
            A dict containing all stored data for the student. Empty dict
            if no data exists.
        """
        ...

    @abstractmethod
    async def get_class_insights(
        self, class_id: str, school_id: str
    ) -> ClassInsights | None:
        """Retrieves aggregated, anonymous class-level patterns.

        Returns statistical summaries only — never individual student data.
        Used by the teacher dashboard.

        Args:
            class_id: The class identifier.
            school_id: Multi-tenant isolation key.

        Returns:
            ClassInsights if data exists for the class, None otherwise.
        """
        ...


# ---------------------------------------------------------------------------
# Session storage (ephemeral, 24h TTL)
# ---------------------------------------------------------------------------


class SessionStore(ABC):
    """Ephemeral storage for active game sessions (24h TTL).

    Holds raw student responses and conversation exchanges — data that
    must not persist long-term. Deliberately separate from DatabaseAdapter
    per the two-layer memory design: session (24h TTL) + learning profile
    (persistent, GDPR-safe).

    The store is responsible for TTL enforcement: get_session returns None
    for expired sessions. Callers never check expiry manually. The
    GameSession.expires_at field is the TTL source of truth.

    TEAM: Replace the stub (InMemorySessionStore) with your session store.
    The stub uses Python dicts and loses data on restart.
    """

    @abstractmethod
    async def get_session(self, session_id: str) -> GameSession | None:
        """Retrieves an active game session.

        Returns None for both non-existent and expired sessions. The store
        enforces TTL — callers don't check expires_at themselves.

        Args:
            session_id: The session identifier.

        Returns:
            The GameSession if it exists and hasn't expired, None otherwise.
        """
        ...

    @abstractmethod
    async def save_session(self, session: GameSession) -> None:
        """Creates or updates a game session.

        The session's expires_at field determines when it becomes
        invisible to get_session.

        Args:
            session: The GameSession to persist.
        """
        ...

    @abstractmethod
    async def delete_session(self, session_id: str) -> None:
        """Deletes a game session immediately.

        Used for explicit session termination (student logs out, session
        reset). For normal expiry, TTL handles cleanup.

        Args:
            session_id: The session identifier.
        """
        ...


# ---------------------------------------------------------------------------
# File storage (task assets — images, audio, etc.)
# ---------------------------------------------------------------------------


class FileStorage(ABC):
    """Storage for task assets — images, audio clips, documents.

    Assets are pre-authored content bundled with tasks, not user uploads.
    The interface abstracts over local filesystem (development) and cloud
    storage (production — S3, GCS, etc.).

    TEAM: Replace the stub (LocalFileStorage) with your cloud storage.
    The stub reads/writes from the local content/tasks/ directory.
    """

    @abstractmethod
    async def get_asset_url(self, task_id: str, filename: str) -> str:
        """Returns a URL the frontend can use to load a task asset.

        For the stub, this is /api/v1/assets/{task_id}/{filename}.
        For production, it could be a CDN URL or signed cloud storage URL.

        Args:
            task_id: The task that owns the asset.
            filename: The asset filename within the task.

        Returns:
            A URL string that resolves to the asset content.
        """
        ...

    @abstractmethod
    async def store_asset(
        self, task_id: str, filename: str, data: bytes
    ) -> str:
        """Persists an asset and returns its URL.

        Args:
            task_id: The task that owns the asset.
            filename: The asset filename within the task.
            data: Raw bytes of the asset content.

        Returns:
            A URL string for the stored asset (same format as get_asset_url).
        """
        ...


# ---------------------------------------------------------------------------
# Rate limiting (stub ABC for team)
# ---------------------------------------------------------------------------


class RateLimiter(ABC):
    """Rate limiting for API actions.

    An interface stub for the team to implement when needed. No stub
    implementation is provided — the platform runs without rate limiting
    during development. The team adds a real implementation (Redis-based
    sliding window, token bucket, etc.) when deploying to production.

    TEAM: Implement this when you need rate limiting. The action parameter
    is a free string — define your own vocabulary (e.g., "ai_call",
    "session_create", "export_data").
    """

    @abstractmethod
    async def check_rate_limit(self, user_id: str, action: str) -> bool:
        """Checks whether an action is allowed under the current rate limit.

        Args:
            user_id: The user attempting the action.
            action: A string identifying the action type (e.g., "ai_call",
                "session_create"). The vocabulary is defined by the team.

        Returns:
            True if the action is allowed, False if rate-limited.
        """
        ...

    @abstractmethod
    async def record_action(self, user_id: str, action: str) -> None:
        """Records that an action was taken, for sliding window tracking.

        Call this after a successful action to update the rate limit counters.

        Args:
            user_id: The user who performed the action.
            action: The action type string (same vocabulary as check_rate_limit).
        """
        ...
