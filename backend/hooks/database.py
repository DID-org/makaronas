"""In-memory database — development stub for DatabaseAdapter.

Python dict-backed storage for student profiles and class insights.
Data lives only in memory and is lost on restart. Multi-tenant isolation
is enforced by composite (student_id, school_id) keys — a query with
the wrong school_id returns None, never another school's data.

TEAM: Replace this with your real database (Postgres, etc.). Subclass
DatabaseAdapter from backend.hooks.interfaces and implement all five
abstract methods. The stub uses model_dump() for GDPR export — your
implementation should return equivalent data from your database.

Tier 2 service module: imports from backend.hooks.interfaces (Tier 1)
and backend.schemas (Tier 1).

Usage:
    from backend.hooks.database import InMemoryStore

    db = InMemoryStore()
    await db.save_student_profile(profile)
    await db.get_student_profile("student-1", "school-1")
"""

from typing import Any

from backend.hooks.interfaces import DatabaseAdapter
from backend.schemas import ClassInsights, StudentProfile


class InMemoryStore(DatabaseAdapter):
    """STUB — dict-backed storage, loses data on restart.

    Profiles are keyed by (student_id, school_id) tuples.
    Class insights are keyed by (class_id, school_id) tuples.

    TEAM: Replace with your database adapter. Satisfy the DatabaseAdapter
    interface from backend.hooks.interfaces. The composite-key pattern
    here shows the multi-tenant isolation guarantee your implementation
    must provide.
    """

    def __init__(self) -> None:
        """Initialises empty in-memory stores."""
        self._profiles: dict[tuple[str, str], StudentProfile] = {}
        self._insights: dict[tuple[str, str], ClassInsights] = {}

    async def get_student_profile(
        self, student_id: str, school_id: str
    ) -> StudentProfile | None:
        """Retrieves a student profile by composite key.

        Args:
            student_id: The student identifier.
            school_id: Multi-tenant isolation key.

        Returns:
            The StudentProfile if found within the school, None otherwise.
        """
        return self._profiles.get((student_id, school_id))

    async def save_student_profile(self, profile: StudentProfile) -> None:
        """Stores a student profile, keyed by its student_id and school_id.

        Args:
            profile: The StudentProfile to store. Creates or overwrites.
        """
        self._profiles[(profile.student_id, profile.school_id)] = profile

    async def delete_student_profile(
        self, student_id: str, school_id: str
    ) -> None:
        """Deletes a student profile. No-op if not found (idempotent).

        Args:
            student_id: The student identifier.
            school_id: Multi-tenant isolation key.
        """
        self._profiles.pop((student_id, school_id), None)

    async def export_student_data(
        self, student_id: str, school_id: str
    ) -> dict[str, Any]:
        """Exports all stored data for a student as a dict (GDPR).

        Args:
            student_id: The student identifier.
            school_id: Multi-tenant isolation key.

        Returns:
            {"profile": <model_dump output>} if found, empty dict if not.
        """
        profile = self._profiles.get((student_id, school_id))
        if profile is None:
            return {}
        return {"profile": profile.model_dump()}

    async def get_class_insights(
        self, class_id: str, school_id: str
    ) -> ClassInsights | None:
        """Retrieves class insights by composite key.

        Args:
            class_id: The class identifier.
            school_id: Multi-tenant isolation key.

        Returns:
            ClassInsights if found, None otherwise.
        """
        return self._insights.get((class_id, school_id))

    def seed_class_insights(self, insights: ClassInsights) -> None:
        """Pre-populates class insights for testing.

        Not part of the DatabaseAdapter ABC — this is a stub convenience
        method. The real database computes insights from aggregated
        student data.

        Args:
            insights: The ClassInsights to store.
        """
        self._insights[(insights.class_id, insights.school_id)] = insights
