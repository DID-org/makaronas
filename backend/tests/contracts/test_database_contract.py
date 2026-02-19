"""Contract tests for DatabaseAdapter — behavioral specification.

Verifies that any DatabaseAdapter implementation satisfies:
- CRUD operations for student profiles
- Multi-tenant isolation (school_id boundaries — Framework Principle 13)
- GDPR right to access via export (Framework Principle 3)
- GDPR right to deletion (Framework Principle 3)
- Class insights retrieval

These tests use only the public interface — no internal state inspection.

Run against registered implementations:
    python -m pytest backend/tests/contracts/test_database_contract.py -v
"""

import pytest

from backend.schemas import StudentProfile


class TestDatabaseContract:
    """Behavioral contract for DatabaseAdapter implementations."""

    # -- CRUD basics -------------------------------------------------------

    @pytest.mark.asyncio
    async def test_save_then_get_returns_profile(
        self, database, sample_profile
    ) -> None:
        """Save a profile, then get it back by student_id + school_id."""
        await database.save_student_profile(sample_profile)
        result = await database.get_student_profile(
            sample_profile.student_id, sample_profile.school_id
        )
        assert result is not None
        assert result.student_id == sample_profile.student_id
        assert result.school_id == sample_profile.school_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self, database) -> None:
        """Getting a student that doesn't exist must return None."""
        result = await database.get_student_profile("ghost", "school-contract-a")
        assert result is None

    @pytest.mark.asyncio
    async def test_save_overwrites_existing(self, database) -> None:
        """Second save with same student_id + school_id overwrites the first."""
        profile_v1 = StudentProfile(
            student_id="overwrite-test",
            school_id="school-contract-a",
            sessions_completed=0,
        )
        profile_v2 = StudentProfile(
            student_id="overwrite-test",
            school_id="school-contract-a",
            sessions_completed=5,
        )
        await database.save_student_profile(profile_v1)
        await database.save_student_profile(profile_v2)
        result = await database.get_student_profile(
            "overwrite-test", "school-contract-a"
        )
        assert result is not None
        assert result.sessions_completed == 5

    @pytest.mark.asyncio
    async def test_delete_then_get_returns_none(
        self, database, sample_profile
    ) -> None:
        """Deleted profile must not be retrievable."""
        await database.save_student_profile(sample_profile)
        await database.delete_student_profile(
            sample_profile.student_id, sample_profile.school_id
        )
        result = await database.get_student_profile(
            sample_profile.student_id, sample_profile.school_id
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_is_idempotent(self, database) -> None:
        """Deleting a non-existent student must not raise an error."""
        await database.delete_student_profile("ghost", "school-contract-a")

    # -- Multi-tenant isolation (Framework Principle 13) -------------------

    @pytest.mark.asyncio
    async def test_get_with_wrong_school_returns_none(
        self, database, sample_profile
    ) -> None:
        """Profile saved under school-a must not be visible from school-b."""
        await database.save_student_profile(sample_profile)
        result = await database.get_student_profile(
            sample_profile.student_id, "school-other"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_with_wrong_school_does_not_remove(
        self, database, sample_profile
    ) -> None:
        """Delete with wrong school_id must not affect the real profile."""
        await database.save_student_profile(sample_profile)
        await database.delete_student_profile(
            sample_profile.student_id, "school-other"
        )
        result = await database.get_student_profile(
            sample_profile.student_id, sample_profile.school_id
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_same_student_different_schools_coexist(self, database) -> None:
        """Same student_id in different schools must be independent."""
        profile_a = StudentProfile(
            student_id="shared-id",
            school_id="school-a",
            sessions_completed=1,
        )
        profile_b = StudentProfile(
            student_id="shared-id",
            school_id="school-b",
            sessions_completed=9,
        )
        await database.save_student_profile(profile_a)
        await database.save_student_profile(profile_b)

        result_a = await database.get_student_profile("shared-id", "school-a")
        result_b = await database.get_student_profile("shared-id", "school-b")

        assert result_a is not None
        assert result_b is not None
        assert result_a.sessions_completed == 1
        assert result_b.sessions_completed == 9

    # -- GDPR right to access (Framework Principle 3) ----------------------

    @pytest.mark.asyncio
    async def test_export_existing_returns_nonempty_dict(
        self, database, sample_profile
    ) -> None:
        """Export for an existing student must return a non-empty dict."""
        await database.save_student_profile(sample_profile)
        export = await database.export_student_data(
            sample_profile.student_id, sample_profile.school_id
        )
        assert isinstance(export, dict)
        assert len(export) > 0

    @pytest.mark.asyncio
    async def test_export_contains_identifying_data(
        self, database, sample_profile
    ) -> None:
        """Export must contain the student_id and school_id somewhere in the data."""
        await database.save_student_profile(sample_profile)
        export = await database.export_student_data(
            sample_profile.student_id, sample_profile.school_id
        )
        # Check identifying data appears somewhere in the serialized export.
        # The exact nesting is implementation-specific, so we check the
        # string representation rather than asserting a fixed structure.
        export_str = str(export)
        assert sample_profile.student_id in export_str
        assert sample_profile.school_id in export_str

    @pytest.mark.asyncio
    async def test_export_nonexistent_returns_empty_dict(self, database) -> None:
        """Export for a non-existent student must return an empty dict."""
        export = await database.export_student_data("ghost", "school-contract-a")
        assert export == {}

    # -- GDPR right to deletion (Framework Principle 3) --------------------

    @pytest.mark.asyncio
    async def test_delete_then_export_returns_empty(
        self, database, sample_profile
    ) -> None:
        """After deletion, export must return empty dict (complete wipe)."""
        await database.save_student_profile(sample_profile)
        await database.delete_student_profile(
            sample_profile.student_id, sample_profile.school_id
        )
        export = await database.export_student_data(
            sample_profile.student_id, sample_profile.school_id
        )
        assert export == {}

    # -- Class insights ----------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_insights_nonexistent_returns_none(self, database) -> None:
        """Getting insights for a non-existent class must return None."""
        result = await database.get_class_insights("ghost-class", "school-contract-a")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_insights_with_wrong_school_returns_none(
        self, database, seed_insights, sample_insights
    ) -> None:
        """Insights seeded under school-a must not be visible from school-b."""
        result = await database.get_class_insights(
            sample_insights.class_id, "school-other"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_get_insights_returns_seeded_data(
        self, database, seed_insights, sample_insights
    ) -> None:
        """Seeded insights must be retrievable with correct class_id + school_id."""
        result = await database.get_class_insights(
            sample_insights.class_id, sample_insights.school_id
        )
        assert result is not None
        assert result.class_id == sample_insights.class_id
        assert result.school_id == sample_insights.school_id

    # -- Data integrity (verifies save preserves non-default values) -------

    @pytest.mark.asyncio
    async def test_save_preserves_profile_data(
        self, database, sample_profile
    ) -> None:
        """Saved profile must retain all non-default field values."""
        await database.save_student_profile(sample_profile)
        result = await database.get_student_profile(
            sample_profile.student_id, sample_profile.school_id
        )
        assert result is not None
        assert result.sessions_completed == sample_profile.sessions_completed
        assert result.trigger_vulnerability == sample_profile.trigger_vulnerability
        assert result.tasks_completed == sample_profile.tasks_completed
