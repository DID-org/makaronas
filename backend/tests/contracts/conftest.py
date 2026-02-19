"""Fixtures for contract tests — one parameterized fixture per hook interface.

Each fixture yields a fresh implementation instance. Today there's only the
stub ("stub" param). When the team adds a real implementation (e.g., Postgres,
Redis, S3), they add a second param value and an elif branch.

TEAM: To test your implementation against the contracts:
    1. Add your param string (e.g., "postgres") to the params list.
    2. Add an elif branch that yields your implementation instance.
    3. Run: python -m pytest backend/tests/contracts/ -v
    All tests should pass. If any fail, your implementation doesn't satisfy
    the contract — read the failing test's docstring for what's expected.

Uses @pytest_asyncio.fixture (not @pytest.fixture) for async fixture support
in strict mode. See Phase 4a implementation notes.
"""

from datetime import datetime, timedelta, timezone

import pytest_asyncio

from backend.hooks.auth import FakeAuthService
from backend.hooks.database import InMemoryStore
from backend.hooks.sessions import InMemorySessionStore
from backend.hooks.storage import LocalFileStorage
from backend.schemas import ClassInsights, GameSession, StudentProfile


# ---------------------------------------------------------------------------
# Interface fixtures (parameterized for future implementations)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(params=["stub"])
async def auth_service(request):
    """Yields an AuthService implementation.

    TEAM: Add your auth provider here:
        @pytest_asyncio.fixture(params=["stub", "oauth"])
        async def auth_service(request):
            if request.param == "stub":
                yield FakeAuthService()
            elif request.param == "oauth":
                yield YourOAuthService(test_config)
    """
    if request.param == "stub":
        yield FakeAuthService()


@pytest_asyncio.fixture(params=["stub"])
async def database(request):
    """Yields a DatabaseAdapter implementation.

    TEAM: Add your database adapter here:
        @pytest_asyncio.fixture(params=["stub", "postgres"])
        async def database(request):
            if request.param == "stub":
                yield InMemoryStore()
            elif request.param == "postgres":
                adapter = YourPostgresAdapter(test_dsn)
                yield adapter
                await adapter.cleanup()  # if needed
    """
    if request.param == "stub":
        yield InMemoryStore()


@pytest_asyncio.fixture(params=["stub"])
async def session_store(request):
    """Yields a SessionStore implementation.

    TEAM: Add your session store here:
        @pytest_asyncio.fixture(params=["stub", "redis"])
        async def session_store(request):
            if request.param == "stub":
                yield InMemorySessionStore()
            elif request.param == "redis":
                store = YourRedisSessionStore(test_url)
                yield store
                await store.flush()  # if needed
    """
    if request.param == "stub":
        yield InMemorySessionStore()


@pytest_asyncio.fixture(params=["stub"])
async def file_storage(request, tmp_path):
    """Yields a FileStorage implementation backed by an isolated temp directory.

    TEAM: Add your cloud storage here:
        @pytest_asyncio.fixture(params=["stub", "s3"])
        async def file_storage(request, tmp_path):
            if request.param == "stub":
                yield LocalFileStorage(base_path=str(tmp_path))
            elif request.param == "s3":
                yield YourS3Storage(test_bucket, test_prefix)
    """
    if request.param == "stub":
        yield LocalFileStorage(base_path=str(tmp_path))


# ---------------------------------------------------------------------------
# Helper fixtures (shared test data)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def sample_profile():
    """A StudentProfile with non-default values for data integrity assertions."""
    return StudentProfile(
        student_id="student-contract-1",
        school_id="school-contract-a",
        trigger_vulnerability={"urgency": 0.7, "belonging": 0.4},
        technique_recognition={},
        engagement_signals={"avg_response_time": 12.5},
        growth_trajectory={"trend": "improving"},
        sessions_completed=3,
        last_active=datetime(2026, 2, 18, 10, 30, tzinfo=timezone.utc),
        tasks_completed=["task-1", "task-2", "task-3"],
    )


@pytest_asyncio.fixture
async def sample_session():
    """A GameSession with expires_at 24 hours in the future."""
    return GameSession(
        session_id="sess-contract-1",
        student_id="student-contract-1",
        school_id="school-contract-a",
        language="lt",
        current_task="task-intro",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
    )


@pytest_asyncio.fixture
async def sample_insights():
    """A ClassInsights instance with known values for assertion."""
    return ClassInsights(
        class_id="class-contract-1",
        school_id="school-contract-a",
        trigger_distribution={"urgency": 0.6, "belonging": 0.3, "injustice": 0.1},
        common_failure_points=["cherry-picked-stats", "fabricated-quote"],
        growth_trends={"week_1": 0.4, "week_2": 0.6},
    )


@pytest_asyncio.fixture
async def seed_insights(database, sample_insights):
    """Seeds class insights into the database fixture.

    This helper calls the stub's seed_class_insights() method. When the team
    adds a real database implementation, they'll need their own seeding
    strategy (e.g., INSERT INTO class_insights ...) because
    seed_class_insights() is not part of the DatabaseAdapter ABC.

    TEAM: Extend this fixture for your implementation:
        if hasattr(database, 'seed_class_insights'):
            database.seed_class_insights(sample_insights)
        else:
            # Your seeding strategy here (e.g., raw SQL insert)
            await your_seed_function(database, sample_insights)
    """
    if hasattr(database, "seed_class_insights"):
        database.seed_class_insights(sample_insights)
    return sample_insights
