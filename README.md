# Makaronas

Educational AI platform for media literacy. Trains teenagers (15-18) to recognise manipulation through adversarial AI dialogue — experience-first, not lecture-first.

## Quick Start

Prerequisites: Python 3.13+, pip

```bash
# Clone and set up
git clone <repo-url> && cd makaronas
python -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt

# Configure
cp .env.example .env
# Edit .env with your API keys (optional — stubs work without them)

# Run
uvicorn backend.main:app --reload --port 8000
```

Visit http://localhost:8000/docs for the interactive API explorer.

Run tests:

```bash
python -m pytest backend/tests/ -v
```

## Docker

```bash
# Build
docker build -t makaronas .

# Run
docker run -p 8000:8000 makaronas

# Run with environment variables
docker run -p 8000:8000 -e GOOGLE_API_KEY=your-key makaronas

# Run tests in container
docker run makaronas python -m pytest backend/tests/ -v
```

The container runs as a non-root user. No `.env` file is baked into the image — pass configuration via `-e` flags or `--env-file`.

## Architecture Overview

Makaronas uses a **hooks pattern**: every external dependency (auth, database, sessions, file storage) is defined as an abstract interface in `backend/hooks/interfaces.py`. The platform ships with in-memory stubs so it runs standalone. The team replaces stubs with real implementations behind the same interfaces.

| Interface | Stub | What it abstracts |
|-----------|------|-------------------|
| `AuthService` | `FakeAuthService` | Token validation, user lookup |
| `DatabaseAdapter` | `InMemoryStore` | Student profiles, class insights, GDPR operations |
| `SessionStore` | `InMemorySessionStore` | Game session persistence with 24h TTL |
| `FileStorage` | `LocalFileStorage` | Task asset storage and URL generation |
| `RateLimiter` | *(no stub yet)* | Per-student/school rate limiting |

Singletons are wired in `backend/api/deps.py` — the single swap point for all implementations.

## Project Structure

```
makaronas/
├── backend/
│   ├── main.py              # FastAPI app, middleware, router mounting
│   ├── config.py            # Environment config with typed defaults
│   ├── models.py            # AI model ID registry
│   ├── schemas.py           # Pydantic data models (shared vocabulary)
│   ├── streaming.py         # SSE streaming utilities
│   ├── api/
│   │   ├── deps.py          # Dependency injection (the stub swap point)
│   │   ├── student.py       # Student endpoints
│   │   ├── teacher.py       # Teacher endpoints
│   │   └── composer.py      # Composer + asset endpoints
│   ├── hooks/
│   │   ├── interfaces.py    # Abstract base classes (the contracts)
│   │   ├── auth.py          # FakeAuthService stub
│   │   ├── database.py      # InMemoryStore stub
│   │   ├── sessions.py      # InMemorySessionStore stub
│   │   └── storage.py       # LocalFileStorage stub
│   └── tests/
│       ├── contracts/       # Interface contract tests (run against any impl)
│       └── test_*.py        # Unit/integration tests
├── prompts/                 # Prompt templates (Markdown, team-editable)
│   ├── trickster/           # Adversarial AI prompts
│   ├── composer/            # Teacher collaboration AI prompts
│   └── tasks/               # Per-task prompt templates
├── content/
│   ├── tasks/               # Task cartridge assets (images, audio, text)
│   └── roadmaps/prebuilt/   # Pre-built curriculum roadmaps
├── frontend/                # Frontend integration guide (SPA consumes the API)
├── .env.example             # All configuration variables, documented
├── Dockerfile               # Single-container backend deployment
├── FRAMEWORK.md             # Engineering principles
├── PLATFORM_VISION.md       # Full platform specification
└── ROADMAP.md               # Build order across all visions
```

## API Endpoints

All endpoints are prefixed with `/api/v1`. All responses use the `ApiResponse` envelope: `{"ok": true, "data": {...}}` or `{"ok": false, "error": {"code": "...", "message": "..."}}`.

Interactive documentation with schemas and try-it-out: http://localhost:8000/docs

### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Server health check |

### Student

| Method | Path | Description |
|--------|------|-------------|
| POST | `/student/session` | Create a new game session |
| GET | `/student/session/{id}/next` | Get the next task |
| POST | `/student/session/{id}/respond` | Submit response (SSE stream) |
| GET | `/student/session/{id}/debrief` | Get task debrief (SSE stream) |
| GET | `/student/profile/{id}/radar` | Student skill radar |
| DELETE | `/student/profile/{id}` | GDPR: delete all student data |
| GET | `/student/profile/{id}/export` | GDPR: export all student data |

### Teacher

| Method | Path | Description |
|--------|------|-------------|
| GET | `/teacher/library` | Browse tasks (filter by trigger, technique, etc.) |
| GET | `/teacher/library/{task_id}` | Full task detail |
| GET | `/teacher/roadmaps` | List roadmaps |
| POST | `/teacher/roadmaps` | Create custom roadmap |
| GET | `/teacher/class/{class_id}/insights` | Anonymous class-level patterns |

### Composer

| Method | Path | Description |
|--------|------|-------------|
| POST | `/composer/chat` | Composer AI dialogue (SSE stream) |
| POST | `/composer/roadmap/generate` | Generate roadmap from description |
| POST | `/composer/roadmap/refine` | Refine existing roadmap |

### Assets

| Method | Path | Description |
|--------|------|-------------|
| GET | `/assets/{task_id}/{filename}` | Serve task content files |

## For the Team: Replacing Stubs

The platform is designed for you to swap stubs with real implementations:

1. Read the interface contract in `backend/hooks/interfaces.py`
2. Write your implementation (e.g., `PostgresStore(DatabaseAdapter)`)
3. Swap the import in `backend/api/deps.py` — change one line
4. Run contract tests to verify: `python -m pytest backend/tests/contracts/ -v`

Each stub file has `# TEAM:` markers showing what to replace. The contract tests are parameterized — they run against stubs today and your real implementations tomorrow.

## Configuration

All configuration is documented in `.env.example`. Copy it to `.env` and edit.

Key points:
- **Model swapping** is one line: change the family name in `.env`, restart. See `backend/models.py` for the registry.
- **Real environment variables** override `.env` file values (useful for Docker/CI).
- **Hook configuration** (database URL, Redis, auth provider) is commented out in `.env.example` — uncomment as you implement real services.

## Documentation

| Document | Contents |
|----------|----------|
| `FRAMEWORK.md` | Engineering principles — safety, GDPR, cost, team handoff |
| `PLATFORM_VISION.md` | Full 12-vision platform specification |
| `ROADMAP.md` | Build order and dependencies across visions |
| `PAGRINDAS.md` | Platformos principai (lietuviu kalba) |
| `prompts/README.md` | Prompt file structure and editing guide |
| `content/tasks/README.md` | Task cartridge format and asset conventions |
| `frontend/README.md` | Frontend SPA integration guide |
| `/docs` (running server) | Auto-generated API docs with schemas |
