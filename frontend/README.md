# Frontend Integration Guide

The Makaronas frontend is a **separate SPA** (Single Page Application) that communicates with the backend exclusively via the REST API. The framework choice is yours — React, Vue, Svelte, whatever serves the team best. This guide tells you everything you need to start building without touching the backend.

## Quick Start

1. Start the backend:
   ```bash
   cd /path/to/makaronas
   uvicorn backend.main:app --port 8000 --reload
   ```

2. Browse the interactive API docs:
   ```
   http://localhost:8000/docs
   ```
   The Swagger UI shows every endpoint, request/response shapes, and lets you make test calls directly.

3. Point your SPA dev server at `http://localhost:3000` or `http://localhost:5173` — both are pre-configured as allowed CORS origins.

## API Namespaces

All endpoints live under `/api/v1/`. Four route groups:

| Prefix | Purpose | Auth required |
|---|---|---|
| `/api/v1/student/*` | Student game experience — sessions, tasks, profile, GDPR | Yes (any role) |
| `/api/v1/teacher/*` | Teacher dashboard — task library, roadmaps, class insights | Yes (teacher/admin) |
| `/api/v1/composer/*` | Composer AI chat — teacher's AI collaborator | Yes (teacher/admin) |
| `/api/v1/assets/*` | Static asset serving — images, audio for tasks | Yes (any role) |

Plus a health check at `GET /api/v1/health` (no auth required).

### Student Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/student/session` | Create a new game session |
| GET | `/api/v1/student/session/{id}/next` | Get the next task |
| POST | `/api/v1/student/session/{id}/respond` | Submit response (SSE stream) |
| GET | `/api/v1/student/session/{id}/debrief` | Get task debrief (SSE stream) |
| GET | `/api/v1/student/profile/{id}/radar` | Student's learning profile |
| DELETE | `/api/v1/student/profile/{id}` | GDPR: delete all student data |
| GET | `/api/v1/student/profile/{id}/export` | GDPR: export all student data |

### Teacher Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/v1/teacher/library` | Browse task library (with filters) |
| GET | `/api/v1/teacher/library/{task_id}` | Full task detail |
| GET | `/api/v1/teacher/roadmaps` | List roadmaps |
| POST | `/api/v1/teacher/roadmaps` | Create custom roadmap |
| GET | `/api/v1/teacher/class/{class_id}/insights` | Anonymous class-level patterns |

### Composer Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/composer/chat` | Chat with the Composer (SSE stream) |
| POST | `/api/v1/composer/roadmap/generate` | Generate a roadmap from description |
| POST | `/api/v1/composer/roadmap/refine` | Refine an existing roadmap |

### Asset Endpoint

| Method | Path | Description |
|---|---|---|
| GET | `/api/v1/assets/{task_id}/{filename}` | Serve a task's static asset |

## Response Format

Every endpoint returns an `ApiResponse` envelope:

```json
{
  "ok": true,
  "data": { ... },
  "error": null
}
```

On error:

```json
{
  "ok": false,
  "data": null,
  "error": {
    "code": "SESSION_NOT_FOUND",
    "message": "No session with this ID exists."
  }
}
```

Error codes are uppercase strings (`TASK_NOT_FOUND`, `SESSION_EXPIRED`, `AUTH_ERROR`, `VALIDATION_ERROR`, `INTERNAL_ERROR`, etc.). They grow across visions — don't hard-code an exhaustive list.

## SSE Streaming

Three endpoints stream responses via **Server-Sent Events** (SSE):

- `POST /api/v1/student/session/{id}/respond`
- `GET /api/v1/student/session/{id}/debrief`
- `POST /api/v1/composer/chat`

### Event Types

| Event | Data shape | Meaning |
|---|---|---|
| `token` | `{"text": "..."}` | Incremental text chunk from the AI |
| `done` | `{"full_text": "...", "data": {...}}` | Stream complete — full text + structured payload |
| `error` | `{"code": "...", "message": "...", "partial_text": "..."}` | Failure — includes any partial text received before the error |

### Client Example

```javascript
const response = await fetch('/api/v1/student/session/123/respond', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer your-token',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({ action: 'chat', payload: { message: 'I think this is real' } })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();
let buffer = '';

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  buffer += decoder.decode(value, { stream: true });
  const lines = buffer.split('\n');
  buffer = lines.pop(); // keep incomplete line in buffer

  for (const line of lines) {
    if (line.startsWith('event: ')) {
      const eventType = line.slice(7);
      // next 'data:' line has the payload
    }
    if (line.startsWith('data: ')) {
      const payload = JSON.parse(line.slice(6));
      // handle based on eventType
    }
  }
}
```

Alternatively, use the `EventSource` API for GET-based SSE endpoints, or a library like `eventsource-parser` for POST-based ones.

## Authentication

All endpoints (except health) require a `Bearer` token in the `Authorization` header:

```
Authorization: Bearer your-token-here
```

**Currently stubs.** The backend ships with `FakeAuthService`, which accepts any non-empty token and returns a configurable test user. The team implements real authentication by swapping `FakeAuthService` in `backend/hooks/auth.py` with a real implementation that satisfies the `AuthService` interface.

During development, any non-empty string works as a token.

## CORS

The backend allows cross-origin requests from origins listed in the `CORS_ORIGINS` environment variable. Defaults:

- `http://localhost:3000` (Create React App, Next.js)
- `http://localhost:5173` (Vite)

To add your dev server's origin, update `CORS_ORIGINS` in `.env` (comma-separated list).

## Architecture Notes

- **All state is server-side.** Sessions, profiles, and learning data live in the backend. The frontend is a thin rendering layer.
- **No secrets in the client.** API keys, model credentials, and internal endpoints never reach the browser.
- **The frontend is replaceable.** The API is the contract. You can rebuild the frontend without touching the backend.
- **Currently stubs throughout.** All endpoints return structured stub responses. Real AI responses, task content, and database persistence come in future visions (V2-V3).
