# Task Content Directory

This directory holds **task cartridges** — self-contained content packages for each task in the platform. Each task gets its own subdirectory containing a task definition file and any associated assets (images, audio, etc.).

## Directory Structure

```
content/tasks/
├── README.md                          ← you are here
└── {task_id}/                         ← one directory per task
    ├── task.json                      ← task definition (metadata, content blocks, configuration)
    └── assets/                        ← images, audio, and other media
        ├── misleading_graph.png
        ├── screenshot_feed.png
        └── voice_note.mp3
```

### Example (populated)

```
content/tasks/
├── 001_clickbait_trap/
│   ├── task.json
│   └── assets/
│       └── article_screenshot.png
├── 002_misleading_stats/
│   ├── task.json
│   └── assets/
│       ├── graph_original.png
│       └── graph_misleading.png
└── 003_social_pressure/
    ├── task.json
    └── assets/
```

## Asset Serving

Files placed in this directory are served by the API. The path chain:

1. You place a file at `content/tasks/{task_id}/{filename}`
2. The API serves it at `GET /api/v1/assets/{task_id}/{filename}`
3. The frontend renders it using that URL

This works because `LocalFileStorage` (in `backend/hooks/storage.py`) defaults to `base_path="content/tasks"`, and the asset endpoint in `backend/api/composer.py` serves files from that directory.

When the team replaces `LocalFileStorage` with cloud storage (S3, GCS, etc.), the serving path changes — but the directory structure here remains the authoring convention.

## Task Cartridge Format

The `task.json` schema is defined in V2 (Task Engine). Below is a preview of the expected shape — **this will change** when V2 is implemented. Treat it as orientation, not a contract.

```json
{
  "task_id": "001_clickbait_trap",
  "title": "The Clickbait Trap",
  "description": "A sensational headline designed to provoke sharing",
  "medium": "article",
  "difficulty": 2,
  "time_minutes": 10,
  "triggers": ["urgency", "outrage"],
  "techniques": ["clickbait", "emotional_manipulation"],
  "tags": ["social_media", "news"],
  "content_blocks": [
    {
      "type": "text",
      "content": "Your friend just shared this article..."
    },
    {
      "type": "image",
      "asset": "article_screenshot.png"
    }
  ],
  "ai_config": {
    "model_preference": "GEMINI_FLASH",
    "has_ai_dialogue": true,
    "has_static_fallback": true
  }
}
```

## Content Guidelines

- **Evergreen content only.** No current events, no real public figures, no platform-specific branding. See `FRAMEWORK.md`, Principle 4.
- **Fictional but realistic.** Scenarios should feel plausible without being traceable to real events.
- **Multimodal assets are pre-generated.** Images, audio, and other media are authored and reviewed, not generated at runtime.
- **Assets need text descriptions.** For accessibility and voice-mode compatibility, every visual asset should have a text description in the task definition.
- **One task, one directory.** Keep everything for a task self-contained. If a task is broken, it affects only that task — not the rest of the platform.
