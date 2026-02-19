# Prompts Directory

This directory contains all prompt files for the Makaronas platform's AI personas. The AI integration layer loads these files and injects them into model calls. You edit prompts here — plain Markdown, no code — and the platform picks them up.

## Directory Structure

```
prompts/
├── README.md              ← you are here
├── trickster/             ← adversarial AI persona (student-facing)
│   ├── README.md          ← conventions for this directory
│   ├── persona_base.md    ← default persona prompt (any model)
│   ├── persona_claude.md  ← Claude-specific override (optional)
│   ├── persona_gemini.md  ← Gemini-specific override (optional)
│   └── ...
├── composer/              ← teacher's AI collaborator
│   ├── README.md          ← conventions for this directory
│   ├── persona_base.md
│   └── ...
└── tasks/                 ← per-task prompt overrides
    ├── README.md          ← conventions for this directory
    └── {task_id}/         ← one subdirectory per task
        ├── trickster_base.md
        └── trickster_gemini.md
```

**Note:** The directories exist but are currently empty (except for README files). Actual prompt content is authored in future visions (V3+). The READMEs explain the conventions so the team knows what goes where.

## Model-Specific Variants

Each prompt type has a **base** version and optional **model-specific overrides**:

| File pattern | When it's used |
|---|---|
| `persona_base.md` | Default — used when no model-specific file exists |
| `persona_claude.md` | Used when the active model is Claude (any Claude variant) |
| `persona_gemini.md` | Used when the active model is Gemini (any Gemini variant) |

**Resolution order:** The prompt loader (built in V3) checks for a model-specific file first, then falls back to the base version. You only need model-specific variants when a model's behaviour demands different instructions — for example, Flash Lite needs more explicit formatting instructions than Sonnet.

The naming convention uses the **provider name** (`claude`, `gemini`), not the specific model ID. Model family names and their IDs are defined in `backend/models.py`:

| Family name | Model ID |
|---|---|
| `CLAUDE_HAIKU` | claude-haiku-4-5-20251001 |
| `CLAUDE_SONNET` | claude-sonnet-4-6 |
| `CLAUDE_OPUS` | claude-opus-4-6 |
| `GEMINI_FLASH_LITE` | gemini-flash-lite-latest |
| `GEMINI_FLASH` | gemini-3-flash-preview |
| `GEMINI_PRO` | gemini-3-pro-preview |

## Editing Rules

1. **Prompts are plain Markdown.** No special syntax, no executable code blocks, no templating language. Write naturally.

2. **Don't include student data placeholders.** Context injection (student history, exchange turns, profile data) happens in code via the AI integration layer. Prompt files are the raw ingredients — the code assembles the meal.

3. **Don't touch context management markers.** The AI integration layer (V3) handles context assembly: layering prompts, managing token budgets, injecting conversation history. If you see markers related to context management, they're code-controlled — don't edit them by hand.

4. **Model-specific variants should be minimal.** Only create a model-specific file when the base version doesn't work well with that model. The goal is one base prompt that works everywhere, with overrides only where necessary.

5. **Keep prompts evergreen.** No current events, no real public figures, no platform-specific branding. See `FRAMEWORK.md`, Principle 4.

6. **Git tracks everything.** Commit prompt changes like code changes. Review them, discuss them, version them.

## What These Prompts Control

- **Trickster prompts** define the adversarial AI persona that students interact with. The Trickster pushes back, challenges, and eventually reveals what it did. See `trickster/README.md`.

- **Composer prompts** define the teacher's AI collaborator that helps plan curriculum and build task sequences. See `composer/README.md`.

- **Task prompts** override or extend the persona prompts for specific tasks. A task about misleading statistics might need different Trickster behaviour than a task about emotional manipulation. See `tasks/README.md`.

## How the AI Integration Layer Uses These Files

The prompt loader (V3) builds each AI call from layers:

1. **System persona** — loaded from `trickster/` or `composer/`
2. **Task-specific override** — loaded from `tasks/{task_id}/` (if it exists)
3. **Student context** — injected by code (history, profile, session state)
4. **Conversation history** — injected by code (exchange turns)

The prompt files provide layers 1 and 2. The code handles layers 3 and 4. Token budgets ensure the assembled context fits within the model's limits — if it doesn't, the context manager trims intelligently rather than failing.
