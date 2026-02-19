# Per-Task Prompt Overrides

This directory holds prompt overrides for specific tasks. When a task needs the Trickster (or Composer) to behave differently than its default persona, the override goes here.

## Directory Structure

```
tasks/
├── README.md                  ← you are here
└── {task_id}/                 ← one subdirectory per task
    ├── trickster_base.md      ← task-specific Trickster prompt (any model)
    ├── trickster_gemini.md    ← Gemini-specific override (optional)
    └── composer_base.md       ← task-specific Composer prompt (optional)
```

Each subdirectory is named with the task's ID — the same ID used in the task cartridge (`content/tasks/{task_id}/`).

## How Overrides Work

The prompt loader (V3) assembles AI calls in layers:

1. **Base persona** from `prompts/trickster/` or `prompts/composer/`
2. **Task override** from `prompts/tasks/{task_id}/` (if it exists)
3. **Student context** injected by code

Task overrides **extend or replace** the base persona for that specific task. For example:

- A task about misleading statistics might add: "Focus on cherry-picked data and scale manipulation. Cite real-sounding but fictional studies."
- A task about social pressure might add: "Adopt a peer voice. Use group belonging language."

If no task override exists, the base persona is used as-is.

## Naming Convention

- `trickster_base.md` — task-specific Trickster instructions (any model)
- `trickster_claude.md` — Claude-specific override (optional)
- `trickster_gemini.md` — Gemini-specific override (optional)
- `composer_base.md` — task-specific Composer instructions (optional)

Same model-variant pattern as the parent directories.

## When to Create a Task Override

Create a task override when the task needs specific AI behaviour that doesn't belong in the general persona. Don't create an override just to repeat what the base persona already says.

**Good reasons for an override:**
- The task requires a specific manipulation technique or communication style
- The Trickster needs domain-specific knowledge for this task's scenario
- The reveal/debrief needs task-specific talking points

**No override needed when:**
- The base persona handles the task's general category well enough
- The task is static (no AI interaction)
