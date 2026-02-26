# Task Cartridge Authoring Guide

How to create a new task for the Makaronas platform.

---

## Quickstart

1. **Copy the template directory:**
   ```bash
   cp -r content/tasks/TEMPLATE content/tasks/my-task-name-001
   ```

2. **Rename and edit `task.json`:**
   - Set `task_id` to match your directory name exactly (e.g. `my-task-name-001`)
   - Replace all `[PLACEHOLDER: ...]` text with real Lithuanian content
   - Adjust phases, blocks, and evaluation to match your task design

3. **Create prompt directory** (for hybrid/ai_driven tasks):
   ```bash
   mkdir -p prompts/tasks/my-task-name-001
   ```
   Add a `task.md` file with the Trickster prompt.

4. **Create assets directory** (if your task uses images/audio):
   ```bash
   mkdir -p content/tasks/my-task-name-001/assets
   ```

5. **Validate** — run the server or tests to check your cartridge loads cleanly. See [Validation](#validation) below.

---

## IDE Setup (VS Code)

The JSON Schema provides real-time validation, hover-tooltips, and error highlighting in VS Code.

### Option A: `$schema` property (per-file)

Add this line at the top of your `task.json`:
```json
{
  "$schema": "../task.schema.json",
  "task_id": "my-task-name-001",
  ...
}
```

The `$schema` property is ignored by the Python loader — it's purely for IDE support.

### Option B: VS Code settings (global)

Add to your `.vscode/settings.json`:
```json
{
  "json.schemas": [
    {
      "fileMatch": ["content/tasks/*/task.json"],
      "url": "./content/tasks/task.schema.json"
    }
  ]
}
```

This applies the schema to all `task.json` files in `content/tasks/` subdirectories.

---

## Field Reference

The JSON Schema provides hover-tooltips for every field. This section covers the fields that need extra context.

### Identity

| Field | Notes |
|-------|-------|
| `task_id` | Must match directory name. Pattern: `^[a-z0-9][a-z0-9-]*[a-z0-9]$` |
| `task_type` | `"ai_driven"`, `"static"`, or `"hybrid"`. See [Task Types](#the-three-task-types). |
| `version` | Semantic version (`"1.0.0"`). Increment on content changes. |
| `status` | Start with `"draft"`. Set to `"active"` when content is ready. |

### Classification

| Field | Notes |
|-------|-------|
| `trigger` | Psychological trigger. See [Taxonomy](#taxonomy). |
| `technique` | Manipulation technique. See [Taxonomy](#taxonomy). |
| `medium` | Content medium. See [Taxonomy](#taxonomy). |
| `is_clean` | `true` = no manipulation present. `patterns_embedded` must be empty. |
| `is_evergreen` | `true` = no current events, real figures, or dated references. Required for `active` tasks. |

### Content

| Field | Notes |
|-------|-------|
| `presentation_blocks` | Array of content blocks. Each needs a unique `id`. |
| `phases` | Array of state machine nodes. Each needs a unique `id`. |
| `initial_phase` | Must match a phase `id` — the starting point. |

### AI Configuration

Required for `ai_driven` and `hybrid` tasks. Can be `null` for `static` tasks.

| Field | Notes |
|-------|-------|
| `model_preference` | `"fast"`, `"standard"`, or `"complex"`. Pick the cheapest that works. |
| `prompt_directory` | Path relative to project root: `prompts/tasks/{task_id}` |
| `persona_mode` | How the Trickster behaves: `presenting`, `chat_participant`, `narrator`, `commenter` |
| `context_requirements` | How much student history the AI gets: `session_only`, `learning_profile`, `full_history` |

---

## The Three Task Types

### `hybrid` (most common)

Static phases set context (buttons, choices), AI phases deepen the teaching (freeform conversation). The template uses this pattern. Example: student sees content, makes a choice, then discusses with the Trickster.

### `ai_driven`

Pure AI conversation. No buttons, no static branching. The Trickster engages the student in multi-turn dialogue from start to finish. More expensive, more powerful teaching.

To convert the template to `ai_driven`:
- Change `task_type` to `"ai_driven"`
- Remove button phases, keep only AI evaluation phase + terminal reveals
- Set `initial_phase` to your AI phase

### `static`

No AI at all. Deterministic button paths. Zero token cost. Good for simple choice-and-consequence tasks.

To convert the template to `static`:
- Change `task_type` to `"static"`
- Remove AI phases and `ai_config` (set to `null`)
- All interactions must be `button` type
- Every path must reach a terminal phase via button choices

---

## Block Types

Each block represents a piece of content the student sees.

| Type | Required Fields | Use For |
|------|----------------|---------|
| `text` | `id`, `type`, `text` | Articles, headlines, snippets, captions |
| `image` | `id`, `type`, `src`, `alt_text` | Misleading graphs, photos. `alt_text` required for accessibility |
| `audio` | `id`, `type`, `src`, `transcript` | Voice notes, podcast clips. `transcript` required |
| `video_transcript` | `id`, `type`, `transcript` | Video transcript text |
| `meme` | `id`, `type`, `image_src`, `alt_text` | Memes with overlay text |
| `chat_message` | `id`, `type`, `username`, `text` | Chat messages in a thread |
| `social_post` | `id`, `type`, `author`, `text` | Social media posts |
| `search_result` | `id`, `type`, `query`, `title`, `snippet` | Investigation tree nodes |

**Unknown types pass through.** If you need a block type that doesn't exist yet, use any `type` string with `id` — the loader will accept it as a generic block. The frontend will need a renderer for it.

---

## Common Patterns

### Button Branching (hybrid)

The most common pattern: show content, offer choices via buttons, each button leads to a different phase.

```json
{
  "id": "intro",
  "title": "Introduction",
  "visible_blocks": ["main-content"],
  "trickster_content": "Trickster says something...",
  "is_ai_phase": false,
  "interaction": {
    "type": "button",
    "choices": [
      {
        "label": "Choice A",
        "target_phase": "path_a",
        "context_label": "Student chose A"
      },
      {
        "label": "Choice B",
        "target_phase": "path_b",
        "context_label": "Student chose B"
      }
    ]
  },
  "is_terminal": false
}
```

The `context_label` is recorded to session history so AI phases know what the student chose.

### Freeform AI Evaluation

Multi-turn conversation with the Trickster. Exchange bounds control the conversation length.

```json
{
  "id": "evaluate",
  "title": "AI Evaluation",
  "visible_blocks": ["main-content"],
  "is_ai_phase": true,
  "interaction": {
    "type": "freeform",
    "trickster_opening": "Opening question from the Trickster...",
    "min_exchanges": 2,
    "max_exchanges": 6
  },
  "ai_transitions": {
    "on_success": "reveal_win",
    "on_partial": "reveal_partial",
    "on_max_exchanges": "reveal_timeout"
  },
  "is_terminal": false
}
```

- `min_exchanges` prevents premature evaluation (1 exchange = 1 student message + 1 Trickster reply)
- `max_exchanges` triggers `on_max_exchanges` transition when reached
- `ai_transitions` maps engine signals to reveal phases

### Investigation Tree

For "Follow the Money" style tasks. Uses `search_result` blocks and `investigation` interaction.

```json
{
  "id": "investigate",
  "title": "Investigation",
  "visible_blocks": ["article-1", "article-2"],
  "is_ai_phase": false,
  "interaction": {
    "type": "investigation",
    "starting_queries": ["initial search query 1", "initial search query 2"],
    "submit_target": "evaluate",
    "min_key_findings": 2
  },
  "is_terminal": false
}
```

The tree structure comes from `search_result` blocks in `presentation_blocks`:
- Each `search_result` has a `query` that matches it to a query string
- `child_queries` lists new queries unlocked after viewing this result
- `is_key_finding: true` marks critical discoveries
- `is_dead_end: true` marks plausible but unhelpful results

### Terminal Reveal Phases

Every task needs at least one terminal phase. Terminal phases end the task and declare the outcome.

```json
{
  "id": "reveal_win",
  "title": "Reveal — Student Wins",
  "visible_blocks": ["main-content"],
  "trickster_content": "Pre-authored reveal text explaining the tricks used...",
  "is_ai_phase": false,
  "is_terminal": true,
  "evaluation_outcome": "trickster_loses"
}
```

Three standard outcomes:
- `trickster_wins` — student was fully manipulated
- `partial` — student partially resisted
- `trickster_loses` — student identified the manipulation

---

## The Open Type System

The platform is designed to be extensible without code changes.

### Adding New Block Types

If you need a block type that doesn't exist (e.g. `"timeline"`, `"map"`):

1. Use any `type` string in your block
2. Include `id` and `type` — these are always required
3. Add any fields you need — they'll be preserved as-is

```json
{
  "id": "my-timeline",
  "type": "timeline",
  "events": [...],
  "start_date": "2024-01-01"
}
```

The loader accepts this. The frontend needs a renderer for the new type (or it shows a fallback). The JSON Schema validates the structural minimum (`id` + `type`) without rejecting unknown fields.

### Adding New Interaction Types

Same pattern — use any `type` string with any additional fields:

```json
{
  "type": "drag_and_drop",
  "items": [...],
  "targets": [...]
}
```

---

## Validation

The cartridge goes through two validation layers:

### 1. JSON Schema (IDE — instant feedback)

Catches structural errors while you type:
- Missing required fields
- Wrong types (string where integer expected)
- Values out of range (difficulty > 5)
- Typos in known block type fields (via `additionalProperties: false`)

### 2. Server-Side Loader (runtime — full validation)

Catches logical errors the schema can't express:
- `task_id` doesn't match directory name
- Phase graph has orphan phases (unreachable from `initial_phase`)
- No terminal phases
- Missing asset files (images, audio referenced but not on disk)
- `is_clean: true` with non-empty `patterns_embedded` (contradiction)
- `hybrid` task with no AI phases
- Prompt injection patterns in text content (advisory warning)

**When validation fails:**
- Schema errors → JSON Schema shows red squiggles in the IDE
- Path/identity mismatch → hard `LoadError`, cartridge doesn't load
- Path traversal in asset paths → hard `LoadError`
- Missing assets, graph issues, type mismatches → task demoted to `draft` status with a clear warning
- Unknown taxonomy values → warning (accepted, not rejected)

Run the loader to check:
```bash
python -m pytest backend/tests/test_json_schema.py -v
```

---

## Taxonomy

Trigger, technique, and medium values come from `content/taxonomy.json`.

### Current Values

**Triggers:** urgency, belonging, injustice, authority, identity, fear, greed, cynicism

**Techniques:** cherry_picking, fabrication, emotional_framing, wedge_driving, omission, false_authority, manufactured_deadline, headline_manipulation, source_weaponization, phantom_quote

**Mediums:** article, social_post, chat, investigation, meme, feed, audio, video_transcript, image

### Adding New Values

Edit `content/taxonomy.json`:
```json
{
  "triggers": {
    "urgency": "Skubumas",
    "my_new_trigger": "Naujas trigeris"
  }
}
```

The key is the slug used in cartridges. The value is the Lithuanian display name. No code changes needed — the loader reads taxonomy.json at startup.

Unknown values are accepted with a warning. Add them to taxonomy.json to suppress the warning and get a display name.

---

## Gotchas

### Lithuanian Quotation Marks

Lithuanian uses „..." (U+201E opening, U+201C closing). **The closing quote U+201C looks almost identical to ASCII `"` (U+0022)** — and using ASCII `"` inside a JSON string will break the JSON parser.

**Symptoms:** `json.JSONDecodeError: Expecting ',' delimiter` or `Expecting property name enclosed in double quotes`.

**Prevention:**
- Use a JSON-aware editor that shows syntax errors immediately
- If you paste Lithuanian text, check for stray quotation marks
- When writing scripts that generate JSON, use `json.dump(ensure_ascii=False)` rather than writing raw JSON strings

### Phase IDs Must Be Unique

Every phase `id` within a task must be unique. The initial_phase, all button `target_phase` values, and all `ai_transitions` targets must reference existing phase IDs.

### Block IDs Must Be Unique

Every block `id` in `presentation_blocks` must be unique within the task. Phases reference blocks by ID in `visible_blocks`.

### Asset Paths Are Relative

Image `src`, audio `src`, and meme `image_src` are filenames relative to `content/tasks/{task_id}/assets/`. Don't use absolute paths or `../` — the loader rejects path traversal.

### Directory Structure Matters

The loader derives the project root from the cartridge's location:
```
content/tasks/{task_id}/task.json → project_root = 3 levels up
```

Don't move cartridge directories outside the expected location.

### `$schema` Is Optional

The `$schema` property in task.json is for IDE support only. The Python loader ignores it (Pydantic silently drops unknown top-level fields). Don't list it as a required field.

### `_comment` Fields

Use `_comment` or `_comment_*` fields at the **top level** of task.json for inline notes. Don't put them inside sub-objects like `ai_config`, `evaluation`, or `safety` — those have strict field validation.

### Draft Status

New tasks should start with `"status": "draft"`. The loader's business validation may also demote a task to draft if it finds issues (missing assets, broken graph). Set to `"active"` only when the content is complete and reviewed.
