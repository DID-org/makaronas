# Composer Prompts

Prompt files for the **Composer** — the teacher's AI collaborator that helps plan curriculum, build task sequences, and understand class patterns.

## What Goes Here

Persona and behaviour prompts that define how the Composer communicates with teachers. The Composer shows its reasoning, accepts overrides, and explains its choices — modelling healthy human-AI collaboration.

## Expected Files

```
composer/
├── README.md              ← you are here
├── persona_base.md        ← core persona: helpful, transparent, deferential to teacher
├── persona_claude.md      ← Claude-specific adjustments (optional)
├── persona_gemini.md      ← Gemini-specific adjustments (optional)
├── behaviour_base.md      ← interaction rules: how to suggest, when to defer
├── behaviour_claude.md    ← Claude-specific rules (optional)
└── behaviour_gemini.md    ← Gemini-specific rules (optional)
```

**None of these files exist yet.** They'll be authored when the AI integration layer is built (V3). This README documents the convention.

## Naming Convention

- `{type}_base.md` — default version, used by any model
- `{type}_claude.md` — override for Claude models
- `{type}_gemini.md` — override for Gemini models

The prompt loader tries the model-specific version first, falls back to base.

## Writing Guidelines

- **The Composer is a collaborator, not an authority.** It suggests, explains reasoning, and accepts the teacher's override. It never insists.
- **Show your work.** The Composer explains why it recommends a particular task sequence or difficulty curve. Teachers should understand the reasoning, not just follow instructions.
- **No jargon.** Teachers aren't AI specialists. Explain in plain language.
- **Respect professional expertise.** The teacher knows their students. The Composer knows the task library and learning patterns. Good collaboration leverages both.
- **Plain Markdown only.** No code, no template syntax.

## Composer vs Trickster

| | Trickster | Composer |
|---|---|---|
| Audience | Students (15-18) | Teachers |
| Tone | Adversarial, theatrical | Supportive, transparent |
| Goal | Challenge and reveal | Assist and explain |
| Endpoint | `/api/v1/student/*` | `/api/v1/composer/*` |
