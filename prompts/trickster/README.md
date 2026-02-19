# Trickster Prompts

Prompt files for the **Trickster** — the adversarial AI persona that students interact with during tasks. The Trickster challenges, pushes back, and ultimately reveals its methods.

## What Goes Here

Persona and behaviour prompts that define how the Trickster speaks, argues, and reveals. These are loaded by the AI integration layer (V3) as system-level instructions.

## Expected Files

```
trickster/
├── README.md              ← you are here
├── persona_base.md        ← core persona: voice, tone, boundaries
├── persona_claude.md      ← Claude-specific adjustments (optional)
├── persona_gemini.md      ← Gemini-specific adjustments (optional)
├── behaviour_base.md      ← interaction rules: when to push back, when to reveal
├── behaviour_claude.md    ← Claude-specific rules (optional)
├── behaviour_gemini.md    ← Gemini-specific rules (optional)
└── safety_base.md         ← content boundaries the Trickster must never cross
```

**None of these files exist yet.** They'll be authored when the AI integration layer is built (V3). This README documents the convention so you know what goes where.

## Naming Convention

- `{type}_base.md` — default version, used by any model
- `{type}_claude.md` — override for Claude models
- `{type}_gemini.md` — override for Gemini models

The prompt loader tries the model-specific version first, falls back to base. Only create model-specific files when the base doesn't work well for that model.

## Writing Guidelines

- **The Trickster is adversarial by design. The platform is not.** The Trickster lies, manipulates, and challenges — but these are pedagogical tools. After every trick, the Trickster tells the student exactly what it did.
- **No real harmful content.** No real misinformation about health, violence, self-harm. Manipulation is theatrical and educational. See `FRAMEWORK.md`, Principle 2.
- **Respect the student.** Teenagers detect patronising AI instantly. Write the Trickster as a sharp adversary, not a condescending teacher pretending to be tricky.
- **Keep it evergreen.** No current events, no real public figures. The manipulation techniques are timeless — the scenarios should be too.
- **Plain Markdown only.** No code, no template syntax. The AI integration layer handles context injection.

## Persona vs Behaviour vs Safety

| File type | What it controls |
|---|---|
| `persona_*` | Voice, tone, character — *who* the Trickster is |
| `behaviour_*` | Interaction rules — *when* to push back, reveal, adapt |
| `safety_*` | Hard boundaries — topics and content the Trickster must never touch |

The safety file is separate because it must survive prompt edits. Changing the Trickster's personality should never accidentally remove a safety boundary.
