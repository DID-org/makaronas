# Team GitHub Issues — Draft

*Status: Draft for Vinga's review. Translate to Lithuanian before posting.*
*Date: 2026-03-05*

---

## Overview

These issues are safe to work on in parallel with V5 (Trickster Engine) implementation. They touch content, tooling, and documentation — not the AI layer or backend engine code.

**Reference files the team should read first:**
- `content/tasks/TEMPLATE/AUTHORING_GUIDE.md` — full authoring workflow
- `content/tasks/TEMPLATE/task.json` — template cartridge with placeholders
- `content/tasks/task-clickbait-trap-001/task.json` — reference hybrid task (complete)
- `content/tasks/task-phantom-quote-001/task.json` — reference ai_driven task (complete)
- `content/taxonomy.json` — known triggers, techniques, mediums
- `content/tasks/task.schema.json` — JSON Schema for IDE validation

---

## Issue 1: Add `VideoBlock` to Task Schemas

**Type:** Code (small)
**Time estimate:** 30 minutes
**File:** `backend/tasks/schemas.py`

**What:**
Add a `VideoBlock` class to the task schema, following the exact same pattern as `ImageBlock`. Register it in `KNOWN_BLOCK_TYPES`.

**Why:**
The platform needs to support video content in tasks (e.g., AI-generated video comparison tasks). The schema currently has `ImageBlock`, `AudioBlock`, and `VideoTranscriptBlock` but no block for actual video files.

**Pattern to follow (copy and adapt):**
```python
# Look at ImageBlock (line ~71) and follow the same structure:
class VideoBlock(BaseModel):
    """Video content — AI-generated videos, news clips, social media videos.

    Accessibility: alt_text is required (Framework Principle 14).
    transcript provides text alternative for deaf/hard-of-hearing students.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    type: Literal["video"] = "video"
    src: str
    alt_text: str
    transcript: str | None = None
    duration_seconds: int | None = None
```

**Acceptance criteria:**
- [ ] `VideoBlock` class exists in `backend/tasks/schemas.py`
- [ ] Registered in `KNOWN_BLOCK_TYPES` dict
- [ ] `alt_text` is required (Principle 14 — accessibility)
- [ ] `transcript` is optional (not all videos have transcripts at authoring time)
- [ ] Existing tests pass (run `python -m pytest backend/tests/ -v`)
- [ ] Add a test in the schema tests that validates a cartridge with a `VideoBlock` loads correctly

---

## Issue 2: Build Cartridge Validation CLI Tool

**Type:** Code (tooling)
**Time estimate:** 2-3 hours
**File:** New file — `scripts/validate_cartridge.py`

**What:**
A command-line script that validates a task cartridge against the full loader pipeline and reports errors in human-readable format. Content authors run this before committing.

**Why:**
The existing validation happens at server startup (loader) or in tests. Content authors need a quick way to check their work without running the full server or test suite.

**Usage:**
```bash
# Validate a single cartridge
python scripts/validate_cartridge.py content/tasks/my-new-task-001

# Validate all cartridges
python scripts/validate_cartridge.py content/tasks/
```

**Implementation approach:**
- Import `TaskLoader` from `backend/tasks/loader.py`
- Import taxonomy from `content/taxonomy.json`
- Call `loader.load_task()` with taxonomy context
- Catch and format all validation errors (LoadError, ValidationError, warnings)
- Print clear, actionable messages: "Phase 'intro' references block 'main-image' but no block with that ID exists in presentation_blocks"
- Exit code 0 = clean, 1 = errors, 2 = warnings only

**Acceptance criteria:**
- [ ] Script runs from project root
- [ ] Catches and formats schema validation errors
- [ ] Catches and formats loader business logic errors (path mismatch, orphan phases, missing assets)
- [ ] Reports taxonomy warnings (unknown trigger/technique/medium values)
- [ ] Clear error messages that tell the author what to fix (not stack traces)
- [ ] Works on single cartridge or entire directory
- [ ] Exits with appropriate exit codes

---

## Issue 3: Draft New Task — Adversarial Dialogue (Misleading Statistics)

**Type:** Content
**Time estimate:** 3-4 hours
**Files:** New directory `content/tasks/task-misleading-stats-001/`

**What:**
Create a new hybrid task where the student is shown a social media post citing a real study, but the statistics are presented misleadingly (percentages without base numbers, correlation presented as causation, cherry-picked timeframe).

**Archetype:** Adversarial Dialogue (follows `task-clickbait-trap-001` pattern)
**Persona mode:** `presenting`
**Medium:** `social_post`
**Technique:** `cherry_picking`
**Trigger:** `authority`

**Content requirements:**
- A social media post block (`social_post` type) with misleading statistics about a topic teenagers care about (screen time, social media, school performance — keep it evergreen)
- A source data block (`text` type) showing the actual study data that reveals the misrepresentation
- Static intro phase with buttons (share/question/investigate)
- AI evaluation phase (min 2, max 6 exchanges)
- Three reveal phases (win/partial/timeout) with Lithuanian text explaining the specific statistical manipulation
- Evaluation contract: at least 2 patterns in `patterns_embedded`, at least 1 mandatory checklist item

**Quality checklist:**
- [ ] All text in Lithuanian
- [ ] No real institutions, people, or current events (Principle 4 — Evergreen)
- [ ] `task_id` matches directory name
- [ ] `status: "draft"`
- [ ] Statistics feel realistic but are fictional
- [ ] The manipulation is specific and identifiable (not vague "misleading" — exactly which number is wrong and why)
- [ ] Reveal text explains the technique, not just "you were wrong"
- [ ] Passes `python scripts/validate_cartridge.py` (or loader test)

**Reference:** Copy `content/tasks/TEMPLATE/` as starting point. Study `task-clickbait-trap-001` for the hybrid pattern.

---

## Issue 4: Draft New Task — Adversarial Dialogue (Emotional Headline)

**Type:** Content
**Time estimate:** 3-4 hours
**Files:** New directory `content/tasks/task-emotional-headline-001/`

**What:**
Create a new hybrid task where the student sees a news article with an emotionally charged headline that misrepresents the article's actual content. The article itself is balanced, but the headline selects the most provocative angle.

**Archetype:** Adversarial Dialogue (follows `task-clickbait-trap-001` pattern)
**Persona mode:** `presenting`
**Medium:** `article`
**Technique:** `headline_manipulation`
**Trigger:** `injustice`

**Content requirements:**
- A text block with the full article (balanced, factual)
- The headline that misrepresents it (emotionally charged, provocative)
- The contrast must be clear when you read carefully but easy to miss when skimming (which is how teenagers consume content)
- Static intro phase, AI evaluation phase, three reveal phases
- Evaluation contract: patterns covering headline-body disconnect, emotional framing, omission of nuance

**Quality checklist:**
- Same as Issue 3

**Reference:** Study `task-clickbait-trap-001` — similar structure but different manipulation technique.

---

## Issue 5: Draft New Task — Investigation (Source Tracing)

**Type:** Content
**Time estimate:** 4-5 hours
**Files:** New directory `content/tasks/task-source-trace-001/`

**What:**
Create a new hybrid investigation task where the student traces a claim back through multiple sources, discovering that each "source" references the previous one in a circle — nobody actually verified the original claim.

**Archetype:** Investigation (follows `task-follow-money-001` pattern)
**Persona mode:** `narrator`
**Medium:** `investigation`
**Technique:** `source_weaponization`
**Trigger:** `authority`

**Content requirements:**
- Multiple `search_result` blocks forming an investigation tree
- The trail should feel like genuine research — each result looks credible individually
- At least 2 key findings and 2 dead ends
- `starting_queries` that give the student clear entry points
- The circular reference pattern should be discoverable but not obvious
- AI evaluation phase where the Trickster (as narrator) guides the student through the evidence
- Reveal explaining how circular sourcing works in real-world media

**This is the most complex content task.** Study `task-follow-money-001` carefully before starting.

**Quality checklist:**
- Same as Issues 3-4, plus:
- [ ] Investigation tree has no dead-end loops (every path either reaches a key finding or a clearly marked dead end)
- [ ] `search_result` blocks have realistic `query`, `title`, `snippet` fields
- [ ] `child_queries` correctly chain the investigation forward

---

## Issue 6: Source/Create Image Assets for Visual Manipulation Tasks

**Type:** Content (media)
**Time estimate:** Ongoing
**Files:** `content/tasks/task-misleading-frame-001/assets/` and new task asset directories

**What:**
Create or source image assets for tasks that involve visual manipulation. The existing `task-misleading-frame-001` has placeholder images (`misleading.png`, `context.png`). We need:

1. **For task-misleading-frame-001:** Two photographs of the same scene from different angles/crops — one that tells a misleading story, one that shows the full context. Example: a cropped photo showing an "empty" event vs. the full photo showing a packed venue from a different angle.

2. **For future tasks:** Build a small asset library of:
   - Misleading graphs (real data, misleading presentation — truncated Y-axis, cherry-picked timeframe)
   - Cropped vs. full-context photographs
   - Screenshots of fictional social media posts (styled to evoke but not imitate real platforms — see Principle 5)

**Requirements:**
- All images must be original or CC0/public domain — no copyrighted content
- No real people, real brands, or real events (Principle 4 — Evergreen)
- Each image needs `alt_text` in the cartridge (Principle 14 — Accessibility)
- Images should be clear enough on a school laptop screen (reasonable resolution, good contrast)
- Graphs must use fictional but plausible data

**Acceptance criteria:**
- [ ] `task-misleading-frame-001` has production-quality images replacing placeholders
- [ ] At least 2 misleading graph image sets (graph + source data) ready for future tasks
- [ ] At least 2 cropped-vs-context photo pairs ready for future tasks
- [ ] All assets are in `content/tasks/{task_id}/assets/` directories

---

## Issue 7: Expand Taxonomy

**Type:** Content (data)
**Time estimate:** 1-2 hours
**File:** `content/taxonomy.json`

**What:**
Review the current taxonomy and propose additions based on the Lithuanian media landscape and common manipulation patterns teenagers encounter.

**Current taxonomy:**
- **Triggers (8):** urgency, belonging, injustice, authority, identity, fear, greed, cynicism
- **Techniques (10):** cherry_picking, fabrication, emotional_framing, wedge_driving, omission, false_authority, manufactured_deadline, headline_manipulation, source_weaponization, phantom_quote
- **Mediums (9):** article, social_post, chat, investigation, meme, feed, audio, video_transcript, image

**Questions to consider:**
- Are there manipulation techniques common in Lithuanian media that aren't covered? (e.g., `whataboutism`, `false_equivalence`, `appeal_to_tradition`)
- Are there triggers specific to the Lithuanian context? (e.g., related to national identity, geopolitical concerns)
- Are there mediums teenagers encounter that we're missing? (e.g., `video`, `screenshot`, `voice_message`)
- Are the Lithuanian display names accurate and natural? (check the values in taxonomy.json)

**Process:**
1. Review existing taxonomy.json
2. Research common manipulation patterns in Lithuanian media (news, social media, Telegram channels)
3. Propose additions as a PR — add new entries to taxonomy.json with Lithuanian display names
4. Discuss with the team before merging — taxonomy values appear in evaluation data

**Acceptance criteria:**
- [ ] At least 3-5 new technique or trigger proposals with Lithuanian display names
- [ ] Each addition has a 1-sentence justification (why this is relevant for Lithuanian teenagers)
- [ ] No duplicates of existing values (check semantic overlap, not just string match)
- [ ] `medium: "video"` added (we're adding VideoBlock support)

---

## Issue 8: Draft New Task — Clean Check (Legitimate Article)

**Type:** Content
**Time estimate:** 3-4 hours
**Files:** New directory `content/tasks/task-clean-check-001/`

**What:**
Create the first clean task — an article with NO manipulation that the student must correctly identify as legitimate. This tests the false-positive instinct: not everything is a trick.

**Archetype:** Clean Check
**Persona mode:** `presenting`
**Medium:** `article`
**`is_clean`: `true`**

**Content requirements:**
- A well-written, balanced article about a topic teenagers care about
- The article must be genuinely good journalism — balanced, sourced, nuanced
- The Trickster presents it with the same confidence as a manipulated article — the student must decide
- `patterns_embedded` MUST be empty (is_clean + patterns = hard load error)
- Evaluation inverts: "trickster wins" = student falsely accused clean content (paranoia beat judgment)
- Reveal explains what made this article legitimate and why the student's suspicion was unfounded (or congratulates them for recognizing good content)

**This is pedagogically important.** Without clean tasks, we train paranoia instead of judgment. The reveal must respect the student — "Good judgment means knowing when to trust, not just when to doubt."

**Quality checklist:**
- Same as Issues 3-4, plus:
- [ ] `is_clean: true`
- [ ] `patterns_embedded: []` (empty)
- [ ] The article genuinely has no manipulation (not just subtle manipulation)
- [ ] Evaluation `pass_conditions` describe the inverted outcomes correctly
- [ ] Reveal text includes the calibration message — trusting good content is a skill too

**Note:** The AI prompt for clean tasks is a V5 deliverable (the engine needs to support the inverted evaluation). The team writes the content; V5 writes the prompt. The task will be `status: "draft"` until V5 completes.

---

## Issue 9: Draft Scenario Briefs for Future Tasks

**Type:** Content (planning)
**Time estimate:** 2-3 hours
**Files:** New file `content/tasks/SCENARIO_BRIEFS.md`

**What:**
Write short scenario briefs (half a page each) for 5-8 new tasks across different archetypes. These aren't full cartridges — they're ideas that the team can develop into cartridges later.

**Each brief should include:**
- Task name and archetype (adversarial dialogue, investigation, clean check, sensory trap, empathy flip)
- Topic and manipulation technique
- 2-3 sentence scenario description
- What makes this task interesting/unique for Lithuanian teenagers
- Estimated difficulty (1-5)
- Which content blocks it would need (text, images, audio, video, chat messages)

**Goal:** Build a task pipeline so the team always has the next task ready to develop. Aim for variety across archetypes, techniques, mediums, and difficulty levels.

**The trial target is ~15-20 total tasks.** We have 6 reference cartridges. The team needs to produce 10-15 more. These briefs are the first step.

---

## Priority Order

For the trial timeline, prioritize in this order:

1. **Issue 1** (VideoBlock) — 30 minutes, unblocks video content
2. **Issue 2** (Validation CLI) — useful for all content work that follows
3. **Issue 7** (Taxonomy) — informs all new task content
4. **Issue 9** (Scenario briefs) — plan before building
5. **Issues 3, 4, 5** (New tasks) — the main content pipeline
6. **Issue 6** (Image assets) — can happen in parallel with task writing
7. **Issue 8** (Clean task) — depends on V5 for the AI prompt, but content can be written now

---

*These issues are designed to be self-contained. Each one has a clear pattern to follow, a reference file to study, and acceptance criteria to check against. If anything is unclear, ask in the issue thread or check the AUTHORING_GUIDE.md.*
