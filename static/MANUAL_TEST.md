# Manual Test Checklist

Browser-level verification for features that cannot be tested at the API level.

## Prerequisites

1. Start the dev server:
   ```bash
   cd /home/vinga/projects/makaronas
   python -m uvicorn backend.main:app --reload
   ```
2. Open `http://localhost:8000` in a modern browser (Chrome/Firefox/Edge).
3. Ensure `content/tasks/` has at least `task-clickbait-trap-001` (task-01) and `task-follow-money-001` (task-04) loaded.
4. Browser dev tools open (Console tab) to catch JS errors.

---

## Session Lifecycle

- [ ] Welcome screen shows Lithuanian text (title, description, start button)
- [ ] "Pradėti" button creates session and loads first task
- [ ] Refresh mid-task recovers session (same phase, same content)
- [ ] Completing all tasks in sequence shows end screen
- [ ] "Pradėti iš naujo" clears session and returns to welcome
- [ ] Opening a second tab with the same session works (shared sessionStorage)

## Button Interaction

- [ ] Buttons render with labels from cartridge (Lithuanian text, diacritics correct)
- [ ] Clicking a button transitions to the target phase
- [ ] Buttons are visually disabled during API call (lockout)
- [ ] Rapid double-click does not send two requests
- [ ] Keyboard: Tab to button, Enter activates it

## Freeform Dialogue

- [ ] Trickster intro message appears on phase load (left-aligned bubble)
- [ ] Student message appears right-aligned after send
- [ ] Empty message cannot be sent (send button disabled)
- [ ] Typing indicator (animated dots) shows before first AI token
- [ ] Tokens stream in real-time (character by character)
- [ ] On stream completion, raw text is replaced with markdown-rendered version
- [ ] Auto-scroll follows streaming content
- [ ] Manual scroll-up pauses auto-scroll; new content does not pull back down
- [ ] Draft text persists on refresh (type something, refresh, text is still there)
- [ ] Dialogue history recovers on refresh (all previous bubbles restored)
- [ ] Send button disabled while AI is responding

## Investigation Tree

- [ ] Starting queries render as search result cards
- [ ] Cards with children show expand indicator
- [ ] Clicking expandable card reveals child results
- [ ] Key finding toggle works (visual highlight + counter update)
- [ ] Progress indicator shows `{found}/{required}`
- [ ] Submit button is disabled until `min_key_findings` reached
- [ ] Submit button is enabled once threshold met
- [ ] Clicking submit transitions to the target phase
- [ ] Tree state (expanded nodes, marked findings) recovers on refresh
- [ ] Layout changes to 60/40 split (content/interaction) on desktop

## Generation Flow (Empathy Flip)

- [ ] Generation UI renders when `available_actions` includes `"generate"`
- [ ] Student can type a manipulation attempt in the textarea
- [ ] Submit sends to `/generate` endpoint
- [ ] Loading state shown during generation (button disabled)
- [ ] Generated result displays alongside student prompt
- [ ] "Siųsti Tricksteriui" pause button is visible
- [ ] Clicking pause button transitions to freeform dialogue with generated text
- **Note:** Cannot be tested until an empathy flip cartridge exists. Verify by temporarily modifying `_derive_available_actions()` to return `["generate"]` for a freeform phase.

## Debrief + Reveal

- [ ] After reaching a terminal phase, debrief streams automatically
- [ ] Debrief appears in a visually distinct container (purple/violet register)
- [ ] Reveal section shows below debrief with key_lesson text
- [ ] Reveal uses the green/authoritative colour register
- [ ] "Kitas uždavinys" button appears after reveal
- [ ] Clicking "Kitas uždavinys" loads the next task in sequence
- [ ] If debrief AI is unavailable, reveal still shows with a notice
- [ ] Redaction flash appears if safety violation detected (brief visual flash)

## Error Recovery

- [ ] Network error (disconnect WiFi) shows Lithuanian error message with retry
- [ ] AI unavailable shows skip option ("Praleisti")
- [ ] Rate limit (429) shows cooldown notice and auto-retries
- [ ] Session expired (delete sessionStorage manually) shows welcome on next action
- [ ] Malformed response shows generic Lithuanian error (check console for details)

## Accessibility

- [ ] Tab order follows logical flow: content → interaction → controls
- [ ] All buttons are keyboard-accessible (Tab + Enter)
- [ ] Skip-to-content link visible on first Tab press
- [ ] Focus moves to new section heading on section switch
- [ ] Screen reader: ARIA labels on sections (`role="main"`, landmarks)
- [ ] Screen reader: live region on streaming content (`aria-live="polite"`)
- [ ] Screen reader: investigation tree uses `role="tree"` / `role="treeitem"` / `aria-expanded`
- [ ] No focus traps — Tab always progresses through the page

## Responsive Layout

- [ ] Desktop (>800px): side-by-side content + interaction panels
- [ ] Narrow (<800px): panels stack vertically
- [ ] Investigation: 60/40 split on desktop, stacked on narrow
- [ ] No horizontal scrollbar at any width
- [ ] Text remains readable at all widths

## Lithuanian Rendering

- [ ] All diacritics render correctly: ą č ę ė į š ų ū ž (and capitals)
- [ ] Error messages are in Lithuanian
- [ ] Button labels, headings, status messages all Lithuanian
- [ ] Lithuanian quotes („ ") render correctly in content blocks
- [ ] No mojibake or replacement characters anywhere
