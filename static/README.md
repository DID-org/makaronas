# Makaronas Frontend

Vanilla JS single-page application served as static files from FastAPI.
No framework, no build step, no transpilation. Every web developer can read and modify it.

---

## Architecture

The frontend follows a **unidirectional state flow**: event handlers call `App.updateState(changes)`, which merges changes into a central `state` object and triggers `render()`. The render function reads state and updates the DOM. No two-way bindings, no virtual DOM — just a plain object and imperative DOM updates.

```
User event / API response
    → updateState({ key: value })
        → Object.assign(state, changes)
            → render()
                → renderSection(), renderPhase(), renderInteraction(), ...
```

All modules are IIFEs that expose their public API on `window` (e.g., `window.App`, `window.Api`). Script load order in `index.html` matters — dependencies must load before consumers.

---

## File Map

| File | Purpose | Public API |
|------|---------|------------|
| `js/app.js` | Entry point. Central state, `updateState`/`render` cycle, session lifecycle, task sequence, phase transitions, debrief/reveal flow, session recovery. | `window.App` — `updateState`, `getState`, `render`, `renderPhase`, `loadFirstTask`, `handlePhaseTransition`, `startPostTaskFlow`, `renderDebrief`, `renderReveal`, `skipCurrentTask` |
| `js/api.js` | Backend API client. Fetch wrapper with auth headers, `ApiResponse` envelope unwrapping, error code → Lithuanian message mapping, SSE parser (manual `ReadableStream` for POST, `EventSource` for GET). | `window.Api` — `createSession`, `getCurrentSession`, `loadTask`, `submitChoice`, `generate`, `streamRespond`, `streamDebrief`, `deleteProfile`, `exportProfile`, `getRadar`, `assetUrl` |
| `js/i18n.js` | Lithuanian UI strings as a flat map. Uses `\u` escapes for diacritics. | `window.I18n` — flat key/value object |
| `js/renderer.js` | PresentationBlock renderers. Markdown pipeline (marked + DOMPurify), block type dispatcher, asset URL resolution. | `window.Renderer` — `renderBlocks`, `renderBlock`, `renderMarkdown`, `registerType`, `renderBlocksInto` |
| `js/interactions.js` | Interaction type dispatcher. Routes phase data to the correct interaction handler (button, freeform, investigation, generation). | `window.Interactions` — `renderInteraction` |
| `js/dialogue.js` | Freeform chat UI. Message bubbles, streaming display, typing indicator, auto-scroll with manual-scroll-pause, draft persistence, error/redact handling. | `window.Dialogue` — `renderFreeformInteraction`, `renderDialogueHistory`, `clearDialogue`, `createStreamingDisplay`, `appendStudentBubble` |
| `js/investigation.js` | Investigation tree UI. Tree navigation, key finding tracking, progress indicator, sessionStorage persistence, submission via POST `/choice`. | `window.Investigation` — `renderInvestigation`, `clearInvestigation`, `recoverInvestigation` |
| `js/vendor/marked.min.js` | Vendored markdown parser (marked.js v17). | `window.marked` |
| `js/vendor/purify.min.js` | Vendored HTML sanitizer (DOMPurify). | `window.DOMPurify` |
| `css/main.css` | All styles. Layout primitives, colour tokens (5 registers), typography, section visibility, block styles, responsive layout. | — |
| `index.html` | HTML structure. 6 sections, script tags, ARIA landmarks. | — |

---

## State Management

The central `state` object in `app.js`:

```js
var state = {
  section: 'welcome',    // Active section: welcome | task | debrief | reveal | end | error
  error: null,           // { message: string } or null
  locked: false,         // UI lockout during API calls
  session: null,         // { session_id, auth_token } or null
  task: null,            // Phase data dict from _derive_phase_response()
  phase: null,           // Current phase ID string
  taskSequenceIndex: 0,  // Position in TASK_SEQUENCE
  terminal: null         // { evaluation_outcome, reveal } when is_terminal
};
```

**Rules:**
- All mutations go through `updateState(changes)` — never modify `state` directly from outside `app.js`.
- Setting `error` automatically switches to the error section.
- `session`, `taskSequenceIndex` are persisted to `sessionStorage` for recovery.

---

## Section Switching

Six sections in `index.html`, each with `id="section-{name}"`:
`welcome`, `task`, `debrief`, `reveal`, `end`, `error`.

CSS handles visibility — `app.js` sets `data-active-section` on `#app`:

```css
[data-active-section="welcome"] #section-welcome { display: block; }
/* ... all others hidden by default */
```

On switch, `renderSection()` scrolls to top and moves focus to the new section's `<h2 tabindex="-1">` heading for accessibility.

---

## How to Add a New Block Renderer

1. **Write the renderer function** in `renderer.js`. It receives `(block, taskId)` and must return a DOM element:

```js
function renderMyNewBlock(block, taskId) {
  var el = document.createElement('div');
  el.className = 'block-my-new-type';
  el.textContent = block.data.content || '';
  return el;
}
```

2. **Register it** in the `RENDERERS` map at the bottom of the renderer functions:

```js
var RENDERERS = {
  text: renderTextBlock,
  // ... existing renderers ...
  my_new_type: renderMyNewBlock   // ← add here
};
```

3. **Add CSS** in `main.css` for `.block-my-new-type`.

Unknown block types automatically fall through to `renderGenericBlock`, which displays the type label and a structured JSON dump. You can also register types dynamically via `Renderer.registerType('my_type', renderFn)`.

---

## How to Add a New Interaction Type

1. **Add a case** in `renderInteraction()` in `interactions.js`:

```js
switch (interaction.type) {
  case 'button': /* ... */ break;
  case 'freeform': /* ... */ break;
  case 'investigation': /* ... */ break;
  case 'my_new_type':
    renderMyNewInteraction(panel, interaction, phaseData);
    break;
  default:
    renderGenericInteraction(panel, interaction, phaseData);
}
```

2. **Follow the pattern:** render UI controls → bind event handlers → call API on user action → handle response → call `App.handlePhaseTransition(newPhaseData)` to advance.

3. **Add CSS** for the new interaction's elements.

4. **Handle cleanup** if the interaction has persistent state (add a `clearMyInteraction()` function and call it from `App.renderPhase()` during phase transitions).

---

## Lithuanian Strings

All student-facing text lives in `js/i18n.js` as a flat map on `window.I18n`.

**Diacritic convention:** Lithuanian characters use `\u` escapes to prevent encoding corruption:
- `\u0105` = ą, `\u010D` = č, `\u0119` = ę, `\u0117` = ė
- `\u012F` = į, `\u0161` = š, `\u0173` = ų, `\u016B` = ū, `\u017E` = ž

To add a new string: add a key to `window.I18n` in `i18n.js`, then reference it as `window.I18n.my_key` in JS. Error codes map to i18n keys via `ERROR_MAP` in `api.js`.

---

## Error Handling

```
Backend error (HTTP 4xx/5xx)
  → api.js unwraps ApiResponse envelope
  → Maps error.code to I18n key via ERROR_MAP
  → Throws ApiError { code, message (Lithuanian), statusCode, data }
  → Consumer catches and displays:
      - Inline (dialogue errors → retry button in chat area)
      - Section-level (fatal errors → error section with retry/clear)
```

**Special cases:**
- **429 Rate Limited:** Shows cooldown notice, auto-retries after delay.
- **`AI_UNAVAILABLE`:** Shows skip option ("Praleisti") to advance past AI phase.
- **`SESSION_NOT_FOUND`:** Clears sessionStorage, returns to welcome.
- **`TASK_CONTENT_UPDATED`:** Restarts task attempt from the updated cartridge.

---

## Session Recovery

On page load, `recoverSession()` checks sessionStorage for:
- `makaronas_session_id` — session ID
- `makaronas_auth_token` — auth token
- `makaronas_task_sequence_index` — position in task sequence

If found, calls `GET /session/{id}/current` to get current phase state. On success:
- Restores task/phase state
- Re-renders dialogue history from `dialogue_history` in response
- Rehydrates investigation tree state from `makaronas_interaction_state`
- Restores draft text from `makaronas_interaction_state`

If the session is gone (`SESSION_NOT_FOUND`), clears storage and shows welcome.

---

## CSS Colour Registers

Five semantic colour registers in `main.css`, each with `bg`, `text`, `border`, and accent variants:

| Register | Purpose | Prefix |
|----------|---------|--------|
| **Platform** | Chrome, navigation, welcome/end screens | `--color-platform-*` |
| **Content** | Task content panel, presentation blocks | `--color-content-*` |
| **Dialogue** | Chat area, message bubbles, input | `--color-dialogue-*` |
| **Debrief** | Post-task AI analysis (Layer 2) | `--color-debrief-*` |
| **Reveal** | Authoritative lesson reveal (Layer 3) | `--color-reveal-*` |

Plus shared tokens: `--color-error`, `--color-success`, `--color-warning`, `--font-*`.

---

## Known Limitations

- **Generation flow** cannot be tested end-to-end until an empathy flip cartridge exists. The `_derive_available_actions()` helper currently never returns `"generate"` for any real cartridge.
- **Task sequence** is a hardcoded array (`TASK_SEQUENCE` in `app.js`). V9 (Roadmap Engine) replaces this with teacher-driven sequences.
- **No offline support.** All state depends on the backend session store.
