"""Model ID registry — single source of truth for AI model identifiers.

Every AI call in the platform resolves its model ID through this module.
The rest of the codebase imports family-name constants from here — no raw
model ID strings anywhere else.

To swap a model: change the string value of the constant below. One line
changed, eight hundred schools speak to a new mind.
"""

# --- Claude models ---
CLAUDE_HAIKU: str = "claude-haiku-4-5-20251001"
CLAUDE_SONNET: str = "claude-sonnet-4-6"
CLAUDE_OPUS: str = "claude-opus-4-6"

# --- Gemini models ---
GEMINI_FLASH_LITE: str = "gemini-flash-lite-latest"
GEMINI_FLASH: str = "gemini-3-flash-preview"
GEMINI_PRO: str = "gemini-3-pro-preview"

# Lookup map — env var value → actual model ID.
# Keys match the constant names exactly (case-sensitive).
# Used by config.py to resolve e.g. TRICKSTER_MODEL=GEMINI_FLASH → "gemini-3-flash-preview".
MODEL_MAP: dict[str, str] = {
    "CLAUDE_HAIKU": CLAUDE_HAIKU,
    "CLAUDE_SONNET": CLAUDE_SONNET,
    "CLAUDE_OPUS": CLAUDE_OPUS,
    "GEMINI_FLASH_LITE": GEMINI_FLASH_LITE,
    "GEMINI_FLASH": GEMINI_FLASH,
    "GEMINI_PRO": GEMINI_PRO,
}
