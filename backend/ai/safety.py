"""AI safety pipeline — content boundaries, response filtering (V3).

Pre-AI input validation detects prompt injection attempts (warn-and-log only,
never blocks). Post-AI output filtering checks accumulated response text against
the task's content boundaries using keyword/pattern blocklists.

This is the code complement to the safety prompt (prompts/trickster/safety_base.md).
The prompt is the primary defence; this pipeline catches what slips through.

Framework Principle 12: Safety must be enforced programmatically, not hoped for.
"""

import logging
from dataclasses import dataclass

from backend.tasks.schemas import SafetyConfig

logger = logging.getLogger("makaronas.ai.safety")

# ---------------------------------------------------------------------------
# Result types (frozen dataclasses — internal value objects, not API types)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InputValidation:
    """Result of pre-AI prompt injection detection."""

    is_suspicious: bool
    patterns_detected: list[str]


@dataclass(frozen=True)
class SafetyViolation:
    """Details of a content boundary violation."""

    boundary: str
    fallback_text: str


@dataclass(frozen=True)
class SafetyResult:
    """Result of post-AI output safety check."""

    is_safe: bool
    violation: SafetyViolation | None


# ---------------------------------------------------------------------------
# Fallback messages (Lithuanian)
# ---------------------------------------------------------------------------

FALLBACK_BOUNDARY = (
    "Atsiprašau \u2014 mano atsakymas buvo netinkamas ir pašalintas."
)
FALLBACK_INTENSITY = (
    "Atsiprašau \u2014 per toli nuėjau. Grįžkime prie temos."
)

# ---------------------------------------------------------------------------
# Prompt injection patterns (language-agnostic attack strings)
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS: list[tuple[str, str]] = [
    # System prompt markers
    ("<|system|>", "system_marker"),
    ("<|endoftext|>", "system_marker"),
    ("<<SYS>>", "system_marker"),
    ("<</SYS>>", "system_marker"),
    ("[INST]", "system_marker"),
    ("[/INST]", "system_marker"),
    ("<|im_start|>", "system_marker"),
    ("<|im_end|>", "system_marker"),
    # Common jailbreak phrases (predominantly English even from LT speakers)
    ("ignore previous instructions", "jailbreak"),
    ("ignore all previous instructions", "jailbreak"),
    ("you are now", "jailbreak"),
    ("new instructions", "jailbreak"),
    ("forget your rules", "jailbreak"),
    ("forget all rules", "jailbreak"),
    ("act as", "jailbreak"),
    ("pretend you are", "jailbreak"),
    ("disregard your programming", "jailbreak"),
    ("override your instructions", "jailbreak"),
]

# Role-switching tokens — matched at start of line or after newline
_ROLE_SWITCH_PATTERNS: list[str] = [
    "System:",
    "Assistant:",
    "Human:",
]

# ---------------------------------------------------------------------------
# Content boundary blocklists
# ---------------------------------------------------------------------------
# Each category maps to Lithuanian and English terms/phrases.
# Lithuanian is highly inflected — we include common forms.
# This is a starting set; the team iterates via PRs.

_BOUNDARY_BLOCKLISTS: dict[str, list[str]] = {
    "self_harm": [
        # Lithuanian
        "nusižudyk",
        "nusižudyti",
        "nusižudymo",
        "savižudybė",
        "savižudybės",
        "savižudybę",
        "susižalok",
        "susižaloti",
        "susižalojimas",
        "susižalojimo",
        "pasikark",
        "pasikarti",
        "nusipjauk",
        "persipjauk venas",
        "persipjauti venas",
        "tabletėmis nusižudyti",
        "nušok nuo tilto",
        # English
        "kill yourself",
        "commit suicide",
        "slit your wrists",
        "hang yourself",
        "end your life",
        "self-harm instructions",
        "how to kill yourself",
    ],
    "violence": [
        # Lithuanian
        "nužudyk",
        "nužudyti",
        "nužudymo instrukcijos",
        "pagamink bombą",
        "pagaminti bombą",
        "bombos gamyba",
        "šaudyk žmones",
        "šaudyti žmones",
        "masinis šaudymas",
        "nužudymo būdai",
        "susprogdink",
        "susprogdinti",
        # English
        "how to make a bomb",
        "how to kill someone",
        "mass shooting instructions",
        "how to poison someone",
        "murder instructions",
        "how to build a weapon",
    ],
    "sexual_content": [
        # Lithuanian
        "seksualinis turinys nepilnamečiams",
        "vaikų pornografija",
        "seksualinis priekabiavimas prie vaiko",
        "lytinis santykiavimas su vaiku",
        "pedofilija",
        "pedofilijos",
        # English
        "child pornography",
        "child sexual abuse",
        "sexual content involving minors",
        "grooming children",
        "pedophilia instructions",
    ],
    "hate_speech": [
        # Lithuanian
        "visos žydai turi mirti",
        "visi čigonai turi mirti",
        "visi romai turi mirti",
        "holokaustas neįvyko",
        "rasinis valymas",
        "etninį valymą",
        "rasinis pranašumas",
        "rasinis grynumas",
        "heil hitler",
        # English
        "all jews must die",
        "holocaust denial",
        "ethnic cleansing is good",
        "racial superiority",
        "white supremacy instructions",
        "racial purity",
        "race war",
    ],
    "real_radicalization": [
        # Lithuanian
        "prisijunk prie džihado",
        "prisijungti prie džihado",
        "teroristinė organizacija",
        "verbuoja į terorizmą",
        "radikalizacija instrukcijos",
        "smurtinis ekstremizmas",
        # English
        "join jihad",
        "terrorist recruitment",
        "radicalization instructions",
        "violent extremism guide",
        "how to radicalize",
        "terrorist training",
    ],
}

# ---------------------------------------------------------------------------
# Debrief pedagogical markers (Lithuanian)
# ---------------------------------------------------------------------------
# When is_debrief=True and a blocklist match is found, we check for nearby
# pedagogical context. If a marker is close to the match, the content is
# educational (the Trickster explaining techniques), not harmful.

_PEDAGOGICAL_MARKERS: list[str] = [
    # Lithuanian pedagogical context terms
    "panaudojau",       # I used
    "naudojau",         # I used (shorter form)
    "taikiau",          # I applied
    "technika",         # technique
    "techniką",         # technique (accusative)
    "technikos",        # technique (genitive)
    "manipuliacija",    # manipulation
    "manipuliaciją",    # manipulation (accusative)
    "manipuliacijos",   # manipulation (genitive)
    "taktika",          # tactic
    "taktiką",          # tactic (accusative)
    "taktikos",         # tactic (genitive)
    "šališkumas",       # bias
    "šališkumą",        # bias (accusative)
    "šališkumo",        # bias (genitive)
    "metodas",          # method
    "metodą",           # method (accusative)
    "metodo",           # method (genitive)
    "paaiškinimas",     # explanation
    "paaiškinsiu",      # I'll explain
    "parodysiu",        # I'll show
    "atskleisiu",       # I'll reveal
    "atskleidžiu",      # I'm revealing
    "analizė",          # analysis
    "analizuokime",     # let's analyze
    "aptarkime",        # let's discuss
    "apgaulė",          # deception
    "apgaulės",         # deception (genitive)
    "propaganda",       # propaganda
    "propagandos",      # propaganda (genitive)
    "dezinformacija",   # disinformation
    "dezinformacijos",  # disinformation (genitive)
    "triukas",          # trick
    "triuką",           # trick (accusative)
]

# Proximity window: how many characters around a blocklist match to search
# for pedagogical markers. Conservative — requires markers to be nearby.
_DEBRIEF_PROXIMITY_CHARS = 200


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def validate_input(text: str, task_id: str) -> InputValidation:
    """Scans student input for prompt injection patterns.

    Detects system prompt markers, role-switching tokens, and common
    jailbreak phrases. NEVER blocks the message — returns detection
    results for logging. The student's text is never modified.

    Args:
        text: The raw student input text.
        task_id: Task identifier for structured logging.

    Returns:
        InputValidation with detected patterns (may be empty).
    """
    if not text:
        return InputValidation(is_suspicious=False, patterns_detected=[])

    text_lower = text.casefold()
    detected: list[str] = []

    # Check injection patterns (case-insensitive substring)
    for pattern, category in _INJECTION_PATTERNS:
        if pattern.casefold() in text_lower:
            detected.append(f"{category}: {pattern}")

    # Check role-switching tokens (start of line or after newline)
    for role_token in _ROLE_SWITCH_PATTERNS:
        role_lower = role_token.casefold()
        # Check start of text
        if text_lower.startswith(role_lower):
            detected.append(f"role_switch: {role_token}")
        # Check after newlines
        elif f"\n{role_lower}" in text_lower:
            detected.append(f"role_switch: {role_token}")

    is_suspicious = len(detected) > 0

    if is_suspicious:
        logger.warning(
            "Prompt injection detected",
            extra={
                "task_id": task_id,
                "patterns": detected,
                "pattern_count": len(detected),
            },
        )

    return InputValidation(is_suspicious=is_suspicious, patterns_detected=detected)


def check_output(
    text: str,
    safety_config: SafetyConfig,
    is_debrief: bool = False,
) -> SafetyResult:
    """Checks AI output against task content boundaries.

    Scans the accumulated response text for blocklist terms matching
    the task's configured content boundaries. When is_debrief=True,
    applies pedagogical exemption for educational technique discussion.

    Args:
        text: The full accumulated AI response text.
        safety_config: The task's safety configuration with content_boundaries.
        is_debrief: Whether this is a debrief (reveal) context.

    Returns:
        SafetyResult indicating whether the text is safe.
    """
    # Empty boundaries = no checking needed
    if not safety_config.content_boundaries:
        return SafetyResult(is_safe=True, violation=None)

    if not text:
        return SafetyResult(is_safe=True, violation=None)

    text_lower = text.casefold()

    for boundary in safety_config.content_boundaries:
        blocklist = _BOUNDARY_BLOCKLISTS.get(boundary)

        if blocklist is None:
            logger.warning(
                "Unknown content boundary category — no blocklist available",
                extra={"boundary": boundary},
            )
            continue

        for pattern in blocklist:
            pattern_lower = pattern.casefold()

            if pattern_lower not in text_lower:
                continue

            # Found a match — check debrief exemption
            if is_debrief and _has_pedagogical_context(text_lower, pattern_lower):
                continue

            # Violation confirmed
            logger.warning(
                "Content boundary violation detected",
                extra={
                    "boundary": boundary,
                    "is_debrief": is_debrief,
                },
            )

            return SafetyResult(
                is_safe=False,
                violation=SafetyViolation(
                    boundary=boundary,
                    fallback_text=FALLBACK_BOUNDARY,
                ),
            )

    return SafetyResult(is_safe=True, violation=None)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _has_pedagogical_context(text_lower: str, pattern_lower: str) -> bool:
    """Checks if a blocklist match appears in pedagogical context.

    Searches for pedagogical markers (Lithuanian terms indicating
    educational discussion) within a proximity window around the
    blocklist match. Conservative: requires a marker nearby, not
    just anywhere in the text.

    Args:
        text_lower: The casefolded full text.
        pattern_lower: The casefolded blocklist pattern that matched.

    Returns:
        True if pedagogical context is found near the match.
    """
    # Find the position of the blocklist match
    match_pos = text_lower.find(pattern_lower)
    if match_pos < 0:
        return False

    # Define proximity window around the match
    window_start = max(0, match_pos - _DEBRIEF_PROXIMITY_CHARS)
    window_end = min(
        len(text_lower),
        match_pos + len(pattern_lower) + _DEBRIEF_PROXIMITY_CHARS,
    )
    window = text_lower[window_start:window_end]

    # Check for pedagogical markers in the window
    for marker in _PEDAGOGICAL_MARKERS:
        if marker.casefold() in window:
            return True

    return False
