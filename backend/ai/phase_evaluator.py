"""Phase transition evaluator — Flash Lite decides when to advance.

Separates conversation (Flash) from transition decisions (Flash Lite).
Flash talks to the student with no tool pressure. After each exchange,
Flash Lite evaluates whether the student has demonstrated sufficient
understanding to move to the next phase.

Flash Lite receives ~350 tokens per call:
- System prompt: ~100 tokens (fixed evaluator instructions)
- Checklist: ~50 tokens (from cartridge evaluation.checklist)
- One exchange: ~100-200 tokens (student + assistant messages)

Returns: "continue" (no transition) or a transition signal with
satisfied checklist items.
"""

import logging
from dataclasses import dataclass

from backend.ai.providers.base import AIProvider, UsageInfo
from backend.models import ModelConfig

logger = logging.getLogger(__name__)

EVALUATOR_SYSTEM_PROMPT = """You evaluate ONE exchange between a teacher and a student.

You receive a checklist and the latest exchange (student message + teacher response).

RULES:
1. Is the teacher wrapping up, expressing satisfaction, or NOT asking more questions? (e.g. "puikiai", "geras darbas", summarises findings, no question at the end)
2. If YES → call transition_phase. Use signal="understood" if mandatory checklist items are covered, signal="partial" if not.
3. If the teacher IS still asking questions or pushing for more → respond "continue".

The teacher's behaviour is the PRIMARY signal. If the teacher stops asking questions, ALWAYS transition — the student has nothing more to say.

Do NOT generate conversation. Just evaluate."""

TRANSITION_TOOL = {
    "name": "transition_phase",
    "description": "Call when mandatory checklist items are satisfied AND the teacher is wrapping up.",
    "parameters": {
        "type": "object",
        "properties": {
            "signal": {
                "type": "string",
                "enum": ["understood", "partial"],
                "description": "understood = all mandatory covered, partial = some gaps",
            },
            "satisfied_items": {
                "type": "string",
                "description": "Comma-separated checklist item IDs satisfied in this exchange",
            },
        },
        "required": ["signal"],
    },
}


@dataclass
class EvaluatorResult:
    """Result of a phase transition evaluation."""

    should_transition: bool
    signal: str | None = None  # "understood" or "partial"
    satisfied_items: list[str] | None = None
    usage: UsageInfo | None = None


def format_checklist(checklist_items: list[dict]) -> str:
    """Formats cartridge checklist items for the evaluator prompt.

    Args:
        checklist_items: List of checklist dicts from cartridge.evaluation.checklist.
            Each has: id, description, is_mandatory, pattern_refs.

    Returns:
        Formatted checklist string for the evaluator.
    """
    lines = ["CHECKLIST:"]
    for item in checklist_items:
        mandatory = "[mandatory]" if item.get("is_mandatory", False) else "[optional]"
        lines.append(f"- {mandatory} {item['id']}: {item['description']}")
    return "\n".join(lines)


async def evaluate_exchange(
    provider: AIProvider,
    model_config: ModelConfig,
    student_message: str,
    assistant_response: str,
    checklist_text: str,
) -> EvaluatorResult:
    """Evaluates a single exchange to decide whether to transition.

    Args:
        provider: AI provider instance (should be configured for Flash Lite).
        model_config: Model config for Flash Lite.
        student_message: The student's message in this exchange.
        assistant_response: Makaronas's response in this exchange.
        checklist_text: Pre-formatted checklist string.

    Returns:
        EvaluatorResult with transition decision.
    """
    prompt = (
        f"{checklist_text}\n\n"
        f"EXCHANGE:\n"
        f"STUDENT: {student_message}\n"
        f"MAKARONAS: {assistant_response}"
    )

    try:
        # Use complete() with tools — Flash Lite handles this cleanly
        text, usage = await provider.complete(
            system_prompt=EVALUATOR_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            model_config=model_config,
            tools=[TRANSITION_TOOL],
        )

        # If text response (no tool call), it should be "continue"
        if text and text.strip().lower() == "continue":
            return EvaluatorResult(
                should_transition=False,
                usage=usage,
            )

        # If we got here with text, it might be an unexpected response
        logger.warning("Evaluator returned unexpected text: %s", text[:100])
        return EvaluatorResult(should_transition=False, usage=usage)

    except Exception as exc:
        logger.error("Phase evaluator failed: %s", exc)
        return EvaluatorResult(should_transition=False)


async def evaluate_exchange_with_tool(
    provider: AIProvider,
    model_config: ModelConfig,
    student_message: str,
    assistant_response: str,
    checklist_text: str,
) -> EvaluatorResult:
    """Evaluates using streaming to capture tool call events.

    The complete() method may not return tool calls in text form.
    This version uses streaming to capture ToolCallEvents directly.
    """
    from backend.ai.providers.base import TextChunk, ToolCallEvent

    prompt = (
        f"{checklist_text}\n\n"
        f"EXCHANGE:\n"
        f"STUDENT: {student_message}\n"
        f"MAKARONAS: {assistant_response}"
    )

    text_parts: list[str] = []
    tool_signal: str | None = None
    tool_items: str | None = None
    usage: UsageInfo | None = None

    try:
        async for event in provider.stream(
            system_prompt=EVALUATOR_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            model_config=model_config,
            tools=[TRANSITION_TOOL],
            force_tool=False,
        ):
            if isinstance(event, TextChunk):
                text_parts.append(event.text)
            elif isinstance(event, ToolCallEvent):
                if event.function_name == "transition_phase":
                    tool_signal = event.arguments.get("signal", "understood")
                    tool_items = event.arguments.get("satisfied_items", "")

        usage = getattr(provider, "_last_usage", None)

        if tool_signal is not None:
            items = [s.strip() for s in (tool_items or "").split(",") if s.strip()]
            logger.info(
                "Evaluator: transition signal=%s items=%s",
                tool_signal, items,
            )
            return EvaluatorResult(
                should_transition=True,
                signal=tool_signal,
                satisfied_items=items,
                usage=usage,
            )

        text = "".join(text_parts).strip()
        if text.lower() == "continue":
            logger.debug("Evaluator: continue")
        else:
            logger.warning("Evaluator returned unexpected text: %s", text[:100])

        return EvaluatorResult(should_transition=False, usage=usage)

    except Exception as exc:
        logger.error("Phase evaluator failed: %s", exc)
        return EvaluatorResult(should_transition=False)
