"""Context assembly — layering, budgeting, prompt snapshotting.

Assembles the full AI call payload from 8 layers: prompt files (persona,
behaviour, safety, task override), cartridge data (task context, safety
config), language instruction, and student path context. Produces an
AssembledContext that maps directly to AIProvider.stream()/complete() args.

Consumed by:
- TricksterEngine (Phase 5a) — calls assemble_trickster_call() before every AI call
- TricksterEngine (Phase 5b) — calls assemble_debrief_call() for reveals

This is a Tier 2 service module: imports from ai/prompts (Tier 2),
schemas (Tier 1), and tasks/schemas (Tier 1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from backend.ai.prompts import PromptLoader, TricksterPrompts
from backend.schemas import Exchange, GameSession
from backend.tasks.schemas import TaskCartridge

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Character-to-token ratio for budget estimation.
# Lithuanian averages ~3 chars/token; use this as conservative default.
# ---------------------------------------------------------------------------
_CHARS_PER_TOKEN = 3

# ---------------------------------------------------------------------------
# Default token budget — well under any model's context window.
# ---------------------------------------------------------------------------
_DEFAULT_TOKEN_BUDGET = 100_000

# ---------------------------------------------------------------------------
# Transition tool definition (JSON Schema format for AIProvider).
# Included in AssembledContext.tools when exchange_count >= min_exchanges.
# ---------------------------------------------------------------------------
TRANSITION_TOOL: dict[str, Any] = {
    "name": "transition_phase",
    "description": "Signal that the conversation phase should transition.",
    "parameters": {
        "type": "object",
        "properties": {
            "signal": {
                "type": "string",
                "enum": ["understood", "partial", "max_reached"],
                "description": "The transition signal.",
            },
        },
        "required": ["signal"],
    },
}

# ---------------------------------------------------------------------------
# Role mapping: Exchange roles -> provider-neutral message roles.
# Both Gemini and Anthropic accept "user"/"assistant" and re-map internally.
# ---------------------------------------------------------------------------
_ROLE_MAP: dict[str, str] = {
    "student": "user",
    "trickster": "assistant",
}


@dataclass(frozen=True)
class AssembledContext:
    """Provider-ready AI call payload.

    Maps directly to AIProvider.stream() / AIProvider.complete() arguments:
    - system_prompt -> system_prompt parameter
    - messages -> messages parameter
    - tools -> tools parameter
    """

    system_prompt: str
    messages: list[dict[str, str]]
    tools: list[dict] | None


class ContextManager:
    """Assembles AI call payloads from prompts, cartridge data, and session state.

    Handles 8-layer system prompt assembly, token budgeting with exchange
    trimming, prompt snapshotting for live session integrity (P21), and
    redaction context injection.

    Args:
        prompt_loader: Injected PromptLoader for loading prompt files.
        token_budget: Maximum estimated tokens for the full payload.
    """

    def __init__(
        self,
        prompt_loader: PromptLoader,
        token_budget: int = _DEFAULT_TOKEN_BUDGET,
    ) -> None:
        self._loader = prompt_loader
        self._token_budget = token_budget

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def assemble_trickster_call(
        self,
        session: GameSession,
        cartridge: TaskCartridge,
        provider: str,
        exchange_count: int,
        min_exchanges: int,
    ) -> AssembledContext:
        """Assembles the full Trickster dialogue call payload.

        Builds an 8-layer system prompt, formats exchange history as
        provider-neutral messages, trims oldest exchanges if over budget,
        and conditionally includes the transition tool.

        Args:
            session: Current game session with exchanges and choices.
            cartridge: Task cartridge with AiConfig, evaluation, safety.
            provider: Provider name (e.g. "gemini", "anthropic").
            exchange_count: Current exchange count (including the
                about-to-be-added student message).
            min_exchanges: Minimum exchanges before transition tool appears.

        Returns:
            AssembledContext ready for AIProvider.stream().
        """
        # Log context level if not session_only (MVP stub).
        if cartridge.ai_config is not None:
            ctx_req = cartridge.ai_config.context_requirements
            if ctx_req != "session_only":
                logger.debug(
                    "Context level '%s' requested but resolving as "
                    "session_only (MVP)",
                    ctx_req,
                )

        prompts = self._resolve_prompts(session, cartridge, provider)
        system_prompt = self._build_dialogue_system_prompt(
            prompts, session, cartridge,
        )

        messages = self._format_exchanges(session.exchanges)
        messages = self._trim_if_needed(system_prompt, messages)

        tools: list[dict] | None = None
        if exchange_count >= min_exchanges:
            tools = [TRANSITION_TOOL]

        return AssembledContext(
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
        )

    def assemble_debrief_call(
        self,
        session: GameSession,
        cartridge: TaskCartridge,
        provider: str,
    ) -> AssembledContext:
        """Assembles the debrief (reveal) call payload.

        Uses prompt layers 1-4 (snapshot if available), plus a debrief-specific
        layer 5 with EvaluationContract data and a Lithuanian debrief
        instruction. Full exchange history is included without trimming.

        Args:
            session: Current game session with exchanges.
            cartridge: Task cartridge with evaluation contract.
            provider: Provider name.

        Returns:
            AssembledContext for the debrief call (tools=None).
        """
        prompts = self._resolve_prompts(session, cartridge, provider)
        system_prompt = self._build_debrief_system_prompt(
            prompts, session, cartridge,
        )

        # Debrief includes full history — no trimming.
        messages = self._format_exchanges(session.exchanges)

        return AssembledContext(
            system_prompt=system_prompt,
            messages=messages,
            tools=None,
        )

    def snapshot_prompts(
        self,
        session: GameSession,
        trickster_prompts: TricksterPrompts,
    ) -> None:
        """Saves prompt layers 1-4 into the session for live session integrity.

        Called by the trickster engine on the first AI call for a task attempt
        (when session.prompt_snapshots is None). Only non-None fields are stored.

        Args:
            session: Mutable game session to update.
            trickster_prompts: Loaded prompt layers to snapshot.
        """
        snapshot: dict[str, str] = {}
        if trickster_prompts.persona is not None:
            snapshot["persona"] = trickster_prompts.persona
        if trickster_prompts.behaviour is not None:
            snapshot["behaviour"] = trickster_prompts.behaviour
        if trickster_prompts.safety is not None:
            snapshot["safety"] = trickster_prompts.safety
        if trickster_prompts.task_override is not None:
            snapshot["task_override"] = trickster_prompts.task_override

        session.prompt_snapshots = snapshot

    def get_prompt_snapshot(
        self,
        session: GameSession,
    ) -> TricksterPrompts | None:
        """Retrieves snapshotted prompts from the session.

        Args:
            session: Game session that may contain prompt snapshots.

        Returns:
            Reconstructed TricksterPrompts from snapshot, or None if no
            snapshot exists.
        """
        if session.prompt_snapshots is None:
            return None

        return TricksterPrompts(
            persona=session.prompt_snapshots.get("persona"),
            behaviour=session.prompt_snapshots.get("behaviour"),
            safety=session.prompt_snapshots.get("safety"),
            task_override=session.prompt_snapshots.get("task_override"),
        )

    # -------------------------------------------------------------------
    # Prompt resolution
    # -------------------------------------------------------------------

    def _resolve_prompts(
        self,
        session: GameSession,
        cartridge: TaskCartridge,
        provider: str,
    ) -> TricksterPrompts:
        """Resolves prompt layers: snapshot first, then loader fallback.

        Args:
            session: Game session (may contain prompt snapshot).
            cartridge: Task cartridge (for task_id in prompt loading).
            provider: Provider name for model-specific fallback.

        Returns:
            TricksterPrompts from snapshot or loader.
        """
        snapshot = self.get_prompt_snapshot(session)
        if snapshot is not None:
            return snapshot

        task_id = cartridge.task_id if cartridge.ai_config else None
        return self._loader.load_trickster_prompts(provider, task_id)

    # -------------------------------------------------------------------
    # System prompt assembly — Dialogue
    # -------------------------------------------------------------------

    def _build_dialogue_system_prompt(
        self,
        prompts: TricksterPrompts,
        session: GameSession,
        cartridge: TaskCartridge,
    ) -> str:
        """Builds the 8-layer system prompt for Trickster dialogue."""
        layers: list[str] = []

        # Layer 1-4: Prompt files
        self._append_prompt_layers(layers, prompts)

        # Layer 5: Task context (persona mode, phase, evaluation contract)
        layer5 = self._build_task_context(session, cartridge)
        if layer5:
            layers.append(layer5)

        # Layer 6: Safety config
        layer6 = self._build_safety_config(cartridge)
        if layer6:
            layers.append(layer6)

        # Layer 7: Language instruction
        layers.append(self._build_language_instruction())

        # Layer 8: Student path context (context labels from choices)
        layer8 = self._build_context_labels(session)
        if layer8:
            layers.append(layer8)

        # Redaction context (injected after main layers if flagged)
        redaction = self._build_redaction_context(session)
        if redaction:
            layers.append(redaction)

        return "\n\n".join(layers)

    # -------------------------------------------------------------------
    # System prompt assembly — Debrief
    # -------------------------------------------------------------------

    def _build_debrief_system_prompt(
        self,
        prompts: TricksterPrompts,
        session: GameSession,
        cartridge: TaskCartridge,
    ) -> str:
        """Builds the system prompt for the debrief (reveal) call.

        Same layers 1-4 as dialogue, but layer 5 is debrief-specific:
        full EvaluationContract data plus a Lithuanian instruction to drop
        the adversarial stance and reveal manipulation techniques.
        """
        layers: list[str] = []

        # Layer 1-4: Prompt files (same sources as dialogue)
        self._append_prompt_layers(layers, prompts)

        # Layer 5: Debrief-specific context
        layer5 = self._build_debrief_context(cartridge)
        if layer5:
            layers.append(layer5)

        # Layer 6: Safety config
        layer6 = self._build_safety_config(cartridge)
        if layer6:
            layers.append(layer6)

        # Layer 7: Language instruction
        layers.append(self._build_language_instruction())

        # Layer 8: Student path context
        layer8 = self._build_context_labels(session)
        if layer8:
            layers.append(layer8)

        return "\n\n".join(layers)

    # -------------------------------------------------------------------
    # Individual layer builders
    # -------------------------------------------------------------------

    @staticmethod
    def _append_prompt_layers(
        layers: list[str],
        prompts: TricksterPrompts,
    ) -> None:
        """Appends non-None prompt layers 1-4 to the layers list."""
        if prompts.persona is not None:
            layers.append(prompts.persona)
        if prompts.behaviour is not None:
            layers.append(prompts.behaviour)
        if prompts.safety is not None:
            layers.append(prompts.safety)
        if prompts.task_override is not None:
            layers.append(prompts.task_override)

    @staticmethod
    def _build_task_context(
        session: GameSession,
        cartridge: TaskCartridge,
    ) -> str:
        """Builds layer 5: task context with evaluation contract.

        Serializes persona mode, current phase, embedded patterns, checklist
        items, and pass conditions as readable Lithuanian-labeled text.
        """
        parts: list[str] = []
        parts.append("## Uzduoties kontekstas")

        if cartridge.ai_config is not None:
            parts.append(f"\nPersona: {cartridge.ai_config.persona_mode}")

        if session.current_phase is not None:
            parts.append(f"Faze: {session.current_phase}")

        evaluation = cartridge.evaluation

        # Patterns embedded
        if evaluation.patterns_embedded:
            parts.append("\n### Vertinimo kriterijai")
            for i, pattern in enumerate(evaluation.patterns_embedded, 1):
                parts.append(
                    f"{i}. **{pattern.description}**\n"
                    f"   Technika: {pattern.technique}\n"
                    f"   Ryšys su realybe: {pattern.real_world_connection}"
                )

        # Checklist
        if evaluation.checklist:
            parts.append("\n### Kontrolinis sarasas")
            for item in evaluation.checklist:
                mandatory = " [PRIVALOMA]" if item.is_mandatory else ""
                parts.append(f"- {item.description}{mandatory}")

        # Pass conditions
        parts.append("\n### Vertinimo salygos")
        pc = evaluation.pass_conditions
        parts.append(
            f"- Triksteris laimi: {pc.trickster_wins}\n"
            f"- Is dalies: {pc.partial}\n"
            f"- Triksteris pralaimi: {pc.trickster_loses}"
        )

        return "\n".join(parts)

    @staticmethod
    def _build_debrief_context(cartridge: TaskCartridge) -> str:
        """Builds layer 5 for debrief: evaluation data + debrief instruction.

        Instructs the Trickster to drop its adversarial stance, reveal the
        manipulation techniques it used, connect them to the student's actual
        statements, and explain the pedagogical lesson.
        """
        parts: list[str] = []
        parts.append("## Atskleidimo kontekstas")

        evaluation = cartridge.evaluation

        # Patterns
        if evaluation.patterns_embedded:
            parts.append("\n### Panaudoti manipuliacijos metodai")
            for i, pattern in enumerate(evaluation.patterns_embedded, 1):
                parts.append(
                    f"{i}. **{pattern.description}**\n"
                    f"   Technika: {pattern.technique}\n"
                    f"   Ryšys su realybe: {pattern.real_world_connection}"
                )

        # Checklist
        if evaluation.checklist:
            parts.append("\n### Ko mokinys turejo pastebeti")
            for item in evaluation.checklist:
                mandatory = " [PRIVALOMA]" if item.is_mandatory else ""
                parts.append(f"- {item.description}{mandatory}")

        # Pass conditions
        parts.append("\n### Vertinimo salygos")
        pc = evaluation.pass_conditions
        parts.append(
            f"- Triksteris laimi: {pc.trickster_wins}\n"
            f"- Is dalies: {pc.partial}\n"
            f"- Triksteris pralaimi: {pc.trickster_loses}"
        )

        # Debrief instruction
        parts.append(
            "\n### Instrukcija\n"
            "Dabar tu nebesi priesininkas. Nusimesk Triksterio kauke ir "
            "iskisk atvirai su mokiniu. Papasakok, kokius manipuliacijos "
            "metodus panaudojai, nurodydamas konkrecius pavyzdzius is pokalbio. "
            "Kai mokinys sake kazka konkretaus, susiek tai su manipuliacijos "
            "technika. Pvz.: 'Kai sakei, kad saltinis patikimas — tai buvo "
            "autoriteto salisumo spastai.' Pabaigoje paaisink, ko galima "
            "ismokti is sios patirties ir kaip atpazinti panasias situacijas "
            "realiame gyvenime."
        )

        return "\n".join(parts)

    @staticmethod
    def _build_safety_config(cartridge: TaskCartridge) -> str:
        """Builds layer 6: safety configuration from cartridge."""
        safety = cartridge.safety
        parts: list[str] = []
        parts.append("## Saugumo nustatymai")
        if safety.content_boundaries:
            boundaries = ", ".join(safety.content_boundaries)
            parts.append(f"\nTurinio ribos: {boundaries}")
        parts.append(f"Intensyvumo lubos: {safety.intensity_ceiling}/5")
        return "\n".join(parts)

    @staticmethod
    def _build_language_instruction() -> str:
        """Builds layer 7: hard-coded Lithuanian language instruction."""
        return (
            "## Kalbos instrukcija\n\n"
            "Visada atsakyk lietuviškai. Niekada nepersijunk "
            "i kita kalba, net jei mokinys raso kita kalba."
        )

    @staticmethod
    def _build_context_labels(session: GameSession) -> str | None:
        """Builds layer 8: student path context from choice context_labels.

        Returns None if no choices have context_label.
        """
        labels = [
            choice["context_label"]
            for choice in session.choices
            if "context_label" in choice
        ]
        if not labels:
            return None

        lines = ["## Mokinio pasirinkimai", ""]
        for label in labels:
            lines.append(f"- {label}")
        return "\n".join(lines)

    @staticmethod
    def _build_redaction_context(session: GameSession) -> str | None:
        """Builds redaction note and clears the flag.

        If session.last_redaction_reason is set, appends a system note
        explaining that the previous response was redacted. Clears the
        flag so it doesn't persist to subsequent turns.

        Returns None if no redaction flag is set.
        """
        reason = session.last_redaction_reason
        if reason is None:
            return None

        # Clear after reading (one-shot injection).
        session.last_redaction_reason = None

        return (
            "## Sistemos pastaba\n\n"
            f"Tavo ankstesnis atsakymas buvo pasalintas saugumo sistemos "
            f"del: {reason}. "
            "Mokinys mate bendra pakaitini pranesima. Laikykis personazo — "
            "jei mokinys klausia apie cenzura, pripazink tai naturaliai ir "
            "koreguok savo pozuri."
        )

    # -------------------------------------------------------------------
    # Exchange formatting
    # -------------------------------------------------------------------

    @staticmethod
    def _format_exchanges(exchanges: list[Exchange]) -> list[dict[str, str]]:
        """Converts Exchange objects to provider-neutral message dicts.

        Role mapping: "student" -> "user", "trickster" -> "assistant".
        Messages are in chronological order (same as session.exchanges).
        """
        return [
            {"role": _ROLE_MAP[ex.role], "content": ex.content}
            for ex in exchanges
        ]

    # -------------------------------------------------------------------
    # Token budgeting
    # -------------------------------------------------------------------

    def _trim_if_needed(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Trims oldest exchange pairs if total exceeds token budget.

        Uses character-based heuristic (~3 chars/token for Lithuanian).
        Removes complete exchange pairs (user + assistant together) from
        the front to maintain conversation coherence.

        The system prompt is NEVER trimmed.
        """
        system_tokens = len(system_prompt) / _CHARS_PER_TOKEN
        message_tokens = sum(
            len(m["content"]) / _CHARS_PER_TOKEN for m in messages
        )
        total = system_tokens + message_tokens

        if total <= self._token_budget:
            return messages

        overage = total - self._token_budget
        trimmed = list(messages)

        while overage > 0 and len(trimmed) >= 2:
            # Remove the oldest pair (user + assistant).
            pair_tokens = (
                len(trimmed[0]["content"]) / _CHARS_PER_TOKEN
                + len(trimmed[1]["content"]) / _CHARS_PER_TOKEN
            )
            trimmed = trimmed[2:]
            overage -= pair_tokens

        logger.debug(
            "Trimmed %d messages from exchange history (budget=%d tokens)",
            len(messages) - len(trimmed),
            self._token_budget,
        )

        return trimmed
