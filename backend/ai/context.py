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

import base64
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.ai.prompts import PromptLoader, TricksterPrompts
from backend.schemas import Exchange, GameSession
from backend.tasks.schemas import ImageBlock, MemeBlock, TaskCartridge

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
# Fixed token cost per image for budget estimation.
# Gemini charges ~258 tokens minimum per image. Anthropic charges more but
# scales with resolution. Using 258 as a conservative floor for budgeting.
# ---------------------------------------------------------------------------
_TOKENS_PER_IMAGE = 258

# ---------------------------------------------------------------------------
# Task history context — bounded prior task outcomes for cross-task awareness.
# ---------------------------------------------------------------------------
_MAX_HISTORY_TASKS = 3

_OUTCOME_LABELS: dict[str, str] = {
    "on_success": "Mokinys suprato",
    "on_partial": "Dalinis supratimas",
    "on_max_exchanges": "Nepavyko suprasti",
}

# ---------------------------------------------------------------------------
# File extension -> MIME type mapping for image assets.
# ---------------------------------------------------------------------------
_IMAGE_MEDIA_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

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

# ---------------------------------------------------------------------------
# Neutral system prompt for context-isolated generation (Cluster H).
# The tool AI is a plain helpful assistant — no Trickster persona, no teaching
# context, no evaluation rubric. Lithuanian only.
# ---------------------------------------------------------------------------
_GENERATION_SYSTEM_PROMPT = (
    "Tu esi pagalbinis AI asistentas. "
    "Atlik tiksliai tai, ko pra\u0161o vartotojas. "
    "Nekomentuok u\u017eduoties, neprid\u0117k papildom\u0173 paaiškinim\u0173. "
    "Atsakyk lietuvi\u0173 kalba."
)


@dataclass(frozen=True)
class AssembledContext:
    """Provider-ready AI call payload.

    Maps directly to AIProvider.stream() / AIProvider.complete() arguments:
    - system_prompt -> system_prompt parameter
    - messages -> messages parameter
    - tools -> tools parameter
    """

    system_prompt: str
    messages: list[dict[str, Any]]
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
        content_dir: Path | None = None,
    ) -> None:
        self._loader = prompt_loader
        self._token_budget = token_budget
        self._content_dir = content_dir

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
            prompts, session, cartridge, provider,
        )

        messages: list[dict[str, Any]] = self._format_exchanges(session.exchanges)

        # Inject multimodal image context if the current phase has images.
        image_parts = self._extract_visible_images(cartridge, session)
        context_prefix_count = 0
        if image_parts:
            image_msg = self._build_image_context_message(image_parts)
            messages = [image_msg] + messages
            context_prefix_count = 1

        messages = self._trim_if_needed(
            system_prompt, messages, context_prefix_count,
        )

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

        # Load fourth wall prompt: snapshot first, then fresh load.
        fourth_wall = self.get_fourth_wall_snapshot(session)
        if fourth_wall is None:
            fourth_wall = self._loader.load_fourth_wall_prompt(provider)

        system_prompt = self._build_debrief_system_prompt(
            prompts, session, cartridge, fourth_wall_prompt=fourth_wall,
        )

        # Debrief includes full history — no trimming.
        messages = self._format_exchanges(session.exchanges)

        return AssembledContext(
            system_prompt=system_prompt,
            messages=messages,
            tools=None,
        )

    def assemble_generation_call(
        self,
        source_content: str,
        student_prompt: str,
    ) -> AssembledContext:
        """Assembles a context-isolated generation call payload.

        Builds a minimal context for the tool AI: neutral system prompt,
        source content as a user message, and the student's generation
        prompt as a second user message. No Trickster persona, no
        evaluation rubric, no teaching context, no transition tools.

        Args:
            source_content: The source material the student is working with
                (e.g., a counter-article for the empathy flip).
            student_prompt: The student's instruction for what to generate.

        Returns:
            AssembledContext with neutral system_prompt, messages, and
            tools=None.
        """
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": source_content},
            {"role": "user", "content": student_prompt},
        ]
        return AssembledContext(
            system_prompt=_GENERATION_SYSTEM_PROMPT,
            messages=messages,
            tools=None,
        )

    def snapshot_prompts(
        self,
        session: GameSession,
        trickster_prompts: TricksterPrompts,
        fourth_wall: str | None = None,
    ) -> None:
        """Saves prompt layers 1-4 into the session for live session integrity.

        Called by the trickster engine on the first AI call for a task attempt
        (when session.prompt_snapshots is None). Only non-None fields are stored.

        Args:
            session: Mutable game session to update.
            trickster_prompts: Loaded prompt layers to snapshot.
            fourth_wall: Optional fourth wall prompt content to freeze.
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
        if trickster_prompts.mode_behaviour is not None:
            snapshot["mode_behaviour"] = trickster_prompts.mode_behaviour
        if fourth_wall is not None:
            snapshot["fourth_wall"] = fourth_wall

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
            mode_behaviour=session.prompt_snapshots.get("mode_behaviour"),
        )

    def get_fourth_wall_snapshot(
        self,
        session: GameSession,
    ) -> str | None:
        """Retrieves the snapshotted fourth wall prompt from the session.

        Args:
            session: Game session that may contain prompt snapshots.

        Returns:
            Fourth wall prompt content from snapshot, or None.
        """
        if session.prompt_snapshots is None:
            return None
        return session.prompt_snapshots.get("fourth_wall")

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
        persona_mode = cartridge.ai_config.persona_mode if cartridge.ai_config else None
        return self._loader.load_trickster_prompts(
            provider, task_id, persona_mode=persona_mode,
        )

    # -------------------------------------------------------------------
    # System prompt assembly — Dialogue
    # -------------------------------------------------------------------

    def _build_dialogue_system_prompt(
        self,
        prompts: TricksterPrompts,
        session: GameSession,
        cartridge: TaskCartridge,
        provider: str,
    ) -> str:
        """Builds the 8-layer system prompt for Trickster dialogue."""
        layers: list[str] = []

        # Layer 1-4: Prompt files
        self._append_prompt_layers(layers, prompts)

        # Layer 5: Task context (persona mode, phase, evaluation contract)
        layer5 = self._build_task_context(session, cartridge, provider)
        if layer5:
            layers.append(layer5)

        # Task history context (bounded prior task outcomes)
        history = self._build_task_history_context(session)
        if history:
            layers.append(history)

        # De-escalation context (between task and safety, conditional)
        deesc = self._build_deescalation_context(session, cartridge)
        if deesc:
            layers.append(deesc)

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
        fourth_wall_prompt: str | None = None,
    ) -> str:
        """Builds the system prompt for the debrief (reveal) call.

        Same layers 1-4 as dialogue, but layer 5 is debrief-specific:
        full EvaluationContract data plus a Lithuanian instruction to drop
        the adversarial stance and reveal manipulation techniques.

        When fourth_wall_prompt is provided, inserts a persona override
        (after prompt layers) and the fourth wall content (after debrief
        context) for the AI literacy moment.
        """
        layers: list[str] = []

        # Layer 1-4: Prompt files (same sources as dialogue)
        self._append_prompt_layers(layers, prompts)

        # Persona override for fourth wall (after persona layers, before
        # debrief context). Only injected when fourth wall is active.
        if fourth_wall_prompt is not None:
            layers.append(
                "## Persona per\u0117jimas\n\n"
                "Atskleidimo pabaigoje tu nustoji b\u016bti Makaronas. "
                "Tu esi dirbtinio intelekto sistema. Kalb\u0117k tiesiogiai "
                "kaip AI \u2014 ne kaip persona\u017eas."
            )

        # Layer 5: Debrief-specific context
        layer5 = self._build_debrief_context(cartridge)
        if layer5:
            layers.append(layer5)

        # Fourth wall AI literacy content (after debrief context,
        # before safety config — the culminating moment).
        if fourth_wall_prompt is not None:
            layers.append(fourth_wall_prompt)

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
        if prompts.mode_behaviour is not None:
            layers.append(prompts.mode_behaviour)
        if prompts.behaviour is not None:
            layers.append(prompts.behaviour)
        if prompts.safety is not None:
            layers.append(prompts.safety)
        if prompts.task_override is not None:
            layers.append(prompts.task_override)

    def _build_task_context(
        self,
        session: GameSession,
        cartridge: TaskCartridge,
        provider: str,
    ) -> str:
        """Builds layer 5: task context with evaluation contract.

        Branches on cartridge.is_clean: clean tasks get clean-specific
        framing with the clean task prompt; adversarial tasks get the
        standard patterns/checklist framing. Appends generated artifacts
        context when the session has generation artifacts.
        """
        if cartridge.is_clean:
            base = self._build_clean_task_context(
                session, cartridge, provider,
            )
        else:
            base = self._build_adversarial_task_context(session, cartridge)

        artifacts = self._build_generation_artifacts_context(session)
        if artifacts:
            base = base + "\n\n" + artifacts
            creation_eval = self._loader.load_creation_eval_prompt(provider)
            if creation_eval:
                base = base + "\n\n" + creation_eval
            else:
                logger.warning(
                    "Creation eval prompt not found for provider '%s'; "
                    "creation evaluation context will lack coaching framework.",
                    provider,
                )
        return base

    def _build_clean_task_context(
        self,
        session: GameSession,
        cartridge: TaskCartridge,
        provider: str,
    ) -> str:
        """Builds layer 5 for clean tasks: legitimacy defense framing."""
        parts: list[str] = []
        parts.append("## Svaraus turinio kontekstas")

        if cartridge.ai_config is not None:
            parts.append(f"\nPersona: {cartridge.ai_config.persona_mode}")

        if session.current_phase is not None:
            parts.append(f"Faz\u0117: {session.current_phase}")

        # Load clean task prompt content.
        clean_prompt = self._loader.load_clean_task_prompt(provider)
        if clean_prompt:
            parts.append(f"\n{clean_prompt}")
        else:
            logger.warning(
                "Clean task prompt file not found for provider '%s'; "
                "clean task context will lack specific instructions.",
                provider,
            )

        # Pass conditions (inverted semantics for clean tasks).
        parts.append("\n### Vertinimo s\u0105lygos")
        pc = cartridge.evaluation.pass_conditions
        parts.append(
            f"- Triksteris laimi: {pc.trickster_wins}\n"
            f"- I\u0161 dalies: {pc.partial}\n"
            f"- Triksteris pralaimi: {pc.trickster_loses}"
        )

        return "\n".join(parts)

    @staticmethod
    def _build_adversarial_task_context(
        session: GameSession,
        cartridge: TaskCartridge,
    ) -> str:
        """Builds layer 5 for adversarial tasks: patterns/checklist framing."""
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
                    f"   Ry\u0161ys su realybe: {pattern.real_world_connection}"
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
    def _build_generation_artifacts_context(
        session: GameSession,
    ) -> str | None:
        """Builds generated artifacts context for Trickster evaluation.

        Returns a Lithuanian-language context block when
        session.generated_artifacts is non-empty, None otherwise.
        Each artifact is shown chronologically so the Trickster can
        reference the student's creative evolution. Includes context
        fencing to prevent verbatim echo.
        """
        if not session.generated_artifacts:
            return None

        lines = [
            "### Mokinio sukurtas turinys",
            "",
        ]

        for i, artifact in enumerate(session.generated_artifacts, 1):
            lines.append(f"**Bandymas {i}:**")
            lines.append(
                f"Mokinio nurodymas: \u201e"
                f"{artifact.get('student_prompt', '')}\u201c"
            )

            if artifact.get("safety_redacted"):
                lines.append(
                    "Sugeneruotas turinys: "
                    "(saugumo sistema pakeit\u0117 turin\u012f)"
                )
            else:
                lines.append(
                    f"Sugeneruotas turinys: \u201e"
                    f"{artifact.get('generated_text', '')}\u201c"
                )

            lines.append("")

        lines.append(
            "INSTRUKCIJA: Vertink mokinio k\u016brin\u012f remdamasis "
            "auk\u0161\u010diau pateiktais bandymais. "
            "Gali komentuoti mokinio nurodymus ir sugeneruot\u0105 "
            "turin\u012f (pvz., \u201eTavo antras bandymas buvo daug "
            "konkretesnis\u201c). NIEKADA necituok sugeneruoto turinio "
            "pa\u017eod\u017eiui \u2014 referuok j\u012f savo "
            "\u017eod\u017eiais."
        )

        return "\n".join(lines)

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

    @staticmethod
    def _build_deescalation_context(
        session: GameSession,
        cartridge: TaskCartridge,
    ) -> str | None:
        """Builds de-escalation instruction when prior turn was too intense.

        Checks the last entry in session.turn_intensities against the
        cartridge's intensity ceiling. Returns Lithuanian de-escalation
        text when exceeded, None otherwise. Pure read — does not modify
        session state.
        """
        if not session.turn_intensities:
            return None

        last_score = session.turn_intensities[-1]
        ceiling = cartridge.safety.intensity_ceiling

        if last_score <= ceiling:
            return None

        return (
            "## De-eskalacijos instrukcija\n\n"
            f"Tavo ankstesnis atsakymas buvo per intensyvus "
            f"(vir\u0161ijo {ceiling}/5 rib\u0105). "
            "I\u0161laikyk savo persona\u017e\u0105 ir adversarin\u012f "
            "vaidmen\u012f, bet suma\u017eink konfrontacin\u0119 kalb\u0105. "
            "Vietoj tiesioginiu puolimu naudok subtilesnius metodus \u2014 "
            "retorinius klausimus, netiesiogin\u012f spaudim\u0105, "
            "nutyl\u0117jimus. Tikslas \u2014 mokyti mokin\u012f m\u0105styti "
            "kriti\u0161kai, ne j\u012f u\u017ego\u017eti."
        )

    @staticmethod
    def _build_task_history_context(session: GameSession) -> str | None:
        """Builds task history context layer from prior task outcomes.

        Returns None when session.task_history is empty. Bounded to the
        3 most recent entries (oldest dropped). Summary is factual from
        structured data (Framework P19), with explicit context fencing
        instructions.
        """
        if not session.task_history:
            return None

        recent = session.task_history[-_MAX_HISTORY_TASKS:]

        lines = [
            "## Ankstesni\u0173 u\u017eduo\u010di\u0173 kontekstas",
            "",
        ]

        for entry in recent:
            outcome_raw = entry.get("evaluation_outcome", "")
            label = _OUTCOME_LABELS.get(outcome_raw, outcome_raw)
            exchanges = entry.get("exchange_count", "?")
            parts = [f"{label} ({exchanges} apsikeitimai)"]

            if entry.get("is_clean"):
                parts.append("(\u0161varus turinys)")

            intensity = entry.get("intensity_score")
            if intensity is not None:
                parts.append(f"(intensyvumas: {intensity}/5)")

            lines.append(f"- {' '.join(parts)}")

        lines.append("")
        lines.append(
            "INSTRUKCIJA: Gali remtis mokinio pedagoginiais d\u0117sniais "
            "(pvz., \u201eTu v\u0117l greitai nuspr\u0119dei\u201c arba "
            "\u201e\u0160\u012f kart\u0105 buvai atidesnis\u201c). "
            "NIEKADA nemini u\u017eduo\u010di\u0173 pavadinim\u0173, tem\u0173, "
            "technik\u0173 ar turinio i\u0161 ankstesni\u0173 u\u017eduo\u010di\u0173. "
            "Mokinio augimo tendencijos yra svarbios; konkret\u016bs ankstesni\u0173 "
            "u\u017eduo\u010di\u0173 motyvai \u2014 ne."
        )

        return "\n".join(lines)

    # -------------------------------------------------------------------
    # Multimodal image extraction
    # -------------------------------------------------------------------

    def _extract_visible_images(
        self,
        cartridge: TaskCartridge,
        session: GameSession,
    ) -> list[tuple[str, str, str]]:
        """Extracts base64-encoded images from the current phase's visible_blocks.

        Resolves image files from disk, base64-encodes them, and returns
        tuples of (block_id, media_type, base64_data). Skips non-image
        blocks and images that can't be read from disk.

        Args:
            cartridge: Task cartridge with presentation_blocks.
            session: Current session (for current_phase).

        Returns:
            List of (block_id, media_type, base64_data) tuples.
        """
        if self._content_dir is None or session.current_phase is None:
            return []

        # Find the current phase object.
        current_phase = None
        for phase in cartridge.phases:
            if phase.id == session.current_phase:
                current_phase = phase
                break
        if current_phase is None or not current_phase.visible_blocks:
            return []

        # Build block lookup.
        block_map = {b.id: b for b in cartridge.presentation_blocks}

        results: list[tuple[str, str, str]] = []
        for block_id in current_phase.visible_blocks:
            block = block_map.get(block_id)
            if block is None:
                continue

            # Determine file path and optional text overlay.
            if isinstance(block, ImageBlock):
                src = block.src
                text_parts: list[dict[str, Any]] = []
            elif isinstance(block, MemeBlock):
                src = block.image_src
                text_parts = []
                overlay_parts = []
                if block.top_text:
                    overlay_parts.append(block.top_text)
                if block.bottom_text:
                    overlay_parts.append(block.bottom_text)
                if overlay_parts:
                    text_parts.append({
                        "type": "text",
                        "text": f"Meme tekstas: {' / '.join(overlay_parts)}",
                    })
            else:
                continue

            # Resolve file path with defense-in-depth.
            asset_path = (
                self._content_dir / "tasks" / cartridge.task_id / "assets" / src
            )
            resolved = asset_path.resolve()
            assets_base = (
                self._content_dir / "tasks" / cartridge.task_id / "assets"
            ).resolve()
            if not resolved.is_relative_to(assets_base):
                logger.warning(
                    "Image path traversal blocked: %s (task %s)",
                    src, cartridge.task_id,
                )
                continue

            # Read and encode.
            try:
                raw_bytes = asset_path.read_bytes()
            except OSError:
                logger.warning(
                    "Image file not found or unreadable: %s (task %s, block %s)",
                    asset_path, cartridge.task_id, block_id,
                )
                continue

            suffix = asset_path.suffix.lower()
            media_type = _IMAGE_MEDIA_TYPES.get(suffix)
            if media_type is None:
                logger.warning(
                    "Unknown image extension '%s' for block %s (task %s)",
                    suffix, block_id, cartridge.task_id,
                )
                continue

            b64_data = base64.b64encode(raw_bytes).decode()

            # For MemeBlock, store text_parts as extra data on the tuple.
            # We'll handle it in _build_image_context_message.
            results.append((block_id, media_type, b64_data))

            # If MemeBlock had text overlay, add a text marker.
            if isinstance(block, MemeBlock) and text_parts:
                # Store text parts by appending a special text-only entry.
                # Convention: media_type="" signals a text-only part.
                for tp in text_parts:
                    results.append((block_id, "", tp["text"]))

        return results

    @staticmethod
    def _build_image_context_message(
        image_parts: list[tuple[str, str, str]],
    ) -> dict[str, Any]:
        """Builds a multimodal user message with image content parts.

        Args:
            image_parts: List of (block_id, media_type, data) tuples.
                media_type="" signals a text-only part (e.g., meme overlay).

        Returns:
            Provider-neutral multimodal message dict.
        """
        content: list[dict[str, Any]] = [
            {"type": "text", "text": "U\u017eduoties vaizdin\u0117 med\u017eiaga:"},
        ]
        for _block_id, media_type, data in image_parts:
            if media_type == "":
                # Text-only part (meme overlay text).
                content.append({"type": "text", "text": data})
            else:
                content.append({
                    "type": "image",
                    "media_type": media_type,
                    "data": data,
                })
        return {"role": "user", "content": content}

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
        messages: list[dict[str, Any]],
        context_prefix_count: int = 0,
    ) -> list[dict[str, Any]]:
        """Trims oldest exchange pairs if total exceeds token budget.

        Uses character-based heuristic (~3 chars/token for Lithuanian).
        Removes complete exchange pairs (user + assistant together) from
        the exchange history to maintain conversation coherence.

        The system prompt and context prefix messages (e.g., image context)
        are NEVER trimmed.

        Args:
            system_prompt: System prompt text.
            messages: Full message list (may include context prefix + exchanges).
            context_prefix_count: Number of leading messages to protect from
                trimming (e.g., 1 for image context message).
        """
        system_tokens = len(system_prompt) / _CHARS_PER_TOKEN
        message_tokens = sum(
            self._estimate_message_tokens(m) for m in messages
        )
        total = system_tokens + message_tokens

        if total <= self._token_budget:
            return messages

        overage = total - self._token_budget

        # Split into protected prefix and trimmable exchange history.
        prefix = messages[:context_prefix_count]
        exchange_msgs = list(messages[context_prefix_count:])

        while overage > 0 and len(exchange_msgs) >= 2:
            # Remove the oldest pair (user + assistant).
            pair_tokens = (
                self._estimate_message_tokens(exchange_msgs[0])
                + self._estimate_message_tokens(exchange_msgs[1])
            )
            exchange_msgs = exchange_msgs[2:]
            overage -= pair_tokens

        trimmed = prefix + exchange_msgs
        logger.debug(
            "Trimmed %d messages from exchange history (budget=%d tokens)",
            len(messages) - len(trimmed),
            self._token_budget,
        )

        return trimmed

    @staticmethod
    def _estimate_message_tokens(message: dict[str, Any]) -> float:
        """Estimates token count for a single message.

        Handles both text-only messages (content is str) and multimodal
        messages (content is list of parts).
        """
        content = message["content"]
        if isinstance(content, str):
            return len(content) / _CHARS_PER_TOKEN

        # Multimodal: sum text chars + fixed cost per image.
        tokens = 0.0
        for part in content:
            if part.get("type") == "text":
                tokens += len(part.get("text", "")) / _CHARS_PER_TOKEN
            elif part.get("type") == "image":
                tokens += _TOKENS_PER_IMAGE
        return tokens
