"""Competitor strategy for the Gauntlet Explorer agent.

The explorer runtime (gauntlet/runner.py) calls the same hooks the bundled
examples call: rank_models, pick_model, build_system_prompt,
build_solver_prompt, get_llm_params, on_time_warning, plan_image_tool,
build_image_prompt. The bundled examples also import this file, so they get
the better model ranking for free.
"""

from __future__ import annotations

import re
from typing import Any

from base_strategy import BaseStrategy, ChallengeContext

try:
    from gauntlet.models import rank_image_models, rank_text_models
except ImportError:  # examples packaged without the gauntlet package
    rank_image_models = None  # type: ignore[assignment]
    rank_text_models = None  # type: ignore[assignment]

try:
    from model_selector import cached_model_rows
except ImportError:
    def cached_model_rows() -> list[dict[str, Any]]:  # type: ignore[misc]
        return []


class MyStrategy(BaseStrategy):
    """Explorer strategy: strongest model that fits the clock, strict answers."""

    # AGENT_ID must come from .env (or the organizer's runner). An empty default
    # makes the agent refuse to run on a shared server without one.
    agent_id = ""
    agent_name = "AHGroup"

    text_system_prompt = (
        "You solve timed text challenges: logic, reasoning, retrieval, and structured output. "
        "Work through every clue and constraint, check the result against each one, then answer. "
        "Your first line must be: ANSWER: <final answer>. "
        "Match the required output format exactly (case, separators, ordering, JSON shape). "
        "Never answer 'unknown'; give your best concrete answer."
    )
    text_strategy_notes = "Verify against every clue before answering."
    text_temperature = 0.0
    text_max_tokens = 8192

    image_strategy_notes = (
        "Prefer image_edit when an input image exists. Change only what the instruction names, "
        "keep composition, identity, colors and lighting elsewhere. Standard resolution only."
    )

    # -- models -------------------------------------------------------------

    def rank_models(self, ctx: ChallengeContext, available_models: list[str]) -> list[str]:
        if not available_models:
            return []
        rows = cached_model_rows()
        if self.is_image_challenge(ctx):
            if rank_image_models is not None:
                ranked = rank_image_models(available_models, rows)
                if ranked:
                    return ranked
            preferred = self.prefer_image_models(available_models)
            return preferred or list(available_models)
        if rank_text_models is None:
            return list(available_models)
        scored = rank_text_models(
            available_models,
            max_time_s=float(ctx.max_time_s or 0),
            difficulty=ctx.difficulty,
            rows=rows,
        )
        return [m.name for m in scored]

    def pick_model(self, stage: str, ranked_models: list[str], ctx: ChallengeContext) -> str:
        if not ranked_models:
            return ""
        if stage in {"fast", "repair", "format"}:
            return ranked_models[min(len(ranked_models) - 1, 2)]
        return ranked_models[0]

    # -- prompts ------------------------------------------------------------

    def build_solver_prompt(self, ctx: ChallengeContext) -> str:
        clues = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(ctx.clues) if c.strip()) or "(none)"
        return (
            f"Challenge type: {ctx.challenge_type or 'text'} (difficulty: {ctx.difficulty})\n"
            f"Description:\n{ctx.description.strip() or '(none)'}\n\n"
            f"Rules and output format:\n{ctx.rules.strip() or '(none)'}\n\n"
            f"Clues:\n{clues}\n\n"
            "Solve it. Use every clue. Before answering, check the answer against each clue and the rules.\n"
            "Write list items as bare names or IDs (no shared label words such as 'Team' or 'Event'),\n"
            "unless the rules show the label.\n"
            "First line: ANSWER: <final answer>. At most two short lines of justification after it."
        )

    def get_llm_params(self, ctx: ChallengeContext) -> dict[str, Any]:
        max_tokens = self.text_max_tokens
        if ctx.max_time_s and ctx.max_time_s <= 60:
            max_tokens = 6144
        elif "hard" in (ctx.difficulty or "").lower() and ctx.max_time_s >= 120:
            max_tokens = 12288
        return {"temperature": float(self.text_temperature), "max_tokens": int(max_tokens)}

    def build_image_prompt(self, ctx: ChallengeContext) -> str:
        """Kit prompt plus explicit constraints. The image judge scores constraint
        compliance separately; Practice docked a poster to 70 on that axis."""
        base = super().build_image_prompt(ctx)
        text = f"{ctx.rules} {ctx.description}".lower()
        extras = []
        if re.search(r"\bno (visible )?(text|words|letters|typography|lettering)\b|avoid[^.]{0,40}\b(text|logo|watermark)|without (any )?text", text):
            extras.append("Absolutely no text of any kind anywhere in the image: no letters, words, numbers, "
                          "signage, captions, labels, logos, brand marks, or watermarks. Surfaces that would carry "
                          "text (signs, screens, posters) stay blank or abstract.")
        if ctx.image_url:
            extras.append("Change only what the instruction asks for. Keep everything else identical to the input "
                          "image: subject identity, facial features, pose, camera angle, framing, background, "
                          "lighting, and colors. No style transfer, no added or removed objects.")
        if re.search(r"single (coherent )?scene|clean composition", text):
            extras.append("One coherent scene with one clear focal subject and a clean composition.")
        return base + ("\n\n" + "\n".join(extras) if extras else "")

    def on_time_warning(self, remaining_s: float, current_answer: str, ctx: ChallengeContext) -> str | None:
        return current_answer or None
