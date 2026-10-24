"""Model roster ranking.

Model choice is scored. Quality carries 70% of the score, speed 15%, tokens 5%.
So the ranker prefers the strongest text reasoner that fits the time limit,
and only trades down in quality when the clock is tight.

The roster is discovered at runtime from /proxy/models. Names below are
pattern fragments, not a promise about which aliases exist.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

# (pattern, quality 0-100, speed 0-100, reasoning?)
# Higher quality first within a family. Unknown names fall back to size parsing.
FAMILY_TABLE: list[tuple[str, int, int, bool]] = [
    (r"gpt-5(\.\d)?(?!.*(mini|nano))", 95, 45, True),
    (r"claude.*opus", 95, 40, True),
    (r"claude.*sonnet", 90, 60, True),
    (r"gemini.*(2\.5|3).*pro", 92, 45, True),
    (r"o[134](-pro)?(?!.*mini)", 92, 35, True),
    (r"deepseek.*(r1|reasoner|v3\.[1-9]|v4)", 88, 35, True),
    (r"qwen.*3.*(235b|480b|max)", 88, 45, True),
    (r"kimi.*k2", 87, 50, True),
    (r"glm.?4\.[5-9]|glm.?5", 85, 50, True),
    (r"nemotron.*ultra", 88, 65, True),  # Practice: 2.5 s median, 4/4
    (r"nemotron.*super", 84, 55, True),
    (r"llama.*nemotron.*(253b|ultra)", 88, 35, True),
    (r"llama.*nemotron.*(49b|super)", 84, 55, True),
    (r"nemotron.*(lightning|flash|nano|mini)", 72, 92, True),
    (r"qwen.*(flash|turbo)", 78, 88, False),
    (r"qwen.?3\.[5-9]-\d+b", 74, 80, True),
    (r"qwen.?3\.[5-9].*(plus|max|pro)", 88, 50, True),
    (r"gpt-oss.*120b", 86, 60, True),
    (r"gpt-oss.*20b", 74, 80, True),
    (r"gemini.*[3-9](\.\d)?.*flash(?!.*(lite|image))", 87, 84, True),
    (r"gemini.*flash(?!.*lite)", 82, 85, False),
    (r"gemini.*flash.*lite", 70, 92, False),
    (r"gpt-5.*mini|gpt-4\.1(?!.*(mini|nano))|gpt-4o(?!.*mini)", 82, 70, False),
    (r"claude.*haiku", 78, 85, False),
    (r"llama.*4.*maverick", 80, 70, False),
    (r"llama.*4.*scout", 72, 80, False),
    (r"llama.*3\.[13].*(405b)", 80, 40, False),
    (r"llama.*3\.[13].*70b", 74, 65, False),
    (r"mistral.*(large|medium)", 76, 65, False),
    (r"qwen.*(32b|72b|coder)", 78, 65, True),
    (r"nemotron.*nano", 66, 90, True),
    (r"(mini|nano|small|tiny|lite|8b|7b|3b|1b)\b", 55, 95, False),
]

_SIZE_RE = re.compile(r"(\d{1,4})(?:\.\d)?b\b", re.I)


@dataclass
class ModelScore:
    name: str
    quality: float
    speed: float
    reasoning: bool
    image: bool
    total: float = 0.0

    def as_row(self) -> str:
        kind = "img" if self.image else ("reason" if self.reasoning else "chat")
        return f"{self.name}  q={self.quality:.0f} s={self.speed:.0f} {kind} -> {self.total:.1f}"


def _row_caps(rows: list[dict[str, Any]] | None, name: str) -> set[str] | None:
    for row in rows or []:
        if str(row.get("id") or "").strip() == name and isinstance(row.get("capabilities"), list):
            return {str(c).strip().lower() for c in row["capabilities"]}
    return None


def is_image_model(name: str, rows: list[dict[str, Any]] | None = None) -> bool:
    caps = _row_caps(rows, name)
    if caps is not None:
        return "image" in caps or "image_generation" in caps
    alias = name.rsplit("/", 1)[-1].lower()
    return alias.endswith("-image") or "-image-" in alias or "image-gen" in alias or "dall-e" in alias or "imagen" in alias


def score_model(name: str, rows: list[dict[str, Any]] | None = None) -> ModelScore:
    low = name.lower()
    quality, speed, reasoning = 65.0, 65.0, False
    for pattern, q, s, r in FAMILY_TABLE:
        if re.search(pattern, low):
            quality, speed, reasoning = float(q), float(s), r
            break
    else:
        size = _SIZE_RE.search(low)
        if size:
            b = int(size.group(1))
            quality = min(88.0, 50.0 + 9.0 * (b ** 0.35))
            speed = max(25.0, 95.0 - 0.18 * b)
    if re.search(r"(think|reason|r1\b|qwq)", low):
        reasoning = True
    if re.search(r"(large|xl|max|ultra|pro)\b", low) and quality < 85:
        quality += 8
        speed -= 8
    if re.search(r"(instruct|chat)", low) and not reasoning:
        speed += 3
    version = re.search(r"(\d+(?:\.\d+)?)", low.rsplit("/", 1)[-1])
    if version:
        try:
            quality += min(1.0, float(version.group(1)) * 0.1)  # newer release breaks ties
        except ValueError:
            pass
    if re.search(r"(free|preview|exp)\b", low):
        speed -= 5  # shared or experimental endpoints queue more
    return ModelScore(name, quality, speed, reasoning, is_image_model(name, rows))


_STATS_CACHE: dict[str, Any] | None = None


def load_stats() -> dict[str, Any]:
    """Measured latency and accuracy from tools/calibrate_models.py, when present."""
    global _STATS_CACHE
    if _STATS_CACHE is not None:
        return _STATS_CACHE
    path = os.getenv("GAUNTLET_MODEL_STATS") or os.path.join(os.path.dirname(__file__), "model_stats.json")
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        _STATS_CACHE = data.get("models", {}) if isinstance(data, dict) else {}
    except (OSError, ValueError):
        _STATS_CACHE = {}
    return _STATS_CACHE


def _apply_stats(m: ModelScore, stats: dict[str, Any]) -> None:
    row = stats.get(m.name)
    if not isinstance(row, dict):
        return
    latency = row.get("median_latency_s")
    if isinstance(latency, (int, float)) and latency > 0:
        measured_speed = max(5.0, min(100.0, 100.0 - 6.0 * float(latency)))
        m.speed = 0.3 * m.speed + 0.7 * measured_speed
    acc = row.get("accuracy")
    if isinstance(acc, (int, float)) and row.get("trials", 0):
        m.quality = 0.6 * m.quality + 0.4 * (100.0 * float(acc))
    if row.get("errors", 0) and row.get("errors", 0) >= row.get("trials", 1):
        m.quality -= 40  # every calibration call failed: likely broken on this roster


def rank_text_models(
    available: list[str],
    *,
    max_time_s: float = 0.0,
    difficulty: str = "",
    rows: list[dict[str, Any]] | None = None,
) -> list[ModelScore]:
    """Rank text-capable models for a puzzle under a time limit."""
    pinned = [m.strip() for m in os.getenv("GAUNTLET_MODEL_PREFERENCE", "").split(",") if m.strip()]
    scored = [score_model(name, rows) for name in available]
    stats = load_stats()
    for m in scored:
        _apply_stats(m, stats)
    text_models = [m for m in scored if not m.image] or scored
    tight = 0 < max_time_s <= 60
    hard = "hard" in (difficulty or "").lower()
    # Quality is 70% of the score and speed 15%, so speed only breaks near-ties
    # unless the clock is tight.
    speed_weight = 0.25 if tight else (0.1 if hard else 0.15)
    for m in text_models:
        m.total = (1 - speed_weight) * m.quality + speed_weight * m.speed
        if tight and m.reasoning and m.speed < 50:
            m.total -= 5  # slow thinkers risk the deadline on 60s puzzles
        if m.reasoning and not tight:
            m.total += 3
        if hard and m.reasoning:
            m.total += 4
    for i, frag in enumerate(pinned):
        for m in text_models:
            if frag.lower() in m.name.lower():
                m.total += 100 - i
    text_models.sort(key=lambda m: m.total, reverse=True)
    return text_models


def rank_image_models(available: list[str], rows: list[dict[str, Any]] | None = None) -> list[str]:
    pinned = [m.strip() for m in os.getenv("GAUNTLET_IMAGE_MODEL_PREFERENCE", "").split(",") if m.strip()]
    image = [m for m in available if is_image_model(m, rows)]

    def key(name: str) -> tuple[int, int]:
        low = name.lower()
        pin = next((i for i, frag in enumerate(pinned) if frag.lower() in low), 999)
        quality = 0
        if re.search(r"(pro|ultra|gpt-image|imagen-?4|flux.*(pro|max))", low):
            quality = 2
        elif re.search(r"(flash|fast|turbo|mini|schnell)", low):
            quality = 1
        return (pin, -quality)

    return sorted(image, key=key)


def vision_capable(name: str) -> bool:
    """Best guess whether a chat model accepts image_url content parts."""
    low = name.lower()
    return bool(
        re.search(r"(gpt-4o|gpt-4\.1|gpt-5|claude|gemini|vision|vl\b|-vl-|llama.*4|pixtral|qwen.*vl|nemotron.*vl|omni)", low)
    )
