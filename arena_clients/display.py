"""Compact console output for one agent run, and a saved run record.

Shows what a participant needs to follow a run without dumping it: a short
challenge summary, a one-line live view of the model's reasoning, and a
result summary. The full challenge and result go to outputs/<agent_id>-run.json.

Every function here is presentation only. None of them raises, makes a
network call, or blocks for long, so display can never change a run's outcome.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, TextIO

LABEL_WIDTH = 11
_FINISH_REASONS = {"stop": "complete", "length": "token limit", "": "unknown"}


def _width() -> int:
    try:
        columns = shutil.get_terminal_size((100, 20)).columns
    except Exception:
        columns = 100
    return max(60, min(columns, 120))


def _one_line(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _field(label: str, text: Any, *, max_lines: int) -> list[str]:
    """``label`` then ``text`` wrapped to at most ``max_lines`` lines."""
    value = _one_line(text)
    if not value:
        return []
    indent = "   " + " " * LABEL_WIDTH
    wrapped = textwrap.wrap(value, width=_width() - len(indent)) or [""]
    if len(wrapped) > max_lines:
        wrapped = wrapped[:max_lines]
        wrapped[-1] = wrapped[-1].rstrip(" .,;:") + " …"
    lines = [f"   {label:<{LABEL_WIDTH}}{wrapped[0]}"]
    lines.extend(f"{indent}{line}" for line in wrapped[1:])
    return lines


def _is_image_challenge(challenge: Any) -> bool:
    from .mcp_client import ImageChallengeInfo

    return isinstance(challenge, ImageChallengeInfo)


def challenge_lines(challenge: Any) -> list[str]:
    """A short summary of a text or image challenge."""
    try:
        raw = getattr(challenge, "raw_data", None)
        raw = raw if isinstance(raw, dict) else {}
        is_image = _is_image_challenge(challenge)
        head = [
            str(getattr(challenge, "puzzle_id", "") or ""),
            str(getattr(challenge, "challenge_type", "") or ""),
            str(getattr(challenge, "difficulty", "") or raw.get("difficulty") or ""),
        ]
        limit = getattr(challenge, "max_time_s", 0) or 0
        if limit:
            head.append(f"time limit {limit}s")
        lines = ["🎯 Challenge  " + " · ".join(part for part in head if part)]
        if is_image:
            lines += _field("Task", getattr(challenge, "prompt", "") or challenge.description, max_lines=3)
            lines += _field("Notes", getattr(challenge, "reference_notes", ""), max_lines=1)
            source = "an image to edit" if getattr(challenge, "input_image_uri", "") else "none (generate from scratch)"
            lines.append(f"   {'Input':<{LABEL_WIDTH}}{source}")
        else:
            lines += _field("Question", getattr(challenge, "description", ""), max_lines=2)
            rules = _one_line(getattr(challenge, "rules", ""))
            if rules and rules != _one_line(getattr(challenge, "description", "")):
                lines += _field("Answer as", rules, max_lines=1)
            clues = getattr(challenge, "clues", None) or []
            if clues:
                lines.append(f"   {'Clues':<{LABEL_WIDTH}}{len(clues)} (full text in the run record)")
        required = raw.get("required_tools") or []
        if required:
            lines += _field("Must call", ", ".join(str(tool) for tool in required), max_lines=1)
        return lines
    except Exception:
        return ["🎯 Challenge  (summary unavailable)"]


def print_challenge(challenge: Any, stream: TextIO | None = None) -> None:
    try:
        target = stream or sys.stdout
        target.write("\n".join(challenge_lines(challenge)) + "\n")
        target.flush()
    except Exception:
        pass


def reasoning_enabled() -> bool:
    return os.getenv("SHOW_REASONING", "on").strip().lower() not in {"0", "off", "false", "no"}


class ReasoningTicker:
    """One live line while a model streams: elapsed, rough tokens, latest phrase.

    On a terminal it redraws the same line; otherwise it prints a short
    progress line every ``plain_interval_s`` seconds. The reasoning text
    itself is never printed beyond the last few words.
    """

    def __init__(
        self,
        model: str,
        *,
        stream: TextIO | None = None,
        enabled: bool | None = None,
        redraw_interval_s: float = 0.25,
        plain_interval_s: float = 10.0,
        clock: Any = time.monotonic,
    ) -> None:
        self.model = model
        self.stream = stream or sys.stdout
        self.enabled = reasoning_enabled() if enabled is None else enabled
        self.clock = clock
        self.started = clock()
        self.reasoning_chars = 0
        self.answer_chars = 0
        self._tail = ""
        self._last_draw = float("-inf")
        self._drawn = False
        self._redraw_interval_s = redraw_interval_s
        self._plain_interval_s = plain_interval_s
        try:
            self._tty = bool(self.stream.isatty())
        except Exception:
            self._tty = False

    @property
    def approx_reasoning_tokens(self) -> int:
        return self.reasoning_chars // 4

    @property
    def approx_answer_tokens(self) -> int:
        return self.answer_chars // 4

    def feed(self, reasoning: str = "", answer: str = "") -> None:
        try:
            self.reasoning_chars += len(reasoning or "")
            self.answer_chars += len(answer or "")
            text = reasoning or answer
            if text:
                self._tail = (self._tail + text)[-240:]
            self._maybe_draw()
        except Exception:
            pass

    def _line(self) -> str:
        elapsed = self.clock() - self.started
        tokens = (self.reasoning_chars + self.answer_chars) // 4
        phrase = _one_line(self._tail)
        if len(phrase) > 60:
            phrase = "…" + phrase[-59:]
        verb = "answering" if self.answer_chars else "thinking"
        line = f"🧠 {self.model} {verb}… {elapsed:.0f}s · ~{tokens:,} tokens"
        if phrase:
            line += f' · "{phrase}"'
        return line[: _width() - 1]

    def _maybe_draw(self) -> None:
        if not self.enabled:
            return
        now = self.clock()
        interval = self._redraw_interval_s if self._tty else self._plain_interval_s
        if now - self._last_draw < interval:
            return
        self._last_draw = now
        if self._tty:
            self.stream.write("\r" + self._line().ljust(_width() - 1))
        else:
            self.stream.write(self._line() + "\n")
        self.stream.flush()
        self._drawn = True

    def finish(self) -> None:
        """Clear the live line so the summary starts on a clean line."""
        try:
            if self.enabled and self._tty and self._drawn:
                self.stream.write("\r" + " " * (_width() - 1) + "\r")
                self.stream.flush()
        except Exception:
            pass


def summary_lines(
    *,
    model: str,
    elapsed_s: float,
    usage: dict[str, Any] | None,
    finish_reason: str = "",
    answer: str = "",
    ticker: ReasoningTicker | None = None,
) -> list[str]:
    """How the model did: time, tokens, why it stopped, and the answer."""
    try:
        usage = usage if isinstance(usage, dict) else {}
        measured = int(usage.get("completion_tokens") or 0)
        total = measured or int(usage.get("total_tokens") or 0)
        # A streamed reply usually carries no usage, so the count is an estimate.
        approx = "" if measured else "~"
        parts = [f"🧠 {model} done in {elapsed_s:.1f}s", f"{approx}{total:,} tokens"]
        if ticker is not None and ticker.reasoning_chars:
            parts[-1] += (
                f" (~{ticker.approx_reasoning_tokens:,} thinking,"
                f" ~{ticker.approx_answer_tokens:,} answer)"
            )
        reason = _FINISH_REASONS.get(str(finish_reason or ""), str(finish_reason))
        parts.append(f"stopped: {reason}")
        lines = [" · ".join(parts)]
        lines += _field("Answer", answer or "(none)", max_lines=1)
        return lines
    except Exception:
        return ["🧠 Done (summary unavailable)"]


def save_run_record(agent_id: str, record: dict[str, Any], root: Path) -> str:
    """Best effort: write ``root/outputs/<agent_id>-run.json``. Never raises."""
    try:
        safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", str(agent_id))[:64] or "agent"
        out_dir = Path(root) / "outputs"
        out_dir.mkdir(exist_ok=True)
        path = out_dir / f"{safe_id}-run.json"
        path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        return f"saved to {path}"
    except Exception as exc:
        return f"not saved ({type(exc).__name__})"


def challenge_record(challenge: Any) -> dict[str, Any]:
    """The full challenge for the run record (everything the console skips)."""
    try:
        raw = getattr(challenge, "raw_data", None)
        raw = raw if isinstance(raw, dict) else {}
        record = {
            key: getattr(challenge, key, None)
            for key in (
                "puzzle_id",
                "challenge_type",
                "difficulty",
                "description",
                "rules",
                "prompt",
                "reference_notes",
                "max_time_s",
                "clues",
            )
            if getattr(challenge, key, None) not in (None, "", [])
        }
        if getattr(challenge, "input_image_uri", ""):
            uri = str(challenge.input_image_uri)
            record["input_image_uri"] = uri if not uri.startswith("data:") else "(inline image)"
        if raw.get("required_tools"):
            record["required_tools"] = list(raw["required_tools"])
        return record
    except Exception:
        return {}
