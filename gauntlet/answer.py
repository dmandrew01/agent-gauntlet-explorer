"""Answer extraction and output-format enforcement.

Quality is judged against "the puzzle's expected result and required format",
so a right answer in the wrong shape loses points. This module reads the
format the rules ask for and repairs the obvious mismatches without an LLM.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .llm import parse_json_loose

_ANSWER_LINE = re.compile(r"^\s*[#>*_\s]*(?:final\s+answer|answer)\s*\**\s*[:=]\s*\**\s*(.*)$", re.I)
_THINK = re.compile(r"<think>.*?(?:</think>|$)", re.S | re.I)
_BAD_PREFIX = ("<toolcall", "toolcall", "<think", "reasoning:", "analysis:", "action:", "thought:")


@dataclass
class FormatSpec:
    json: bool = False
    json_keys: list[str] = field(default_factory=list)
    single_word: bool = False
    number: bool = False
    integer: bool = False
    upper: bool = False
    lower: bool = False
    yes_no: bool = False
    list_sep: str = ""
    max_words: int = 0
    template: str = ""
    exact_choices: list[str] = field(default_factory=list)

    def describe(self) -> str:
        parts = []
        if self.json:
            parts.append("valid JSON" + (f" with keys {self.json_keys}" if self.json_keys else ""))
        if self.single_word:
            parts.append("a single word")
        if self.integer:
            parts.append("an integer only")
        elif self.number:
            parts.append("a number only")
        if self.upper:
            parts.append("UPPERCASE")
        if self.lower:
            parts.append("lowercase")
        if self.yes_no:
            parts.append("yes or no")
        if self.list_sep:
            parts.append(f"items separated by {self.list_sep!r}")
        if self.max_words:
            parts.append(f"at most {self.max_words} words")
        if self.template:
            parts.append(f"shaped like {self.template!r}")
        if self.exact_choices:
            parts.append("one of " + ", ".join(self.exact_choices[:10]))
        return "; ".join(parts) or "as the rules state"


def infer_format(rules: str, description: str = "") -> FormatSpec:
    text = f"{rules}\n{description}"
    low = text.lower()
    spec = FormatSpec()
    if re.search(r"\bjson\b", low):
        spec.json = True
        keys = re.findall(r"[\"'`]([A-Za-z_][A-Za-z0-9_]{0,40})[\"'`]\s*:", text)
        keys += re.findall(r"keys?\s+[\"'`]?([A-Za-z_][A-Za-z0-9_]*)[\"'`]?(?:\s*(?:,|and)\s*[\"'`]?([A-Za-z_][A-Za-z0-9_]*)[\"'`]?)*", text)  # type: ignore[arg-type]
        flat: list[str] = []
        for k in keys:
            for item in (k if isinstance(k, tuple) else (k,)):
                if item and item not in flat and item.lower() not in {"json", "with", "the", "and"}:
                    flat.append(item)
        spec.json_keys = flat[:12]
    if re.search(r"\b(single|one)[ -]word\b", low):
        spec.single_word = True
    if re.search(r"\b(integer|whole number)\b(?! of)", low) and re.search(r"\b(only|just|answer (is|with|as)|respond with)\b", low):
        spec.integer = True
    elif re.search(r"\b(number|numeric value|digits?)\s+only\b|\bonly (the|a) number\b", low):
        spec.number = True
    # Case rules apply to the whole answer only when they say so; "Use uppercase for
    # ARCH_FAMILY" (event web_search_001) covers one field, not the payload.
    whole = r"(answer|respond|reply|output|return|write|submit)\w*\W+(?:\w+\W+){0,4}?(in )?"
    if re.search(whole + r"(uppercase|upper case|all caps|capital letters)\b", low) and not re.search(r"(uppercase|upper case) for\b", low):
        spec.upper = True
    elif re.search(whole + r"(lowercase|lower case)\b", low) and not re.search(r"(lowercase|lower case) for\b", low):
        spec.lower = True
    if re.search(r"\b(yes or no|yes/no|true or false)\b", low):
        spec.yes_no = True
    if re.search(r"comma[- ]separated|separated by commas", low):
        spec.list_sep = ", "
    elif re.search(r"(pipe|\|)[- ]separated|separated by (a )?pipes?", low):
        spec.list_sep = "|"
    m = re.search(r"(?:at most|no more than|maximum of|max(?:imum)?)\s+(\d{1,3})\s+words", low)
    if m:
        spec.max_words = int(m.group(1))
    tm = re.search(r"(?:format|form|exactly like|shaped like|e\.g\.|example)\s*[:\-]?\s*[`\"']([^`\"'\n]{2,120})[`\"']", text, re.I)
    if tm:
        spec.template = tm.group(1).strip()
    cm = re.search(r"(?:one of|choose from|must be)\s*[:\-]?\s*((?:[\"'`][^\"'`]{1,40}[\"'`]\s*(?:,|or)?\s*){2,12})", text, re.I)
    if cm:
        spec.exact_choices = re.findall(r"[\"'`]([^\"'`]{1,40})[\"'`]", cm.group(1))
    return spec


def is_invalid(answer: str | None) -> bool:
    text = str(answer or "").strip()
    if not text:
        return True
    low = text.lower()
    if low in {"unknown", "n/a", "none", "null", "i don't know", "idk", "cannot determine"}:
        return True
    if low.startswith(_BAD_PREFIX):
        return True
    return any(tok in low for tok in ("<toolcall", "</toolcall", '"tool_calls"', "function_call"))


def extract_answer(raw: str, spec: FormatSpec | None = None) -> str:
    """Pull the final answer out of model output."""
    text = _THINK.sub("", raw or "").strip()
    if not text:
        return ""
    lines = text.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        m = _ANSWER_LINE.match(lines[i])
        if not m:
            continue
        first = m.group(1).strip()
        if spec and spec.json and (not first or first[:1] in "{[`"):
            tail = "\n".join([first] + lines[i + 1 :])
            obj = parse_json_loose(tail)
            if obj is not None:
                return json.dumps(obj, ensure_ascii=False, separators=(", ", ": "))
        if first:
            return _strip_wrapping(first)
        if i + 1 < len(lines) and lines[i + 1].strip():
            return _strip_wrapping(lines[i + 1].strip())
    if spec and spec.json:
        obj = parse_json_loose(text)
        if obj is not None:
            return json.dumps(obj, ensure_ascii=False, separators=(", ", ": "))
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return boxed[-1].strip()
    for line in reversed(lines):
        line = line.strip()
        if line and len(line) < 400 and not is_invalid(line):
            return _strip_wrapping(line)
    return ""


def _strip_wrapping(text: str) -> str:
    text = text.strip()
    if text.startswith("**") and text.endswith("**") and len(text) > 4:
        text = text[2:-2].strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "`\"'":
        text = text[1:-1].strip()
    return text.rstrip()


GENERIC_PREFIXES = {
    "team", "group", "player", "person", "event", "house", "room", "station", "candidate",
    "runner", "car", "box", "door", "city", "project", "task", "step", "stage", "level",
    "demo", "booth", "table", "seat", "slot", "day", "plan", "option", "item", "building",
}


def strip_shared_prefix(answer: str) -> str:
    """'Team Alpha, Team Bravo' -> 'Alpha, Bravo'.

    Practice: a judge docked an otherwise perfect order to 80 for the "Team "
    prefix ("return only the team names"), after another judge run accepted it.
    Only a generic category word shared by every item gets removed.
    """
    for sep in (", ", ",", " -> ", " > ", "; ", "|"):
        parts = [p.strip() for p in answer.split(sep)]
        if len(parts) >= 2 and all(parts):
            break
    else:
        return answer
    firsts = {p.split()[0].lower() for p in parts if len(p.split()) >= 2}
    if len(firsts) != 1 or any(len(p.split()) < 2 for p in parts):
        return answer
    word = firsts.pop().rstrip(":")
    if word not in GENERIC_PREFIXES:
        return answer
    return sep.join(p.split(None, 1)[1] for p in parts)


def enforce(answer: str, spec: FormatSpec) -> tuple[str, list[str]]:
    """Repair mechanical format issues. Returns (answer, remaining_problems)."""
    problems: list[str] = []
    out = (answer or "").strip()
    if not spec.json:
        out = strip_shared_prefix(out)
    if spec.json:
        obj = parse_json_loose(out)
        if obj is None:
            problems.append("answer is not valid JSON")
        else:
            out = json.dumps(obj, ensure_ascii=False, separators=(", ", ": "))
            if spec.json_keys and isinstance(obj, dict):
                missing = [k for k in spec.json_keys if k not in obj]
                if missing and len(missing) < len(spec.json_keys):
                    problems.append(f"JSON is missing keys {missing}")
        return out, problems
    if spec.yes_no:
        low = out.lower()
        for word in ("yes", "no", "true", "false"):
            if re.match(rf"^{word}\b", low):
                out = out[: len(word)]
                break
    if spec.integer:
        m = re.search(r"-?\d[\d,]*", out)
        if m:
            out = m.group(0).replace(",", "")
        else:
            problems.append("no integer in answer")
    elif spec.number:
        m = re.search(r"-?\d[\d,]*(?:\.\d+)?", out)
        if m:
            out = m.group(0).replace(",", "")
        else:
            problems.append("no number in answer")
    if spec.single_word:
        words = out.split()
        if len(words) > 1:
            out = words[0].strip(".,;:!?")
    if spec.max_words and len(out.split()) > spec.max_words:
        problems.append(f"more than {spec.max_words} words")
    if spec.upper:
        out = out.upper()
    elif spec.lower:
        out = out.lower()
    if spec.exact_choices:
        match = next((c for c in spec.exact_choices if c.lower() == out.lower()), None)
        if match:
            out = match
        else:
            near = next((c for c in spec.exact_choices if c.lower() in out.lower()), None)
            if near:
                out = near
            else:
                problems.append(f"answer is not one of {spec.exact_choices}")
    if out.endswith(".") and not spec.template.endswith(".") and len(out.split()) <= 6:
        out = out[:-1]
    return out, problems
