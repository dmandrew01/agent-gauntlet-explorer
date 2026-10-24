"""LLM proxy access: OpenAI-compatible chat with attribution headers.

- Keeps X-Agent-ID and X-Round-ID on every call (token telemetry needs both).
- Runs the blocking SDK in a worker thread so the MCP SSE stream stays alive.
- Caps every request by the time left on the puzzle clock.
- Remembers which models reject native tool calls and falls back to a JSON
  action protocol for those.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from arena_clients.proxy_headers import build_proxy_headers

try:
    from openai import OpenAI
except ImportError:  # the kit's lock has openai; this keeps bare machines working
    OpenAI = None  # type: ignore[assignment]

_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)", re.S | re.I)
_FN_NAME_OK = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""


@dataclass
class Reply:
    content: str
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    elapsed_ms: int = 0
    error: str = ""


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    first_token_ms: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def safe_fn_name(name: str, taken: set[str]) -> str:
    """OpenAI function names allow [A-Za-z0-9_-]{1,64}; MCP names may have dots."""
    base = name if _FN_NAME_OK.match(name) else re.sub(r"[^A-Za-z0-9_-]", "__", name)[:60] or "tool"
    cand, i = base, 2
    while cand in taken:
        cand = f"{base[:58]}_{i}"
        i += 1
    taken.add(cand)
    return cand


class LLM:
    def __init__(self, base_url: str, api_key: str, agent_id: str, usage_scope: str | None = None) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.agent_id = agent_id
        self.usage = Usage()
        self.no_native_tools: set[str] = set()
        self.no_temperature: set[str] = set()
        self._scope = usage_scope
        self._client = self._make_client()

    def _make_client(self) -> Any:
        headers = build_proxy_headers(self.agent_id, self._scope)
        if OpenAI is None:
            return _MiniClient(self.base_url, self.api_key, headers)
        return OpenAI(base_url=self.base_url, api_key=self.api_key, default_headers=headers, max_retries=1)

    def set_scope(self, scope: str | None) -> None:
        if scope and scope != self._scope:
            self._scope = scope
            self._client = self._make_client()

    @property
    def headers(self) -> dict[str, str]:
        return build_proxy_headers(self.agent_id, self._scope)

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float | None = 0.0,
        timeout_s: float = 60.0,
        response_format: dict[str, Any] | None = None,
    ) -> Reply:
        timeout_s = max(3.0, timeout_s)
        kwargs: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens, "timeout": timeout_s}
        if temperature is not None and model not in self.no_temperature:
            kwargs["temperature"] = temperature
        if tools and model not in self.no_native_tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if response_format:
            kwargs["response_format"] = response_format
        started = time.monotonic()
        try:
            resp = await asyncio.wait_for(
                asyncio.to_thread(self._client.chat.completions.create, **kwargs),
                timeout=timeout_s + 2,
            )
        except asyncio.TimeoutError:
            return Reply("", error="timeout", model=model, elapsed_ms=int((time.monotonic() - started) * 1000))
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            low = msg.lower()
            if "tools" in kwargs and any(w in low for w in ("tool", "function")):
                self.no_native_tools.add(model)
                return await self.chat(model, messages, tools=None, max_tokens=max_tokens, temperature=temperature,
                                       timeout_s=max(3.0, timeout_s - (time.monotonic() - started)),
                                       response_format=response_format)
            if "temperature" in kwargs and "temperature" in low:
                self.no_temperature.add(model)
                return await self.chat(model, messages, tools=tools, max_tokens=max_tokens, temperature=None,
                                       timeout_s=max(3.0, timeout_s - (time.monotonic() - started)),
                                       response_format=response_format)
            if response_format and ("response_format" in low or "json" in low):
                return await self.chat(model, messages, tools=tools, max_tokens=max_tokens, temperature=temperature,
                                       timeout_s=max(3.0, timeout_s - (time.monotonic() - started)))
            return Reply("", error=msg[:500], model=model, elapsed_ms=int((time.monotonic() - started) * 1000))
        elapsed = int((time.monotonic() - started) * 1000)
        return self._to_reply(resp, model, elapsed)

    def _to_reply(self, resp: Any, model: str, elapsed_ms: int) -> Reply:
        choice = resp.choices[0] if getattr(resp, "choices", None) else None
        msg = getattr(choice, "message", None)
        content = str(getattr(msg, "content", "") or "")
        reasoning = str(getattr(msg, "reasoning_content", "") or getattr(msg, "reasoning", "") or "")
        think = _THINK_RE.findall(content)
        if think:
            reasoning = (reasoning + "\n" + "\n".join(think)).strip()
            content = _THINK_RE.sub("", content).strip()
        calls: list[ToolCall] = []
        for tc in getattr(msg, "tool_calls", None) or []:
            fn = getattr(tc, "function", None)
            raw_args = str(getattr(fn, "arguments", "") or "")
            calls.append(ToolCall(str(getattr(tc, "id", "") or f"call_{len(calls)}"), str(getattr(fn, "name", "")), parse_json_loose(raw_args) or {}, raw_args))
        usage = getattr(resp, "usage", None)
        pt = int(getattr(usage, "prompt_tokens", 0) or 0)
        ct = int(getattr(usage, "completion_tokens", 0) or 0)
        if not usage:
            ct = max(1, len((content + reasoning).split()) * 4 // 3)
        self.usage.prompt_tokens += pt
        self.usage.completion_tokens += ct
        self.usage.calls += 1
        if not self.usage.first_token_ms:
            self.usage.first_token_ms = elapsed_ms
        return Reply(
            content=content,
            reasoning=reasoning,
            tool_calls=calls,
            finish_reason=str(getattr(choice, "finish_reason", "") or ""),
            prompt_tokens=pt,
            completion_tokens=ct,
            model=str(getattr(resp, "model", "") or model),
            elapsed_ms=elapsed_ms,
        )


def parse_json_loose(text: str) -> Any:
    """Parse JSON from model output: fenced blocks, trailing prose, single quotes."""
    if not text:
        return None
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start < 0:
            continue
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    chunk = text[start : i + 1]
                    try:
                        return json.loads(chunk)
                    except json.JSONDecodeError:
                        try:
                            return json.loads(chunk.replace("'", '"'))
                        except json.JSONDecodeError:
                            break
    return None


# ---------------------------------------------------------------------------
# Minimal stdlib client with the same call shape as the OpenAI SDK.


class _NS:
    def __init__(self, data: Any) -> None:
        self._data = data

    def __getattr__(self, name: str) -> Any:
        if not isinstance(self._data, dict) or name not in self._data:
            return None
        return _wrap(self._data[name])


def _wrap(value: Any) -> Any:
    if isinstance(value, dict):
        return _NS(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


class _MiniCompletions:
    def __init__(self, base_url: str, api_key: str, headers: dict[str, str]) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", **headers}

    def create(self, **kwargs: Any) -> Any:
        from urllib.error import HTTPError
        from urllib.request import Request, urlopen

        timeout = float(kwargs.pop("timeout", 60) or 60)
        body = json.dumps(kwargs).encode("utf-8")
        req = Request(self.url, data=body, headers=self.headers, method="POST")
        try:
            with urlopen(req, timeout=timeout) as resp:
                return _wrap(json.loads(resp.read().decode("utf-8")))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


class _MiniChat:
    def __init__(self, completions: _MiniCompletions) -> None:
        self.completions = completions


class _MiniClient:
    def __init__(self, base_url: str, api_key: str, headers: dict[str, str]) -> None:
        self.chat = _MiniChat(_MiniCompletions(base_url, api_key, headers))
