"""Reason-act loop over discovered MCP tools plus local helper tools.

Works with native OpenAI tool calls when the model supports them, and with a
JSON action protocol when it does not. Every step is bounded by the puzzle
clock, and the loop always returns some answer before the deadline.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from . import compat, decoders, pngtools
from .answer import FormatSpec, extract_answer, is_invalid
from .clock import Clock
from .explorer import NEVER_AUTO_CALL, BlackBox, classify
from .llm import LLM, Reply, parse_json_loose, safe_fn_name

HIDE_KINDS = NEVER_AUTO_CALL | {"broadcast", "timer"}
TOOL_RESULT_CHARS = int(os.getenv("GAUNTLET_TOOL_RESULT_CHARS", "2500"))

LOCAL_TOOLS: list[dict[str, Any]] = [
    {
        "name": "decode",
        "description": (
            "Decode or decrypt text. With no op, tries layered encodings automatically "
            "(base64/32/85, hex, binary, decimal, morse, url, rot13, caesar, atbash, rot47, "
            "reverse, rail fence, single-byte xor, zlib/gzip) and returns ranked candidates. "
            "With op (vigenere, xor, caesar, rail_fence, or any of the above) applies one step; "
            "key is the cipher key, shift, or rail count."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "op": {"type": "string"},
                "key": {"type": "string"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "inspect_image",
        "description": (
            "Inspect an image for hidden data: format, size, PNG text chunks, bytes after the "
            "end marker, LSB payloads, printable strings. Pass a data URI or base64, or "
            "index=N for the Nth image already seen in tool results."
        ),
        "parameters": {
            "type": "object",
            "properties": {"data": {"type": "string"}, "index": {"type": "integer"}},
        },
    },
    {
        "name": "python",
        "description": (
            "Run a short Python 3 snippet (stdlib only, no network, 8 s limit) and get stdout. "
            "Use it for arithmetic, brute-force search over small spaces, logic grids, or parsing."
        ),
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
    },
    {
        "name": "read_resource",
        "description": "Read an MCP resource by URI (side-effect free).",
        "parameters": {"type": "object", "properties": {"uri": {"type": "string"}}, "required": ["uri"]},
    },
    {
        "name": "get_prompt",
        "description": "Fetch an MCP prompt template by name with optional string arguments.",
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}},
            "required": ["name"],
        },
    },
    {
        "name": "call_unlisted_tool",
        "description": (
            "Call a tool by exact name even though it is not in the tool list (hidden tools "
            "mentioned in clues, descriptions, or decoded text). Never use it to submit."
        ),
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}},
            "required": ["name"],
        },
    },
    {
        "name": "final_answer",
        "description": "Finish. Give the exact final answer string in the required output format.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number", "description": "0 to 1"},
            },
            "required": ["answer"],
        },
    },
]


@dataclass
class ReactResult:
    answer: str = ""
    confidence: float = 0.0
    steps: int = 0
    finished: bool = False
    candidates: list[str] = field(default_factory=list)
    log: list[str] = field(default_factory=list)


EventFn = Callable[[str], Awaitable[None]]


class ReactAgent:
    def __init__(
        self,
        box: BlackBox,
        llm: LLM,
        model: str,
        clock: Clock,
        *,
        spec: FormatSpec,
        on_event: EventFn | None = None,
        max_steps: int = 14,
        max_tokens: int = 4096,
        allow_python: bool | None = None,
        extra_hidden: set[str] | None = None,
    ) -> None:
        self.box = box
        self.llm = llm
        self.model = model
        self.clock = clock
        self.spec = spec
        self.on_event = on_event
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        if allow_python is None:
            allow_python = os.getenv("GAUNTLET_ALLOW_PYTHON", "1").lower() not in {"0", "false", "no"}
        self.allow_python = allow_python
        self.extra_hidden = extra_hidden or set()
        self._fn_to_tool: dict[str, str] = {}
        self._facts_seen = 0
        self._repeats: dict[str, list[Any]] = {}

    # -- tool specs ----------------------------------------------------------

    def _visible_tools(self) -> list[Any]:
        return [
            t for t in self.box.tools.values()
            if t.kind not in HIDE_KINDS and t.name not in self.extra_hidden and not t.destructive
        ]

    def _openai_tools(self) -> list[dict[str, Any]]:
        taken: set[str] = set()
        self._fn_to_tool = {}
        specs: list[dict[str, Any]] = []
        for local in LOCAL_TOOLS:
            if local["name"] == "python" and not self.allow_python:
                continue
            taken.add(local["name"])
            specs.append({"type": "function", "function": local})
        for tool in self._visible_tools():
            fn = safe_fn_name(tool.name, taken)
            self._fn_to_tool[fn] = tool.name
            params = dict(tool.schema) if tool.schema else {"type": "object", "properties": {}}
            params.setdefault("type", "object")
            props = dict(params.get("properties") or {})
            props.pop("agent_id", None)
            params["properties"] = props
            if isinstance(params.get("required"), list):
                params["required"] = [r for r in params["required"] if r != "agent_id"]
            desc = (tool.description or f"MCP tool {tool.name}")[:900]
            specs.append({"type": "function", "function": {"name": fn, "description": f"[MCP {tool.name}] {desc}", "parameters": params}})
        return specs

    def _json_protocol_text(self) -> str:
        lines = ["You act by replying with ONE JSON object and nothing else:",
                 '  {"tool": "<name>", "args": {...}}   to call a tool',
                 '  {"final_answer": "<exact answer>", "confidence": 0.0-1.0}   to finish',
                 "Available tools:"]
        for local in LOCAL_TOOLS:
            if local["name"] in {"final_answer"} or (local["name"] == "python" and not self.allow_python):
                continue
            props = ", ".join(local["parameters"].get("properties", {}))
            lines.append(f"- {local['name']}({props}): {local['description'][:200]}")
        for tool in self._visible_tools():
            lines.append("- " + tool.brief())
        return "\n".join(lines)

    # -- main loop -----------------------------------------------------------

    async def run(self, system: str, task: str) -> ReactResult:
        result = ReactResult()
        native = self.model not in self.llm.no_native_tools
        tools = self._openai_tools()
        protocol = "" if native else "\n\n" + self._json_protocol_text()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system + protocol},
            {"role": "user", "content": task},
        ]
        self._facts_seen = len(self.box.notebook.facts)
        while result.steps < self.max_steps:
            left = self.clock.left()
            if left < 6:
                result.log.append("clock: stopping loop")
                break
            result.steps += 1
            timeout = min(left - 3, max(8.0, left * 0.6))
            reply = await self.llm.chat(
                self.model, messages, tools=tools if native else None,
                max_tokens=self.max_tokens, temperature=0.0, timeout_s=timeout,
            )
            if native and self.model in self.llm.no_native_tools:
                # The model refused native tools and the client retried without them, so this
                # reply was written without the action protocol. Drop it and ask again.
                native = False
                messages[0]["content"] = system + "\n\n" + self._json_protocol_text()
                result.log.append("model rejected native tools; switching to the JSON action protocol")
                continue
            if reply.error:
                result.log.append(f"llm error: {reply.error[:200]}")
                if "timeout" in reply.error:
                    break
                await asyncio.sleep(0.5)
                continue
            if reply.tool_calls and native:
                done = await self._handle_native(reply, messages, result)
                if done:
                    return result
                continue
            action = parse_json_loose(reply.content) if reply.content else None
            if isinstance(action, dict) and ("tool" in action or "final_answer" in action):
                messages.append({"role": "assistant", "content": reply.content})
                if "final_answer" in action:
                    if self._finish(str(action.get("final_answer", "")), action.get("confidence"), result):
                        return result
                    messages.append({"role": "user", "content": self._format_nudge()})
                    continue
                name = str(action.get("tool", ""))
                args = action.get("args") if isinstance(action.get("args"), dict) else {}
                output = await self._dispatch(name, args, result)
                messages.append({"role": "user", "content": f"Result of {name}:\n{output}\n\nNext action (JSON only)."})
                continue
            # Plain text: maybe an ANSWER line.
            candidate = extract_answer(reply.content, self.spec)
            if candidate and not is_invalid(candidate) and re.search(r"(answer|final)\s*[:=]", reply.content, re.I):
                if self._finish(candidate, None, result):
                    return result
            messages.append({"role": "assistant", "content": reply.content or "(empty)"})
            messages.append({"role": "user", "content": "Continue. Call a tool, or call final_answer with the exact answer."
                             if native else "Reply with one JSON action object only."})
            self._trim(messages)
        # Out of steps or time: ask once for the best answer without tools.
        if not result.finished and self.clock.left() > 4:
            messages.append({"role": "user", "content": (
                "Time is nearly up. Reply with exactly one line: ANSWER: <your best final answer> "
                f"(format: {self.spec.describe()})."
            )})
            reply = await self.llm.chat(self.model, messages, tools=None, max_tokens=min(1024, self.max_tokens),
                                        temperature=0.0, timeout_s=max(3.0, self.clock.left() - 2))
            cand = extract_answer(reply.content, self.spec)
            if cand and not is_invalid(cand):
                result.answer = cand
                result.candidates.append(cand)
                result.confidence = 0.3
        return result

    async def _handle_native(self, reply: Reply, messages: list[dict[str, Any]], result: ReactResult) -> bool:
        messages.append({
            "role": "assistant",
            "content": reply.content or None,
            "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.raw_arguments or json.dumps(tc.arguments)}}
                for tc in reply.tool_calls
            ],
        })
        finished = False
        for tc in reply.tool_calls:
            if tc.name == "final_answer":
                ok = self._finish(str(tc.arguments.get("answer", "")), tc.arguments.get("confidence"), result)
                content = "accepted" if ok else self._format_nudge()
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
                finished = finished or ok
                continue
            output = await self._dispatch(self._fn_to_tool.get(tc.name, tc.name), tc.arguments, result)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})
        if not finished:
            self._trim(messages)
        return finished

    def _finish(self, answer: str, confidence: Any, result: ReactResult) -> bool:
        answer = (answer or "").strip()
        if is_invalid(answer):
            return False
        try:
            conf = float(confidence) if confidence is not None else 0.7
        except (TypeError, ValueError):
            conf = 0.7
        result.answer = answer
        result.confidence = max(0.0, min(1.0, conf))
        result.candidates.append(answer)
        result.finished = True
        return True

    def _format_nudge(self) -> str:
        return f"Not accepted. Give a concrete final answer in this format: {self.spec.describe()}."

    # -- dispatch ------------------------------------------------------------

    async def _dispatch(self, name: str, args: dict[str, Any], result: ReactResult) -> str:
        started = time.monotonic()
        sig = f"{name}:{json.dumps(args, sort_keys=True, default=str)[:500]}"
        seen = self._repeats.get(sig)
        if seen is not None:
            seen[0] += 1
            note = "failed" if seen[1].startswith("error") else "returned"
            if seen[0] >= 2:
                return (f"STOP: you already made this exact call {seen[0]} times and it {note} the same thing:\n"
                        f"{seen[1][:600]}\nChange the tool, the arguments, or decode what you have.")
            return f"(same call as before; cached result)\n{seen[1]}"
        try:
            out = await self._dispatch_inner(name, args)
        except Exception as exc:
            out = f"error: {type(exc).__name__}: {exc}"
        self._repeats[sig] = [0, out]
        result.log.append(f"{name}({compat.short(args, 160)}) -> {compat.short(out, 160)} [{int((time.monotonic()-started)*1000)}ms]")
        if self.on_event:
            try:
                await self.on_event(f"Tool {name}: {compat.short(out, 140)}")
            except Exception:
                pass
        delta = self._new_facts()
        if delta:
            out = f"{out}\n\n[new facts]\n{delta}"
        return out[: TOOL_RESULT_CHARS + 1500]

    async def _dispatch_inner(self, name: str, args: dict[str, Any]) -> str:
        if name == "decode":
            text = str(args.get("text", ""))
            op = str(args.get("op", "") or "")
            if op:
                return decoders.apply_named(text, op, str(args.get("key", "") or ""))
            cands = decoders.auto_decode(text, keys=self.box.notebook.keys + ([str(args["key"])] if args.get("key") else []))
            return decoders.to_json_safe(cands[:5])
        if name == "inspect_image":
            data = None
            if isinstance(args.get("index"), int) and self.box.notebook.images:
                idx = max(0, min(len(self.box.notebook.images) - 1, int(args["index"])))
                data = compat.decode_b64_loose(self.box.notebook.images[idx]["b64"])
            elif args.get("data"):
                data = compat.decode_b64_loose(str(args["data"]))
            if not data:
                return "error: no image bytes (pass data or index)"
            return json.dumps({k: v for k, v in pngtools.inspect_image(data).items() if k != "chunks"}, ensure_ascii=False)[:TOOL_RESULT_CHARS]
        if name == "python":
            if not self.allow_python:
                return "error: python tool disabled"
            return await asyncio.to_thread(run_python, str(args.get("code", "")), min(8.0, max(1.0, self.clock.left() - 4)))
        if name == "read_resource":
            res = await self.box.read_resource(str(args.get("uri", "")))
            if res.get("error"):
                return f"error: {res['error']}"
            return (res.get("text") or f"{len(res.get('blobs', []))} binary blob(s); use inspect_image index")[:TOOL_RESULT_CHARS]
        if name == "get_prompt":
            raw_args = args.get("arguments") if isinstance(args.get("arguments"), dict) else {}
            return (await self.box.get_prompt(str(args.get("name", "")), {k: str(v) for k, v in raw_args.items()}))[:TOOL_RESULT_CHARS]
        if name == "call_unlisted_tool":
            target = str(args.get("name", ""))
            if classify(target) in NEVER_AUTO_CALL:
                return "error: refusing to call a submit/destructive tool from the loop"
            inner = args.get("arguments") if isinstance(args.get("arguments"), dict) else {}
            return await self._call_mcp(target, inner)
        if name in self.box.tools:
            tool = self.box.tools[name]
            if tool.kind in NEVER_AUTO_CALL:
                return "error: this tool is reserved for the controller"
            return await self._call_mcp(name, args)
        return f"error: unknown tool {name!r}. Use call_unlisted_tool for hidden names."

    async def _call_mcp(self, name: str, args: dict[str, Any]) -> str:
        timeout = max(3.0, min(25.0, self.clock.left() - 3))
        parsed = await self.box.call(name, args, source="react", timeout_s=timeout)
        if parsed.get("error"):
            return f"error: {parsed['error']}"[:TOOL_RESULT_CHARS]
        body = parsed["dict"] if parsed["dict"] else parsed.get("text", "")
        text = compat.short(body, TOOL_RESULT_CHARS)
        if parsed.get("images"):
            text += f"\n[{len(parsed['images'])} image(s) returned; use inspect_image index={len(self.box.notebook.images) - 1}]"
        return text

    def _new_facts(self) -> str:
        facts = self.box.notebook.facts[self._facts_seen :]
        self._facts_seen = len(self.box.notebook.facts)
        keep = [f for f in facts if f.kind in {"decoded", "key", "flag", "discovery", "image", "notification"}]
        return "\n".join(f"- [{f.kind}] {f.text[:400]}" for f in keep[:8])

    def _trim(self, messages: list[dict[str, Any]], max_chars: int = 60000) -> None:
        total = sum(len(str(m.get("content") or "")) for m in messages)
        i = 2
        while total > max_chars and i < len(messages) - 4:
            m = messages[i]
            if m.get("role") in {"tool", "user"} and len(str(m.get("content") or "")) > 600:
                old = str(m["content"])
                m["content"] = old[:400] + "\n[...trimmed...]"
                total -= len(old) - len(m["content"])
            i += 1


_PY_DENY = re.compile(r"(/proc|environ|os\.system|subprocess|socket|urllib|requests|http\.client|shutil\.rmtree|ctypes|__import__\(['\"]os)")


def run_python(code: str, timeout_s: float = 8.0) -> str:
    """Run model-written code in an isolated interpreter with a scrubbed environment."""
    if not code.strip():
        return "error: empty code"
    if _PY_DENY.search(code):
        return "error: code touches the network, processes, or environment; not allowed"
    with tempfile.TemporaryDirectory(prefix="gauntlet-py-") as tmp:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=tmp,
                env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
            )
        except subprocess.TimeoutExpired:
            return f"error: timed out after {timeout_s:.0f}s"
        out = (proc.stdout or "")[-3000:]
        err = (proc.stderr or "")[-1200:]
        if proc.returncode != 0:
            return f"exit {proc.returncode}\nstdout:\n{out}\nstderr:\n{err}"
        return out or "(no output; print() your result)"
