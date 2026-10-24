"""Black-box MCP exploration.

The agent does not know the server's tools in advance. This module:

1. Connects with a client that also answers server-initiated requests
   (sampling, elicitation) and records every notification and log line.
2. Builds an inventory: tools (with schemas), resources, resource templates,
   prompts, and the server's own instructions.
3. Classifies each tool (challenge, clue, timer, submit, checker, image,
   search, danger, unknown) so the agent never probes a submit or reset tool.
4. Synthesizes arguments from JSON Schema and from what it has learned.
5. Mines every result for hidden tool names, ids, URIs, keys, flags, and
   encoded payloads, then follows those leads under a time budget.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from arena_clients.mcp_client import McpArenaClient
from mcp import ClientSession

from . import compat, decoders, pngtools

# ---------------------------------------------------------------------------
# Notebook: everything the agent has learned, in arrival order.


@dataclass
class Fact:
    source: str
    text: str
    kind: str = "note"
    ts: float = field(default_factory=time.time)


class Notebook:
    def __init__(self) -> None:
        self.facts: list[Fact] = []
        self._seen: set[str] = set()
        self.keys: list[str] = []
        self.flags: list[str] = []
        self.ids: dict[str, list[str]] = {}
        self.uris: list[str] = []
        self.hidden_tool_candidates: list[str] = []
        self.images: list[dict[str, str]] = []

    def add(self, source: str, text: str, kind: str = "note") -> bool:
        text = (text or "").strip()
        if not text:
            return False
        sig = f"{kind}:{text[:400]}"
        if sig in self._seen:
            return False
        self._seen.add(sig)
        self.facts.append(Fact(source, text, kind))
        return True

    def add_key(self, key: str) -> None:
        key = key.strip().strip("'\"`.,;")
        if key and key not in self.keys and len(key) <= 80:
            self.keys.append(key)

    def add_flag(self, flag: str) -> None:
        if flag not in self.flags:
            self.flags.append(flag)
            self.add("flag", flag, "flag")

    def render(self, char_budget: int = 6000, kinds: set[str] | None = None) -> str:
        lines: list[str] = []
        used = 0
        for fact in self.facts:
            if kinds and fact.kind not in kinds:
                continue
            line = f"[{fact.kind}:{fact.source}] {fact.text}"
            if used + len(line) > char_budget:
                lines.append("[...older facts trimmed...]")
                break
            lines.append(line)
            used += len(line) + 1
        if self.keys:
            lines.append("Known keys/passwords: " + ", ".join(self.keys[:12]))
        if self.flags:
            lines.append("Flags seen: " + ", ".join(self.flags[:12]))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool classification

_TOKEN_SPLIT = re.compile(r"[._\-\s/:]+|(?<=[a-z0-9])(?=[A-Z])")

DANGER = {"delete", "remove", "reset", "destroy", "drop", "wipe", "register", "unregister",
          "disconnect", "kick", "leaderboard", "shutdown", "logout", "revoke", "purge", "restart"}
SUBMIT = {"submit", "submission", "turnin", "finalize", "finalise"}
CHECKER = {"check", "verify", "validate", "guess", "grade", "evaluate", "score"}
BROADCAST = {"broadcast", "thought", "thoughts", "heartbeat", "announce"}
CHALLENGE = {"challenge", "task", "mission", "puzzle", "objective", "briefing", "quest"}
CLUE = {"clue", "clues", "hint", "hints"}
TIMER = {"time", "timer", "deadline", "clock", "remaining", "countdown"}
IMAGE = {"image", "img", "picture", "photo"}
SEARCH = {"search", "lookup", "retrieve", "fetch", "browse", "query", "wiki", "web", "find", "scrape"}
DECRYPT = {"decrypt", "decode", "decipher", "unlock", "unseal", "crack", "cipher", "vault",
           "key", "keys", "secret", "secrets", "safe", "lock", "door", "gate"}
EXPLORE = {"list", "describe", "inspect", "help", "info", "discover", "catalog", "manifest",
           "schema", "explore", "map", "dir", "ls", "read", "open", "get", "status", "whoami",
           "about", "peek", "scan", "look", "view", "show"}


def tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_SPLIT.split(text or "") if t} | {t.lower() for t in _TOKEN_SPLIT.split(text or "") if t}


def _kind_from_tokens(tok: set[str]) -> str:
    tok = {t.lower() for t in tok}
    if tok & DANGER:
        return "danger"
    if tok & SUBMIT or {"final", "answer"} <= tok:
        return "submit"
    if tok & CHECKER:
        return "checker"
    if tok & BROADCAST:
        return "broadcast"
    if tok & CHALLENGE:
        return "challenge"
    if tok & CLUE:
        return "clue_list" if "list" in tok else "clue_get"
    if tok & TIMER:
        return "timer"
    if tok & IMAGE or tok & {"vision", "ocr", "draw", "paint", "inpaint", "caption"}:
        if tok & {"edit", "inpaint", "modify", "transform"}:
            return "image_edit"
        if tok & {"generate", "gen", "create", "draw", "paint", "render"}:
            return "image_generate"
        return "image_analyze"
    if tok & SEARCH:
        return "search"
    if tok & DECRYPT:
        return "decrypt"
    if tok & EXPLORE:
        return "explore"
    return "unknown"


NEVER_AUTO_CALL = {"danger", "submit"}
RELIST_AFTER = {"decrypt", "explore", "unknown", "checker", "search"}


@dataclass
class ToolInfo:
    name: str
    description: str
    schema: dict[str, Any]
    kind: str
    annotations: dict[str, Any] = field(default_factory=dict)
    listed: bool = True

    @property
    def properties(self) -> dict[str, Any]:
        props = self.schema.get("properties")
        return props if isinstance(props, dict) else {}

    @property
    def required(self) -> list[str]:
        req = self.schema.get("required")
        return [str(r) for r in req] if isinstance(req, list) else []

    @property
    def destructive(self) -> bool:
        return bool(self.annotations.get("destructiveHint") or self.annotations.get("destructive_hint"))

    def brief(self) -> str:
        args = []
        for pname, spec in self.properties.items():
            if pname == "agent_id":
                continue
            ptype = spec.get("type", "any") if isinstance(spec, dict) else "any"
            mark = "*" if pname in self.required else ""
            args.append(f"{pname}{mark}:{ptype}")
        desc = self.description.replace("\n", " ")
        if len(desc) > 220:
            desc = desc[:217] + "..."
        return f"{self.name}({', '.join(args)}) [{self.kind}] {desc}"


def classify(name: str, description: str = "", annotations: dict | None = None) -> str:
    """Kind of a tool from its name; the description can only make it more careful."""
    annotations = annotations or {}
    if annotations.get("destructiveHint") or annotations.get("destructive_hint"):
        return "danger"
    kind = _kind_from_tokens(tokens(name))
    desc_kind = _kind_from_tokens(tokens(description[:240])) if description else "unknown"
    if desc_kind in NEVER_AUTO_CALL and kind not in NEVER_AUTO_CALL | {"broadcast"}:
        # "Submits your final answer" in the description beats a bland name.
        if re.search(r"\b(submit|final answer|irreversible|cannot be undone|resets?)\b", description[:240], re.I):
            return desc_kind
    if kind == "unknown" and desc_kind not in {"explore"}:
        return desc_kind
    return kind


# ---------------------------------------------------------------------------
# Argument synthesis


@dataclass
class ArgContext:
    agent_id: str
    challenge_text: str = ""
    image_uri: str = ""
    known_ids: dict[str, list[str]] = field(default_factory=dict)
    keys: list[str] = field(default_factory=list)
    overrides: dict[str, Any] = field(default_factory=dict)


_TEXTY = re.compile(r"(query|q|text|prompt|question|input|message|content|instruction|search|term|keyword|topic)", re.I)
_URIY = re.compile(r"(uri|url|link|image|src|path|file)", re.I)
_KEYY = re.compile(r"(key|password|passphrase|secret|token|code|pin|cipher|seed)", re.I)
_IDY = re.compile(r"(^id$|_id$|^name$|identifier|slug)", re.I)


def synthesize_args(tool: ToolInfo, ctx: ArgContext) -> dict[str, Any] | None:
    """Fill a tool's arguments from its schema. None when a required arg is unknowable."""
    args: dict[str, Any] = {}
    props = tool.properties
    for pname, spec in props.items():
        spec = spec if isinstance(spec, dict) else {}
        required = pname in tool.required
        value = _value_for(pname, spec, ctx, required)
        if value is _MISSING:
            if required:
                return None
            continue
        args[pname] = value
    if "agent_id" not in args and tool.name.startswith("arena."):
        # Arena tools take agent_id even when a schema forgets to say so.
        args["agent_id"] = ctx.agent_id
    return args


class _Missing:
    pass


_MISSING = _Missing()


def _value_for(pname: str, spec: dict, ctx: ArgContext, required: bool) -> Any:
    if pname in ctx.overrides:
        return ctx.overrides[pname]
    if pname == "agent_id":
        return ctx.agent_id
    if "const" in spec:
        return spec["const"]
    if isinstance(spec.get("enum"), list) and spec["enum"]:
        return spec["enum"][0]
    if "default" in spec:
        return spec["default"]
    ptype = spec.get("type")
    if isinstance(ptype, list):
        ptype = next((t for t in ptype if t != "null"), "string")
    if not ptype and isinstance(spec.get("anyOf"), list):
        types = [s.get("type") for s in spec["anyOf"] if isinstance(s, dict)]
        ptype = next((t for t in types if t and t != "null"), "string")
    lname = pname.lower()
    for id_kind, values in ctx.known_ids.items():
        if values and (lname == id_kind or lname.rstrip("s") == id_kind.rstrip("s")):
            return values[0]
    if ptype in (None, "string"):
        if _KEYY.search(lname) and ctx.keys:
            return ctx.keys[0]
        if _URIY.search(lname) and ctx.image_uri:
            return ctx.image_uri
        if _TEXTY.search(lname) and ctx.challenge_text:
            return ctx.challenge_text[:500]
        if not required:
            return _MISSING
        if _IDY.search(lname):
            return _MISSING
        return ctx.challenge_text[:200] if ctx.challenge_text else _MISSING
    if not required:
        return _MISSING
    if ptype == "integer":
        lo = spec.get("minimum", 0)
        return int(lo) if isinstance(lo, (int, float)) else 0
    if ptype == "number":
        lo = spec.get("minimum", 0)
        return float(lo) if isinstance(lo, (int, float)) else 0.0
    if ptype == "boolean":
        return False
    if ptype == "array":
        return []
    if ptype == "object":
        return {}
    return _MISSING


# ---------------------------------------------------------------------------
# Hint mining

_TOOLISH = re.compile(r"`([A-Za-z_][\w.\-]{2,63})`|\b([a-z][a-z0-9_]*(?:[._][a-z0-9_]+){1,4})\b")
_KEY_PHRASE = re.compile(
    r"(?:key|password|passphrase|pass ?code|secret|code ?word|pin|token)\s*(?:is|=|:|was)\s*[\"'`]?([A-Za-z0-9_\-{}!@#$%^&*.]{3,64})",
    re.I,
)
_URI = re.compile(r"\b([a-z][a-z0-9+.\-]{1,20}://[^\s\"'<>)\]]{2,200})")
_CLUE_ID = re.compile(r"\b((?:clue|hint|step|level|stage|door|room|node)[_\-]?\d+)\b", re.I)
_UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
_CALL_HINT = re.compile(r"\b(?:call|use|invoke|run|try|unlock(?:ed)?|enable[sd]?|reveal(?:s|ed)?)\s+(?:the\s+)?(?:tool\s+)?[`'\"]?([A-Za-z_][\w.\-]{2,63})", re.I)

_COMMON_DOTTED = {"e.g", "i.e", "etc", "vs", "a.m", "p.m"}


@dataclass
class Hints:
    tool_names: list[str] = field(default_factory=list)
    keys: list[str] = field(default_factory=list)
    uris: list[str] = field(default_factory=list)
    ids: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    decoded: list[dict] = field(default_factory=list)


def mine_text(text: str, known_tools: set[str], keys: list[str] | None = None) -> Hints:
    hints = Hints()
    if not text:
        return hints
    json_keys = set(re.findall(r'"([A-Za-z_][\w.\-]*)"\s*:', text))
    strong: list[str] = []
    for m in _CALL_HINT.finditer(text):
        name = m.group(1).strip(".,;:")
        if ("_" in name or "." in name) and name not in known_tools:
            strong.append(name)
    for m in re.finditer(r"`([A-Za-z_][\w.\-]{2,63})`", text):
        if m.group(1) not in known_tools:
            strong.append(m.group(1))
    weak: list[str] = []
    for m in _TOOLISH.finditer(text):
        name = m.group(2)
        if not name or "." not in name:
            continue  # bare snake_case words are field names far more often than tools
        if name.lower() in _COMMON_DOTTED or name in known_tools or name in json_keys:
            continue
        if re.fullmatch(r"[\d._]+", name) or name.endswith((".png", ".jpg", ".com", ".json", ".py", ".txt", ".org", ".io", ".net")):
            continue
        weak.append(name)
    for name in strong + weak:
        if name not in hints.tool_names and name not in json_keys:
            hints.tool_names.append(name)
    hints.keys = [k for k in dict.fromkeys(m.group(1).strip(".,;") for m in _KEY_PHRASE.finditer(text))]
    hints.uris = list(dict.fromkeys(_URI.findall(text)))
    hints.ids = list(dict.fromkeys(m.group(1) for m in _CLUE_ID.finditer(text))) + _UUID.findall(text)
    hints.flags = decoders.find_flags(text)
    hints.decoded = decoders.decode_findings(text, keys=keys or [])
    for item in hints.decoded:
        inner = mine_text(item["text"], known_tools) if item.get("text") and item["text"] != text else Hints()
        hints.flags += [f for f in item.get("flags", []) if f not in hints.flags] + inner.flags
        hints.keys += [k for k in inner.keys if k not in hints.keys]
        hints.tool_names += [t for t in inner.tool_names if t not in hints.tool_names]
        hints.uris += [u for u in inner.uris if u not in hints.uris]
    return hints


# ---------------------------------------------------------------------------
# Client with server-initiated request support


SamplingResponder = Callable[[str, str, int], Awaitable[str]]


class ExplorerArenaClient(McpArenaClient):
    """The kit's MCP client, plus notifications, sampling and elicitation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.init_result: Any = None
        self.notifications: list[str] = []
        self.tools_changed = asyncio.Event()
        self.sampling_responder: SamplingResponder | None = None
        self.elicitation_values: dict[str, Any] = {}

    async def __aenter__(self) -> "ExplorerArenaClient":
        from mcp.client.sse import sse_client

        self._context = sse_client(self.sse_url, headers=self._headers)
        read, write = await self._context.__aenter__()
        kwargs: dict[str, Any] = {}
        params = inspect.signature(ClientSession.__init__).parameters
        if "message_handler" in params:
            kwargs["message_handler"] = self._on_message
        if "logging_callback" in params:
            kwargs["logging_callback"] = self._on_log
        if "sampling_callback" in params:
            kwargs["sampling_callback"] = self._on_sampling
        if "elicitation_callback" in params:
            kwargs["elicitation_callback"] = self._on_elicitation
        self._session = ClientSession(read, write, **kwargs)
        await self._session.__aenter__()
        self.init_result = await self._session.initialize()
        return self

    @property
    def session(self) -> ClientSession:
        if not self._session:
            raise RuntimeError("not connected")
        return self._session

    async def _on_message(self, message: Any) -> None:
        try:
            root = getattr(message, "root", message)
            kind = type(root).__name__
            method = str(getattr(root, "method", "") or "")
            if "ToolListChanged" in kind or method.endswith("tools/list_changed"):
                self.tools_changed.set()
            if isinstance(message, Exception):
                return
            if method and "progress" not in method and not method.endswith("notifications/message"):
                params = getattr(root, "params", None)
                detail = compat.short(params.model_dump() if hasattr(params, "model_dump") else params, 400)
                self.notifications.append(f"{method}: {detail}")
        except Exception:
            return

    async def _on_log(self, params: Any) -> None:
        try:
            data = compat.attr(params, "data", default="")
            level = compat.attr(params, "level", default="info")
            self.notifications.append(f"log[{level}]: {compat.short(data, 500)}")
        except Exception:
            return

    async def _on_sampling(self, context: Any, params: Any) -> Any:
        import mcp.types as T

        messages = compat.attr(params, "messages", default=[]) or []
        system = str(compat.attr(params, "systemPrompt", "system_prompt", default="") or "")
        max_tokens = int(compat.attr(params, "maxTokens", "max_tokens", default=512) or 512)
        prompt_parts = []
        for msg in messages:
            content = compat.attr(msg, "content", default=None)
            role = compat.attr(msg, "role", default="user")
            text = compat.attr(content, "text", default=None) if content is not None else None
            if text:
                prompt_parts.append(f"{role}: {text}")
        prompt = "\n".join(prompt_parts)
        self.notifications.append(f"sampling request: {prompt[:300]}")
        reply = ""
        if self.sampling_responder is not None:
            try:
                reply = await self.sampling_responder(system, prompt, max_tokens)
            except Exception as exc:
                reply = f"(sampling failed: {type(exc).__name__})"
        return T.CreateMessageResult(
            role="assistant",
            content=T.TextContent(type="text", text=reply or "OK"),
            model="arena-proxy",
            stopReason="endTurn",
        )

    async def _on_elicitation(self, context: Any, params: Any) -> Any:
        import mcp.types as T

        message = str(compat.attr(params, "message", default="") or "")
        schema = compat.attr(params, "requestedSchema", "requested_schema", default={}) or {}
        if hasattr(schema, "model_dump"):
            schema = schema.model_dump()
        self.notifications.append(f"elicitation: {message[:300]}")
        props = schema.get("properties", {}) if isinstance(schema, dict) else {}
        content: dict[str, Any] = {}
        for pname, spec in props.items():
            if pname in self.elicitation_values:
                content[pname] = self.elicitation_values[pname]
                continue
            spec = spec if isinstance(spec, dict) else {}
            ptype = spec.get("type", "string")
            if isinstance(spec.get("enum"), list) and spec["enum"]:
                content[pname] = spec["enum"][0]
            elif ptype == "boolean":
                content[pname] = True
            elif ptype in ("integer", "number"):
                content[pname] = spec.get("minimum", 1)
            else:
                content[pname] = self.elicitation_values.get("_default", "yes")
        return T.ElicitResult(action="accept", content=content)


# ---------------------------------------------------------------------------
# The explorer


@dataclass
class CallRecord:
    name: str
    args: dict[str, Any]
    ok: bool
    elapsed_ms: int
    summary: str


class BlackBox:
    """Inventory, safe calls, and lead-following over one MCP connection."""

    def __init__(self, client: ExplorerArenaClient, agent_id: str, *, call_timeout_s: float = 25.0) -> None:
        self.client = client
        self.agent_id = agent_id
        self.call_timeout_s = call_timeout_s
        self.tools: dict[str, ToolInfo] = {}
        self.resources: list[dict[str, str]] = []
        self.resource_templates: list[dict[str, str]] = []
        self.prompts: list[dict[str, Any]] = []
        self.server_name = ""
        self.instructions = ""
        self.notebook = Notebook()
        self.calls: list[CallRecord] = []
        self.tools_used: list[str] = []
        self.ctx = ArgContext(agent_id=agent_id)
        self._probed: set[str] = set()
        self._notif_seen = 0

    # -- inventory ---------------------------------------------------------

    async def inventory(self) -> None:
        init = self.client.init_result
        if init is not None:
            info = compat.attr(init, "serverInfo", "server_info", default=None)
            self.server_name = str(compat.attr(info, "name", default="") or "")
            self.instructions = str(compat.attr(init, "instructions", default="") or "")
            if self.instructions:
                self.notebook.add("server", self.instructions, "instructions")
                self._absorb(self.instructions, "server-instructions")
        await self.refresh_tools()
        await asyncio.gather(self._list_resources(), self._list_prompts(), return_exceptions=True)

    async def refresh_tools(self) -> list[str]:
        """Re-list tools. Returns names that are new since the last listing."""
        before = set(self.tools)
        tools: list[Any] = []
        cursor = None
        for _ in range(20):
            result = await self._paged(self.client.session.list_tools, cursor)
            tools.extend(compat.attr(result, "tools", default=[]) or [])
            cursor = compat.attr(result, "nextCursor", "next_cursor", default=None)
            if not cursor:
                break
        for tool in tools:
            name = compat.tool_name(tool)
            if not name:
                continue
            ann = compat.tool_annotations(tool)
            desc = compat.tool_description(tool)
            self.tools[name] = ToolInfo(name, desc, compat.tool_schema(tool), classify(name, desc, ann), ann)
        self.client.tools_changed.clear()
        new = [n for n in self.tools if n not in before]
        for name in new:
            if before:
                self.notebook.add("discovery", f"New tool appeared: {self.tools[name].brief()}", "discovery")
            self._absorb(self.tools[name].description, f"tool:{name}")
        return new

    async def _paged(self, fn: Callable[..., Awaitable[Any]], cursor: str | None) -> Any:
        if not cursor:
            return await asyncio.wait_for(fn(), timeout=self.call_timeout_s)
        try:
            import mcp.types as T

            return await asyncio.wait_for(fn(params=T.PaginatedRequestParams(cursor=cursor)), timeout=self.call_timeout_s)
        except (TypeError, AttributeError):
            return await asyncio.wait_for(fn(cursor=cursor), timeout=self.call_timeout_s)

    async def _list_resources(self) -> None:
        session = self.client.session
        try:
            res = await asyncio.wait_for(session.list_resources(), timeout=8)
            for r in compat.attr(res, "resources", default=[]) or []:
                item = {
                    "uri": str(compat.attr(r, "uri", default="")),
                    "name": str(compat.attr(r, "name", default="") or ""),
                    "description": str(compat.attr(r, "description", default="") or ""),
                    "mime": str(compat.attr(r, "mimeType", "mime_type", default="") or ""),
                }
                self.resources.append(item)
                self._absorb(item["description"], f"resource:{item['uri']}")
        except Exception:
            pass
        try:
            res = await asyncio.wait_for(session.list_resource_templates(), timeout=8)
            for r in compat.attr(res, "resourceTemplates", "resource_templates", default=[]) or []:
                self.resource_templates.append(
                    {
                        "uriTemplate": str(compat.attr(r, "uriTemplate", "uri_template", default="")),
                        "name": str(compat.attr(r, "name", default="") or ""),
                        "description": str(compat.attr(r, "description", default="") or ""),
                    }
                )
        except Exception:
            pass

    async def _list_prompts(self) -> None:
        try:
            res = await asyncio.wait_for(self.client.session.list_prompts(), timeout=8)
            for p in compat.attr(res, "prompts", default=[]) or []:
                args = compat.attr(p, "arguments", default=[]) or []
                self.prompts.append(
                    {
                        "name": str(compat.attr(p, "name", default="")),
                        "description": str(compat.attr(p, "description", default="") or ""),
                        "arguments": [
                            {
                                "name": str(compat.attr(a, "name", default="")),
                                "required": bool(compat.attr(a, "required", default=False)),
                            }
                            for a in args
                        ],
                    }
                )
        except Exception:
            pass

    def inventory_text(self, include_core: bool = True) -> str:
        lines = []
        if self.server_name:
            lines.append(f"Server: {self.server_name}")
        if self.instructions:
            lines.append(f"Server instructions: {self.instructions[:800]}")
        lines.append("Tools:")
        for tool in self.tools.values():
            if not include_core and tool.kind in {"timer", "broadcast"}:
                continue
            lines.append("  - " + tool.brief())
        if self.resources:
            lines.append("Resources:")
            lines += [f"  - {r['uri']} {r['name']} {r['description'][:120]}" for r in self.resources[:30]]
        if self.resource_templates:
            lines.append("Resource templates:")
            lines += [f"  - {r['uriTemplate']} {r['description'][:120]}" for r in self.resource_templates[:20]]
        if self.prompts:
            lines.append("Prompts:")
            lines += [
                f"  - {p['name']}({', '.join(a['name'] for a in p['arguments'])}) {p['description'][:120]}"
                for p in self.prompts[:20]
            ]
        return "\n".join(lines)

    def by_kind(self, *kinds: str) -> list[ToolInfo]:
        return [t for t in self.tools.values() if t.kind in kinds]

    def find(self, *patterns: str) -> ToolInfo | None:
        """First tool whose exact name matches, else first regex match."""
        for pat in patterns:
            if pat in self.tools:
                return self.tools[pat]
        for pat in patterns:
            rx = re.compile(pat, re.I)
            for tool in self.tools.values():
                if rx.search(tool.name):
                    return tool
        return None

    # -- calls -------------------------------------------------------------

    async def call(
        self,
        name: str,
        args: dict[str, Any] | None = None,
        *,
        source: str = "agent",
        mine: bool = True,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Call a tool. Returns compat.parse_tool_result(...) plus "dict" and "error"."""
        args = dict(args or {})
        tool = self.tools.get(name)
        if "agent_id" not in args and (tool is None or "agent_id" in tool.properties or not tool.listed):
            args["agent_id"] = self.agent_id
        parsed = await self._raw_call(name, args, timeout_s)
        err_text = str(parsed.get("exception") or (parsed.get("text") if parsed.get("is_error") else "") or "").lower()
        validation = re.search(r"(validation error|unexpected keyword|extra inputs|extra_forbidden|missing \d+ required|field required|additional properties|invalid params)", err_text)
        if err_text and "agent_id" in err_text and validation:
            self.calls.append(CallRecord(name, _redact(args), False, int(parsed.get("elapsed_ms", 0)),
                                         f"ERROR (retried) {err_text[:280]}"))
            if re.search(r"(unexpected|extra|not permitted|additional)", err_text) and "agent_id" in args:
                args.pop("agent_id")
                parsed = await self._raw_call(name, args, timeout_s)
            elif re.search(r"(missing|required)", err_text) and "agent_id" not in args:
                args["agent_id"] = self.agent_id
                parsed = await self._raw_call(name, args, timeout_s)
        return await self._finish_call(name, args, parsed, mine)

    async def _raw_call(self, name: str, args: dict[str, Any], timeout_s: float | None) -> dict[str, Any]:
        started = time.monotonic()
        try:
            raw = await asyncio.wait_for(
                self.client.session.call_tool(name, args),
                timeout=timeout_s or self.call_timeout_s,
            )
            parsed = compat.parse_tool_result(raw)
        except asyncio.TimeoutError:
            parsed = {"data": None, "text": "", "images": [], "resources": [], "is_error": True, "exception": "timeout"}
        except Exception as exc:  # protocol errors, unknown tool, validation
            parsed = {"data": None, "text": "", "images": [], "resources": [], "is_error": True, "exception": f"{type(exc).__name__}: {exc}"}
        parsed["elapsed_ms"] = int((time.monotonic() - started) * 1000)
        return parsed

    async def _finish_call(self, name: str, args: dict[str, Any], parsed: dict[str, Any], mine: bool) -> dict[str, Any]:
        elapsed = int(parsed.get("elapsed_ms", 0))
        as_dict = compat.result_as_dict(parsed)
        error = parsed.get("exception") or (as_dict.get("error") if isinstance(as_dict.get("error"), str) else None)
        if parsed.get("is_error") and not error:
            error = parsed.get("text") or "tool reported an error"
        parsed["dict"] = as_dict
        parsed["error"] = error
        ok = not error
        if ok and name not in self.tools_used:
            self.tools_used.append(name)
        summary = compat.short(as_dict if as_dict else parsed.get("text", ""), 300)
        self.calls.append(CallRecord(name, _redact(args), ok, elapsed, summary if ok else f"ERROR {error}"[:300]))
        if mine:
            self._absorb_result(name, parsed)
        await self._drain_notifications()
        tool = self.tools.get(name)
        relist = self.client.tools_changed.is_set() or (
            ok and (tool is None or not tool.listed or tool.kind in RELIST_AFTER)
        )
        if relist:
            # Some SDK versions swallow tools/list_changed, so a call that may have
            # unlocked something is followed by a cheap re-list.
            try:
                await self.refresh_tools()
            except Exception:
                pass
        return parsed

    async def read_resource(self, uri: str) -> dict[str, Any]:
        try:
            res = await asyncio.wait_for(self.client.session.read_resource(uri), timeout=self.call_timeout_s)
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}", "text": ""}
        texts, blobs = [], []
        for item in compat.attr(res, "contents", default=[]) or []:
            text = compat.attr(item, "text", default=None)
            blob = compat.attr(item, "blob", default=None)
            if text:
                texts.append(str(text))
            if blob:
                blobs.append(str(blob))
        out = {"text": "\n".join(texts), "blobs": blobs, "error": None}
        self.notebook.add(f"resource:{uri}", out["text"][:1500] or f"{len(blobs)} binary blob(s)", "resource")
        self._absorb(out["text"], f"resource:{uri}")
        for blob in blobs[:3]:
            self._absorb_image_b64(blob, f"resource:{uri}")
        return out

    async def get_prompt(self, name: str, arguments: dict[str, str] | None = None) -> str:
        try:
            res = await asyncio.wait_for(self.client.session.get_prompt(name, arguments or {}), timeout=self.call_timeout_s)
        except Exception as exc:
            return f"error: {type(exc).__name__}: {exc}"
        parts = []
        for msg in compat.attr(res, "messages", default=[]) or []:
            content = compat.attr(msg, "content", default=None)
            text = compat.attr(content, "text", default=None)
            if text:
                parts.append(str(text))
        text = "\n".join(parts)
        self.notebook.add(f"prompt:{name}", text[:1500], "prompt")
        self._absorb(text, f"prompt:{name}")
        return text

    # -- mining ------------------------------------------------------------

    def _absorb_result(self, name: str, parsed: dict[str, Any]) -> None:
        text_parts = []
        if parsed.get("text"):
            text_parts.append(parsed["text"])
        data = parsed.get("data")
        if data is not None and not parsed.get("text"):
            text_parts.append(json.dumps(data, ensure_ascii=False, default=str))
        blob = "\n".join(text_parts)
        if blob:
            self._absorb(blob, f"tool:{name}")
        for img in parsed.get("images", [])[:3]:
            self._absorb_image_b64(img.get("b64", ""), f"tool:{name}", img.get("mime", ""))
        for res in parsed.get("resources", [])[:5]:
            if res.get("text"):
                self._absorb(str(res["text"]), f"embedded:{res.get('uri')}")
            if res.get("b64"):
                self._absorb_image_b64(str(res["b64"]), f"embedded:{res.get('uri')}", res.get("mime", ""))
            if res.get("uri") and res["uri"] not in self.notebook.uris:
                self.notebook.uris.append(res["uri"])

    def _absorb(self, text: str, source: str) -> None:
        if not text:
            return
        hints = mine_text(text, set(self.tools), self.notebook.keys)
        for key in hints.keys:
            self.notebook.add_key(key)
            self.notebook.add(source, f"possible key: {key}", "key")
        for flag in hints.flags:
            self.notebook.add_flag(flag)
        for uri in hints.uris:
            if uri not in self.notebook.uris:
                self.notebook.uris.append(uri)
        for ident in hints.ids:
            bucket = re.sub(r"[_\-]?\d+$", "", ident.lower()) + "_id"
            self.notebook.ids.setdefault(bucket, [])
            if ident not in self.notebook.ids[bucket]:
                self.notebook.ids[bucket].append(ident)
        for cand in hints.tool_names:
            if cand not in self.tools and cand not in self.notebook.hidden_tool_candidates:
                self.notebook.hidden_tool_candidates.append(cand)
        for item in hints.decoded:
            self.notebook.add(source, f"decoded ({item['chain']}): {item['text'][:600]}", "decoded")
        # Also look for data URIs / long base64 that may be images.
        for m in re.finditer(r"data:image/[a-z+]+;base64,[A-Za-z0-9+/=]{100,}", text):
            self._absorb_image_b64(m.group(0), source)
        self.ctx.keys = self.notebook.keys
        self.ctx.known_ids = {k: v for k, v in self.notebook.ids.items()}

    def _absorb_image_b64(self, b64: str, source: str, mime: str = "") -> None:
        data = compat.decode_b64_loose(b64)
        if not data or len(data) < 24:
            return
        if pngtools.sniff(data) == "unknown":
            return
        report = pngtools.inspect_image(data)
        self.notebook.images.append({"source": source, "mime": mime, "b64": b64 if len(b64) < 3_000_000 else ""})
        summary = {k: v for k, v in report.items() if k in {"format", "width", "height", "text_chunks", "trailing_strings", "lsb", "flags"}}
        self.notebook.add(source, f"image inspected: {compat.short(summary, 700)}", "image")
        for finding in report.get("findings", []):
            self._absorb(finding, f"{source}#image")

    async def _drain_notifications(self) -> None:
        notes = self.client.notifications
        while self._notif_seen < len(notes):
            note = notes[self._notif_seen]
            self._notif_seen += 1
            self.notebook.add("server-notification", note, "notification")
            self._absorb(note, "notification")

    # -- lead following ----------------------------------------------------

    async def follow_leads(self, deadline: float, *, max_calls: int = 12) -> int:
        """Probe safe unknown tools, hidden tool names, resources and prompts.

        Stops at ``deadline`` (time.monotonic()) or after ``max_calls``.
        Returns the number of calls made.
        """
        calls = 0

        def time_left() -> float:
            return deadline - time.monotonic()

        # 1) Resources and prompts are side-effect free by protocol.
        for res in list(self.resources)[:6]:
            if time_left() < 1 or calls >= max_calls:
                return calls
            key = f"res:{res['uri']}"
            if key in self._probed:
                continue
            self._probed.add(key)
            await self.read_resource(res["uri"])
            calls += 1
        for prompt in list(self.prompts)[:4]:
            if time_left() < 1 or calls >= max_calls:
                return calls
            key = f"prompt:{prompt['name']}"
            if key in self._probed or any(a["required"] for a in prompt["arguments"]):
                continue
            self._probed.add(key)
            await self.get_prompt(prompt["name"])
            calls += 1

        # 2) Unknown / explore / decrypt tools with fillable args.
        order = {"explore": 0, "unknown": 1, "decrypt": 2, "search": 3, "checker": 9}
        candidates = sorted(
            (t for t in self.tools.values() if t.kind in order and not t.destructive),
            key=lambda t: (order[t.kind], len(t.required)),
        )
        for tool in candidates:
            if time_left() < 2 or calls >= max_calls:
                return calls
            if tool.kind == "checker":
                continue  # checkers need a real candidate answer; the solver uses them
            args = synthesize_args(tool, self.ctx)
            if args is None:
                continue
            key = f"tool:{tool.name}:{json.dumps(args, sort_keys=True, default=str)[:200]}"
            if key in self._probed or self.already_called(tool.name, args):
                continue
            self._probed.add(key)
            result = await self.call(tool.name, args, source="probe")
            calls += 1
            if not result.get("error"):
                self.notebook.add(f"probe:{tool.name}", compat.short(result["dict"] or result.get("text"), 900), "probe")

        # 3) Hidden tool names mentioned somewhere but not listed.
        for cand in list(self.notebook.hidden_tool_candidates)[:6]:
            if time_left() < 2 or calls >= max_calls:
                return calls
            if cand in self.tools or f"hidden:{cand}" in self._probed:
                continue
            if classify(cand) in NEVER_AUTO_CALL:
                continue
            self._probed.add(f"hidden:{cand}")
            args: dict[str, Any] = {"agent_id": self.agent_id}
            if self.notebook.keys:
                args[self._key_param_for(cand)] = self.notebook.keys[-1]
            result = await self.call(cand, args, source="hidden-probe")
            calls += 1
            if not result.get("error"):
                self.notebook.add(
                    f"hidden:{cand}",
                    f"Unlisted tool {cand} answered: {compat.short(result['dict'] or result.get('text'), 900)}",
                    "discovery",
                )
                self.tools.setdefault(cand, ToolInfo(cand, "unlisted tool discovered by probing", {}, classify(cand), listed=False))
            else:
                # Retry without guessed key when the server rejected our arguments.
                if "key" in args:
                    result = await self.call(cand, {"agent_id": self.agent_id}, source="hidden-probe")
                    calls += 1
                    if not result.get("error"):
                        self.notebook.add(f"hidden:{cand}", compat.short(result["dict"] or result.get("text"), 900), "discovery")
        return calls

    def _key_param_for(self, tool_name: str) -> str:
        """The argument name a hint used next to this tool ("with key=", "password:")."""
        pattern = re.compile(re.escape(tool_name) + r"[^.\n]{0,40}?\b(key|password|passphrase|code|pin|token|secret)\s*[=:]", re.I)
        for fact in reversed(self.notebook.facts):
            m = pattern.search(fact.text)
            if m:
                return m.group(1).lower()
        return "key"

    def already_called(self, name: str, args: dict[str, Any]) -> bool:
        probe = dict(args)
        probe.setdefault("agent_id", self.agent_id)
        for rec in self.calls:
            if rec.name == name and rec.ok:
                other = dict(rec.args)
                other.setdefault("agent_id", self.agent_id)
                if other == probe:
                    return True
        return False

    def catalog_tools(self) -> list[ToolInfo]:
        """Listed tools whose job is to describe other tools (e.g. arena.tools.list)."""
        rx = re.compile(r"(tools?[._\-]?(list|catalog|index|manifest|discover|available|help)|"
                        r"(list|discover|describe)[._\-]?tools?|catalog|manifest|capabilit)", re.I)
        return [t for t in self.tools.values() if rx.search(t.name) and t.kind not in NEVER_AUTO_CALL and not t.destructive]

    async def discover_catalog(self, timeout_s: float = 4.0) -> list[str]:
        """Call catalog tools once and register any tools they reveal.

        Returns the names of tools that were not in list_tools. The calls are
        cheap and side-effect free by intent, so the controller runs them in
        parallel with clue fetching.
        """
        revealed: list[str] = []
        for tool in self.catalog_tools()[:2]:
            if f"catalog:{tool.name}" in self._probed:
                continue
            self._probed.add(f"catalog:{tool.name}")
            args = synthesize_args(tool, self.ctx)
            if args is None:
                continue
            res = await self.call(tool.name, args, source="catalog", timeout_s=timeout_s)
            if res.get("error"):
                continue
            self.notebook.add(f"catalog:{tool.name}", compat.short(res.get("dict") or res.get("text"), 1500), "discovery")
            for item in _tool_entries(res.get("data") if res.get("data") is not None else res.get("dict")):
                name = str(item.get("name") or item.get("tool") or item.get("id") or "").strip()
                if not name or name in self.tools:
                    continue
                desc = str(item.get("description") or item.get("desc") or item.get("summary") or "")
                schema = item.get("input_schema") or item.get("inputSchema") or item.get("parameters") or item.get("schema") or {}
                if not isinstance(schema, dict):
                    schema = {}
                self.tools[name] = ToolInfo(name, desc or "tool revealed by a catalog tool", schema, classify(name, desc), listed=False)
                revealed.append(name)
                self.notebook.add("discovery", f"Catalog revealed unlisted tool: {self.tools[name].brief()}", "discovery")
                if name in self.notebook.hidden_tool_candidates:
                    self.notebook.hidden_tool_candidates.remove(name)
        return revealed

    def call_log_text(self, limit: int = 25) -> str:
        rows = self.calls[-limit:]
        return "\n".join(f"{r.name}({compat.short(r.args, 120)}) -> {r.summary}" for r in rows)


def _redact(args: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 300:
            out[k] = f"<{len(v)} chars>"
        elif "api_key" in k.lower():
            out[k] = "<redacted>"
        else:
            out[k] = v
    return out


def _tool_entries(data: Any) -> list[dict[str, Any]]:
    """Find tool-like dicts (each with a name) anywhere in a catalog payload."""
    out: list[dict[str, Any]] = []

    def walk(node: Any, depth: int) -> None:
        if depth > 4 or len(out) > 200:
            return
        if isinstance(node, list):
            for item in node:
                if isinstance(item, str) and re.fullmatch(r"[A-Za-z_][\w.\-]{2,63}", item) and ("." in item or "_" in item):
                    out.append({"name": item})
                else:
                    walk(item, depth + 1)
        elif isinstance(node, dict):
            if isinstance(node.get("name"), str) and (
                "description" in node or "input_schema" in node or "inputSchema" in node or "parameters" in node
            ):
                out.append(node)
                return
            for value in node.values():
                walk(value, depth + 1)

    walk(data, 0)
    return out
