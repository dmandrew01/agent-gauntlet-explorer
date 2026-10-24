"""Mock Agent Gauntlet arena with a black-box twist. Development only; never packaged.

One local origin serves the kit's three services:
  /api/*    REST (register, competition, thought, draft, submit, session)
  /sse      MCP over SSE (arena.* tools, image tools, and black-box extras)
  /proxy/*  OpenAI-compatible LLM proxy backed by a scripted "mock brain"

Scenarios rotate per registration (MOCK_ROTATION, comma separated):
  text_logic    plain logic puzzle, no exploring needed
  text_encoded  a clue is base64(rot13(...)); right answer only after decoding
  blackbox      the flag sits behind an unlisted tool named in a hex note
  image_edit    edit an input PNG and submit through arena.image.submit_edit

The mock brain answers correctly only when the needed evidence is present in
its prompt, so a pass proves the agent's plumbing found that evidence.

Run (Python with mcp>=2 or mcp 1.x FastMCP is fine for the client side):
    python3 mock_arena/server.py --port 8765
"""

from __future__ import annotations

import argparse
import base64
import codecs
import itertools
import json
import os
import random
import re
import struct
import sys
import time
import uuid
import zlib
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

try:  # mcp 2.x
    from mcp.server.mcpserver import Context, MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import Context, FastMCP as _Server  # type: ignore

MOCK_KEY = os.getenv("MOCK_KEY", "practice-key")
ROTATION = [s.strip() for s in os.getenv("MOCK_ROTATION", "text_logic,text_encoded,blackbox,image_edit").split(",") if s.strip()]
LOBBY_S = float(os.getenv("MOCK_LOBBY_S", "0"))
FLAG = "FLAG{you_found_the_hidden_vault}"

# ---------------------------------------------------------------------------
# Tiny PNG helpers


def _png(width: int, height: int, color_fn) -> bytes:
    rows = b""
    for y in range(height):
        rows += b"\x00" + b"".join(bytes(color_fn(x, y)) for x in range(width))

    def chunk(t: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + t + body + struct.pack(">I", zlib.crc32(t + body) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def _uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


INPUT_IMAGE = _uri(_png(32, 24, lambda x, y: (70, 130, 220) if y < 12 else (40, 160, 60)))
GENERATED_IMAGE = _uri(_png(24, 24, lambda x, y: (20, 30, 80) if y < 14 else (30, 90, 140)))
EDITED_IMAGE = _uri(_png(32, 24, lambda x, y: (240, 140, 40) if y < 12 else (40, 160, 60)))

# ---------------------------------------------------------------------------
# Scenarios


def scenario_payload(name: str) -> dict[str, Any]:
    if name == "text_logic":
        return {
            "challenge_type": "logic", "difficulty": "easy", "max_time_s": int(os.getenv("MOCK_LOGIC_TIME_S", "60")),
            "description": "Ana, Bob and Cy sit in a row. Who sits in the middle?",
            "rules": "Answer with a single name, capitalized.",
            "clue_texts": ["Ana sits at the left end.", "Cy does not sit next to Ana."],
            "required_tools": ["arena.clues.get"], "expected": "Bob",
        }
    if name == "text_encoded":
        secret = "The constellation you need is ORION"
        blob = base64.b64encode(codecs.encode(secret, "rot_13").encode()).decode()
        return {
            "challenge_type": "cipher", "difficulty": "medium", "max_time_s": 90,
            "description": "An intercepted message names a constellation. Which one?",
            "rules": "Answer in UPPERCASE, single word.",
            "clue_texts": ["The message was scrambled twice before sending.", f"Intercept: {blob}"],
            "required_tools": [], "expected": "ORION",
        }
    if name == "blackbox":
        return {
            "challenge_type": "black-box", "difficulty": "hard", "max_time_s": 120,
            "description": "Somewhere on this server is a vault. Retrieve its flag. Explore the tools; not all are listed.",
            "rules": "Answer with the flag only, exactly as found.",
            "clue_texts": ["The archive remembers what the vault forgot."],
            "required_tools": ["archive.list"], "expected": FLAG,
        }
    if name == "image_edit":
        return {
            "challenge_type": "image", "difficulty": "easy", "max_time_s": 120,
            "description": "Edit the photo.", "prompt": "Make the sky orange like a sunset. Keep the grass unchanged.",
            "reference_notes": "Only the sky changes.", "input_image_uri": INPUT_IMAGE, "expected": "orange-sky",
        }
    if name == "text_order":
        return {
            "challenge_type": "logic", "difficulty": "medium", "max_time_s": 90,
            "description": "Five runners finished a race. List them from first to last.",
            "rules": "Answer with the five letters, comma-separated, first to last.",
            "clue_texts": ["E beat everyone.", "A finished right after E.", "D finished last.", "B beat C."],
            "required_tools": ["arena.clues.get"], "expected": "E, A, B, C, D",
        }
    if name == "image_generate":
        return {
            "challenge_type": "image", "difficulty": "medium", "max_time_s": 150,
            "description": "Generate an image.", "prompt": "A lighthouse at dusk over a calm sea. No text in the image.",
            "reference_notes": "No text, no people.", "input_image_uri": "", "expected": "generated",
        }
    raise ValueError(name)


class Session:
    def __init__(self, agent_id: str, name: str, scenario: str) -> None:
        self.agent_id = agent_id
        self.agent_name = name
        self.session_id = uuid.uuid4().hex[:12]
        self.scenario = scenario
        self.payload = scenario_payload(scenario)
        self.status = "registered"
        self.started_at = time.time()
        self.challenge_started: float | None = None
        self.thoughts: list[str] = []
        self.drafts: list[str] = []
        self.tools_used: set[str] = set()
        self.answer: str | None = None
        self.result: dict[str, Any] | None = None
        self.vault_unlocked = False

    @property
    def modality(self) -> str:
        return "image" if self.scenario.startswith("image") else "text"

    def time_remaining(self) -> float:
        if self.challenge_started is None:
            return float(self.payload["max_time_s"])
        return max(0.0, self.payload["max_time_s"] - (time.time() - self.challenge_started))


SESSIONS: dict[str, Session] = {}
_rotation = itertools.cycle(ROTATION)
_first_register: list[float] = []
USAGE: dict[str, dict[str, int]] = {}


def phase() -> str:
    if not _first_register:
        return "lobby" if LOBBY_S > 0 else "running"
    if time.time() - _first_register[0] < LOBBY_S:
        return "lobby"
    return "running"


def _session(agent_id: str) -> Session | None:
    return SESSIONS.get(agent_id)


# ---------------------------------------------------------------------------
# REST


def _auth(request: Request) -> bool:
    return request.headers.get("x-arena-api-key") == MOCK_KEY


async def health(_: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


async def validate(request: Request) -> JSONResponse:
    return JSONResponse({"valid": _auth(request)})


async def register(request: Request) -> JSONResponse:
    if not _auth(request):
        return JSONResponse({"detail": "bad key"}, status_code=401)
    body = await request.json()
    agent_id = body["agent_id"]
    if not _first_register:
        _first_register.append(time.time())
    scenario = os.getenv("MOCK_SCENARIO") or next(_rotation)
    sess = Session(agent_id, body.get("agent_name", agent_id), scenario)
    SESSIONS[agent_id] = sess
    print(f"[mock] register {agent_id} -> {scenario}", flush=True)
    return JSONResponse({"session_id": sess.session_id, "agent_id": agent_id, "agent_name": sess.agent_name,
                         "status": sess.status, "started_at": sess.started_at})


async def get_session(request: Request) -> JSONResponse:
    sess = _session(request.path_params["agent_id"])
    if not sess:
        return JSONResponse({"detail": "missing"}, status_code=404)
    return JSONResponse({"agent_id": sess.agent_id, "status": sess.status, "challenge_type": sess.modality,
                         "answer": sess.answer, "score": (sess.result or {}).get("score")})


async def competition(_: Request) -> JSONResponse:
    return JSONResponse({"phase": phase(), "usage_scope": "mock-round-1", "eligible_agent_ids": list(SESSIONS),
                         "countdown_value": None})


async def status(request: Request) -> JSONResponse:
    body = await request.json()
    sess = _session(body.get("agent_id", ""))
    if sess and body.get("status") not in (None, "running"):
        pass
    return JSONResponse({"updated": True})


async def thought(request: Request) -> JSONResponse:
    body = await request.json()
    sess = _session(body.get("agent_id", ""))
    if sess:
        sess.thoughts.append(body.get("thought", ""))
    return JSONResponse({"accepted": bool(sess)})


async def draft(request: Request) -> JSONResponse:
    body = await request.json()
    sess = _session(body.get("agent_id", ""))
    if sess:
        sess.drafts.append(body.get("draft", ""))
    return JSONResponse({"saved": bool(sess)})


def score(sess: Session, answer: str) -> dict[str, Any]:
    expected = sess.payload["expected"]
    if sess.modality == "image":
        correct = answer == (GENERATED_IMAGE if sess.scenario == "image_generate" else EDITED_IMAGE)
    else:
        correct = answer.strip() == expected
    missing = [t for t in sess.payload.get("required_tools", []) if t not in sess.tools_used]
    quality = 100.0 if correct and not missing else 0.0
    elapsed = time.time() - (sess.challenge_started or time.time())
    speed = max(0.0, 100.0 * (1 - elapsed / sess.payload["max_time_s"]))
    tools = min(3, len(sess.tools_used)) / 3 * 100
    tokens = USAGE.get(sess.agent_id, {}).get("total_tokens", 0)
    token_score = max(0.0, 100 - tokens / 400)
    final = 0.7 * quality + 0.15 * speed + 0.1 * tools + 0.05 * token_score
    penalty = len(sess.thoughts) < 1
    out = {"final_score": round(final * (0.9 if penalty else 1.0), 2), "quality_score": quality, "speed_score": round(speed, 1),
           "tools_score": round(tools, 1), "tokens_score": round(token_score, 1), "models_score": 100,
           "missing_required_tools": missing, "broadcast_thought_count": len(sess.thoughts),
           "broadcast_min_thoughts": 1, "broadcast_penalty_applied": penalty, "expected_for_debug": expected if sess.modality == "text" else "edited image",
           "tools_used": sorted(sess.tools_used)}
    return {"accepted": correct and not missing, "score": out}


async def submit(request: Request) -> JSONResponse:
    body = await request.json()
    sess = _session(body.get("agent_id", ""))
    if not sess:
        return JSONResponse({"detail": "no session"}, status_code=404)
    return JSONResponse(*_submit(sess, str(body.get("answer", ""))))


def _submit(sess: Session, answer: str) -> tuple[dict[str, Any], int]:
    if sess.answer is not None:
        if answer == sess.answer:
            return sess.result or {}, 200
        return {"detail": {"code": "already_submitted"}}, 409
    sess.answer = answer
    sess.status = "submitted"
    verdict = score(sess, answer)
    sess.result = {"accepted": verdict["accepted"], "agent_id": sess.agent_id, "answer": answer if sess.modality == "text" else "(image)",
                   "status": "submitted", "score": verdict["score"]}
    print(f"[mock] submit {sess.agent_id} {sess.scenario}: {answer[:60]!r} -> {verdict}", flush=True)
    return sess.result, 200


async def leaderboard(_: Request) -> JSONResponse:
    return JSONResponse({"detail": "leaderboard is spectator only"}, status_code=403)


# ---------------------------------------------------------------------------
# LLM proxy with the mock brain


# Names follow the Practice roster's families so the ranker treats them alike.
MODELS = [
    {"id": "mock/gemini-3.8-flash"},
    {"id": "mock/qwen3.7-plus"},
    {"id": "mock/nemotron-3-ultra"},
    {"id": "mock/fast-mini"},
    {"id": "mock/painter-image", "capabilities": ["image"]},
]
# Test knobs: one model answers slowly or wrongly.
SLOW_MODEL = os.getenv("MOCK_SLOW_MODEL", "")
SLOW_S = float(os.getenv("MOCK_SLOW_S", "0") or 0)
WRONG_MODEL = os.getenv("MOCK_WRONG_MODEL", "")


async def models(request: Request) -> JSONResponse:
    if request.headers.get("authorization") != f"Bearer {MOCK_KEY}":
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return JSONResponse({"data": MODELS})


async def usage(request: Request) -> JSONResponse:
    agent = request.path_params.get("agent_id", "")
    return JSONResponse({"usage": USAGE.get(agent, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})})


def _all_text(messages: list[dict[str, Any]]) -> str:
    parts = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            parts += [str(p.get("text", "")) for p in c if isinstance(p, dict)]
        elif c:
            parts.append(str(c))
        for tc in m.get("tool_calls") or []:
            parts.append(json.dumps(tc))
    return "\n".join(parts)


def mock_brain(req: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None, list[dict[str, Any]]]:
    """Returns (final_answer_or_None, plain_text_or_None, tool_calls)."""
    msgs = req.get("messages", [])
    text = _all_text(msgs)
    tools = req.get("tools") or []
    fn_names = {t["function"]["name"] for t in tools if isinstance(t, dict) and "function" in t}
    flag = re.search(r"FLAG\{[a-z_]+\}", text)
    if "Five runners finished" in text:
        ok = all(c in text for c in ("E beat everyone", "D finished last", "B beat C"))
        return {"answer": "E, A, B, C, D" if ok else "E, A, C, B, D"}, None, []
    if "Who sits in the middle" in text:
        return {"answer": "Bob"}, None, []
    if "constellation" in text:
        return {"answer": "ORION" if "constellation you need is ORION" in text else "SIRIUS"}, None, []
    if "vault" in text.lower():
        if flag:
            return {"answer": flag.group(0)}, None, []
        if "vault.open" in text and "ORION" in text:
            if "vault__open" in fn_names:
                return None, None, [{"name": "vault__open", "arguments": {"key": "ORION"}}]
            return None, None, [{"name": "call_unlisted_tool", "arguments": {"name": "vault.open", "arguments": {"key": "ORION"}}}]
        if "archive__list" in fn_names and "archive.list(" not in text and '"name": "archive__list"' not in text:
            return None, None, [{"name": "archive__list", "arguments": {}}]
        hexes = re.findall(r"\b[0-9a-f]{40,}\b", text)
        if hexes and "decode" in fn_names and '"name": "decode"' not in text:
            return None, None, [{"name": "decode", "arguments": {"text": hexes[0]}}]
        return {"answer": "FLAG{guess}"}, None, []
    return None, "ANSWER: 42", []


async def chat(request: Request) -> JSONResponse:
    if request.headers.get("authorization") != f"Bearer {MOCK_KEY}":
        return JSONResponse({"error": {"message": "unauthorized"}}, status_code=401)
    req = await request.json()
    agent = request.headers.get("x-agent-id", "")
    model = req.get("model")
    delay = float(os.getenv("MOCK_LLM_DELAY_S", "0") or 0)
    if delay:
        import asyncio

        await asyncio.sleep(delay)
    if model not in {m["id"] for m in MODELS}:
        return JSONResponse({"error": {"message": f"model {model} not on roster"}}, status_code=400)
    if req.get("tools") and model == "mock/fast-mini":
        return JSONResponse({"error": {"message": "This model does not support tools / function calling"}}, status_code=400)
    if model == SLOW_MODEL and SLOW_S:
        import asyncio

        await asyncio.sleep(SLOW_S)
    final, plain, calls = mock_brain(req)
    if final is not None and model == WRONG_MODEL:
        final = {"answer": "Cy" if final["answer"] == "Bob" else "WRONG"}
    json_mode = "ONE JSON object" in _all_text(req.get("messages", [])) or "one JSON action" in _all_text(req.get("messages", []))
    message: dict[str, Any] = {"role": "assistant", "content": None}
    if req.get("tools"):
        if final is not None:
            calls = [{"name": "final_answer", "arguments": {"answer": final["answer"], "confidence": 0.9}}]
        if calls:
            message["tool_calls"] = [{"id": f"call_{i}_{uuid.uuid4().hex[:6]}", "type": "function",
                                      "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}} for i, c in enumerate(calls)]
        else:
            message["content"] = plain or "ANSWER: 42"
    elif json_mode:
        if final is not None:
            message["content"] = json.dumps({"final_answer": final["answer"], "confidence": 0.8})
        elif calls:
            c = calls[0]
            name = {"vault__open": "vault.open", "archive__list": "archive.list"}.get(c["name"], c["name"])
            message["content"] = json.dumps({"tool": name, "args": c["arguments"]})
        else:
            message["content"] = json.dumps({"final_answer": "42", "confidence": 0.1})
    else:
        message["content"] = f"ANSWER: {final['answer']}\nChecked against every clue." if final else (plain or "ANSWER: 42")
    prompt_tokens = len(json.dumps(req.get("messages", []))) // 4
    completion_tokens = 20 + len(json.dumps(message)) // 4
    u = USAGE.setdefault(agent, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    u["prompt_tokens"] += prompt_tokens
    u["completion_tokens"] += completion_tokens
    u["total_tokens"] = u["prompt_tokens"] + u["completion_tokens"]
    return JSONResponse({
        "id": "chatcmpl-mock", "object": "chat.completion", "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if message.get("tool_calls") else "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens},
    })


# ---------------------------------------------------------------------------
# MCP server


HIDDEN = {"vault.open"}


class BlackBoxServer(_Server):
    async def list_tools(self):  # type: ignore[override]
        tools = await super().list_tools()
        return [t for t in tools if t.name not in HIDDEN]


mcp = BlackBoxServer(
    name="agent-gauntlet-mock",
    instructions="Agent Gauntlet mock arena. Call arena.get_challenge first. Not every tool is listed; read carefully.",
)


def _track(agent_id: str, tool: str) -> Session | None:
    sess = _session(agent_id)
    if sess:
        sess.tools_used.add(tool)
    return sess


def _locked() -> dict[str, Any] | None:
    if phase() != "running":
        return {"error": "Challenge locked: waiting for organizer to start the battle."}
    return None


@mcp.tool(name="arena.get_challenge", description="Get your text challenge and start its timer.")
def get_challenge(agent_id: str) -> dict:
    if (lock := _locked()):
        return lock
    sess = _track(agent_id, "arena.get_challenge")
    if not sess:
        return {"error": "Agent not registered"}
    if sess.modality == "image":
        return {"error": "This agent is assigned an image challenge. Use arena.image.get_challenge."}
    if sess.challenge_started is None:
        sess.challenge_started = time.time()
    p = sess.payload
    return {"challenge_type": p["challenge_type"], "challenge_id": sess.session_id, "puzzle_id": f"mock_{sess.scenario}",
            "difficulty": p["difficulty"], "description": p["description"], "rules": p["rules"], "max_time_s": p["max_time_s"],
            "clues": [], "time_remaining_s": sess.time_remaining(), "required_tools": p.get("required_tools", [])}


@mcp.tool(name="arena.clues.list", description="List the clue ids for your challenge.")
def clues_list(agent_id: str) -> dict:
    sess = _track(agent_id, "arena.clues.list")
    if not sess:
        return {"error": "Agent not registered"}
    return {"clue_ids": [f"clue_{i}" for i, _ in enumerate(sess.payload.get("clue_texts", []))]}


@mcp.tool(name="arena.clues.get", description="Get one clue by id.")
def clues_get(clue_id: str, agent_id: str) -> dict:
    sess = _track(agent_id, "arena.clues.get")
    if not sess:
        return {"error": "Agent not registered"}
    texts = sess.payload.get("clue_texts", [])
    idx = int(clue_id.split("_")[-1]) if clue_id.split("_")[-1].isdigit() else -1
    if not 0 <= idx < len(texts):
        return {"error": f"unknown clue {clue_id}"}
    return {"clue_id": clue_id, "text": texts[idx], "time_remaining_s": sess.time_remaining()}


@mcp.tool(name="arena.time_remaining", description="Seconds left on your challenge.")
def time_remaining(agent_id: str) -> dict:
    sess = _track(agent_id, "arena.time_remaining")
    if not sess:
        return {"error": "Agent not registered"}
    t = sess.time_remaining()
    return {"time_remaining_s": t, "elapsed_s": sess.payload["max_time_s"] - t, "max_time_s": sess.payload["max_time_s"], "expired": t <= 0}


@mcp.tool(name="arena.image.get_challenge", description="Get your image challenge and start its timer.")
def image_get_challenge(agent_id: str) -> dict:
    if (lock := _locked()):
        return lock
    sess = _track(agent_id, "arena.image.get_challenge")
    if not sess:
        return {"error": "Agent not registered"}
    if sess.modality != "image":
        return {"error": "not_image_challenge: the running heat is a text challenge", "code": "not_image_challenge"}
    if sess.challenge_started is None:
        sess.challenge_started = time.time()
    p = sess.payload
    return {"challenge_type": "image_edit", "challenge_id": sess.session_id, "puzzle_id": "mock_image_edit", "difficulty": p["difficulty"],
            "description": p["description"], "prompt": p["prompt"], "reference_notes": p["reference_notes"],
            "max_time_s": p["max_time_s"], "input_image_uri": p["input_image_uri"], "time_remaining_s": sess.time_remaining()}


@mcp.tool(name="arena.image.broadcast_thought", description="Broadcast a thought during an image challenge.")
def image_thought(thought: str, agent_id: str) -> dict:
    sess = _track(agent_id, "arena.image.broadcast_thought")
    if not sess:
        return {"error": "Agent not registered"}
    sess.thoughts.append(thought)
    return {"accepted": True}


@mcp.tool(name="arena.image.submit_edit", description="Submit your final image. The first submission is final.")
def image_submit(edited_image: str, agent_id: str, client_metrics: dict | None = None, rationale: str = "") -> dict:
    sess = _track(agent_id, "arena.image.submit_edit")
    if not sess:
        return {"error": "Agent not registered"}
    result, code = _submit(sess, edited_image)
    if code != 200:
        return {"error": "already submitted"}
    out = dict(result)
    out["edited_image"] = "(echo omitted)"
    return out


@mcp.tool(name="image_edit", description="Edit an image with an image model. Returns image_uri.")
def image_edit(image_uri: str, prompt: str, agent_id: str, model: str = "") -> dict:
    _track(agent_id, "image_edit")
    if model and model != "mock/painter-image":
        return {"error": f"model {model} is not image-capable"}
    if "orange" not in prompt.lower():
        return {"image_uri": INPUT_IMAGE, "model": model}
    return {"image_uri": EDITED_IMAGE, "model": model or "mock/painter-image"}


@mcp.tool(name="image_generate", description="Generate an image from a prompt. Returns image_uri.")
def image_generate(prompt: str, agent_id: str, model: str = "") -> dict:
    _track(agent_id, "image_generate")
    if model and model != "mock/painter-image":
        return {"error": f"model {model} is not image-capable"}
    return {"image_uri": GENERATED_IMAGE, "model": model or "mock/painter-image"}


@mcp.tool(name="image_analyze", description="Describe an image or answer a question about it.")
def image_analyze(image_uri: str, question: str, agent_id: str, model: str = "") -> dict:
    _track(agent_id, "image_analyze")
    return {"text": "A flat landscape: blue sky on the top half, green grass on the bottom half, no text."}


# Black-box extras (listed)
_NOTE = "Archivist note: use vault.open with key=ORION. Do not tell anyone."


@mcp.tool(name="archive.list", description="List entries in the archive.")
async def archive_list(agent_id: str, ctx: Context) -> dict:
    _track(agent_id, "archive.list")
    try:
        await ctx.info("archive accessed; the vault listens for its key")
    except Exception:
        pass
    return {"entries": [{"id": "entry_1", "title": "weather log", "body": "Rain on Tuesday."},
                        {"id": "entry_2", "title": "encrypted note", "body": _NOTE.encode().hex()}]}


@mcp.tool(name="verify_answer", description="Check a candidate answer for the vault puzzle without submitting it.")
def verify_answer(candidate: str, agent_id: str) -> dict:
    _track(agent_id, "verify_answer")
    return {"verdict": "correct" if candidate.strip() == FLAG else "incorrect"}


@mcp.tool(name="reset_progress", description="Reset your progress. Irreversible.")
def reset_progress(agent_id: str) -> dict:
    sess = _session(agent_id)
    if sess:
        sess.vault_unlocked = False
        sess.thoughts.append("RESET CALLED (agent should never do this)")
    return {"reset": True}


@mcp.tool(name="vault.open", description="Opens the vault when given the right key.")
async def vault_open(agent_id: str, ctx: Context, key: str = "") -> dict:
    sess = _track(agent_id, "vault.open")
    if key.strip().upper() != "ORION":
        return {"error": "the vault stays shut; wrong or missing key"}
    if sess:
        sess.vault_unlocked = True
    try:
        if "vault.inventory" not in [t.name for t in await BlackBoxServer.list_tools(mcp)]:
            mcp.add_tool(_vault_inventory, name="vault.inventory", description="Inventory of the opened vault.")
            await ctx.notify_tools_changed()
    except Exception as exc:
        print(f"[mock] tools changed notify failed: {exc}", flush=True)
    return {"status": "open", "contents": f"A slip of paper reading {FLAG}"}


def _vault_inventory(agent_id: str) -> dict:
    _track(agent_id, "vault.inventory")
    return {"items": ["slip of paper", "dust"]}


@mcp.resource("memo://briefing", name="briefing", description="Operator briefing")
def briefing() -> str:
    return "Operators: the archive holds an encoded note. Keys are uppercase."


@mcp.prompt(name="hint", description="A gentle hint")
def hint_prompt() -> str:
    return "Look in the archive before you look in the vault."


# ---------------------------------------------------------------------------
# App


def build_app() -> Starlette:
    mcp_app = mcp.sse_app()
    routes = [
        Route("/api/health", health),
        Route("/api/keys/validate", validate),
        Route("/api/session/register", register, methods=["POST"]),
        Route("/api/session/{agent_id}", get_session),
        Route("/api/competition", competition),
        Route("/api/status", status, methods=["POST"]),
        Route("/api/thought", thought, methods=["POST"]),
        Route("/api/draft", draft, methods=["POST"]),
        Route("/api/submit", submit, methods=["POST"]),
        Route("/api/leaderboard", leaderboard),
        Route("/proxy/models", models),
        Route("/proxy/chat/completions", chat, methods=["POST"]),
        Route("/proxy/usage/{scope}/{agent_id}", usage),
        Route("/proxy/usage/{agent_id}", usage),
        Mount("/", app=mcp_app),
    ]
    return Starlette(routes=routes, lifespan=getattr(mcp_app.router, "lifespan_context", None))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    print(f"[mock] arena on http://127.0.0.1:{args.port} rotation={ROTATION} lobby={LOBBY_S}s", flush=True)
    uvicorn.run(build_app(), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
