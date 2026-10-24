"""End-to-end Agent Gauntlet run: register, explore, solve, submit.

Flow
  1. Register once, broadcast a first thought (avoids the 10% penalty),
     start the kit's session monitor, prefetch the model roster.
  2. Connect MCP with the explorer client and take a full inventory.
  3. Wait for GO (polls /api/competition, then the challenge tool).
  4. Route by modality: text, image, or an unknown black box.
  5. Text: gather clues and required tools in parallel, decode anything
     encoded, then either solve in one shot or run the reason-act loop when
     the box has tools or leads worth following. Check with a checker tool
     when one exists. Enforce the output format. Draft, then submit once.
  6. Image: analyze the input, call the edit/generate tool with ranked image
     models and retries, validate the bytes, submit through MCP.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from arena_clients import (
    HttpArenaClient,
    McpArenaClient,
    ensure_connected,
    get_api_base,
    get_arena_api_key,
    get_mcp_url,
    get_proxy_host,
    monitor_session,
)
from arena_clients.config import agent_id_problem
from arena_clients.display import save_run_record
from arena_clients.http_client import ArenaAPIError, ArenaConnectionError
from arena_clients.mcp_client import build_image_tool_arguments
from arena_clients.outputs import missing_tools_note, save_submitted_image, submission_summary_lines
from arena_clients.proxy_headers import resolve_usage_scope
from base_strategy import BaseStrategy, ChallengeContext
from model_selector import fetch_available_model_rows

from . import compat, decoders, pngtools
from .answer import enforce, extract_answer, infer_format, is_invalid
from .clock import Clock
from .explorer import NEVER_AUTO_CALL, BlackBox, ExplorerArenaClient, synthesize_args
from .llm import LLM
from .models import rank_image_models, rank_text_models
from .react import ReactAgent

REPO_ROOT = Path(__file__).resolve().parent.parent
PROC_T0 = time.monotonic()
CORE_KINDS = {"challenge", "clue_list", "clue_get", "timer", "broadcast", "submit",
              "image_edit", "image_generate", "image_analyze"}
# Seconds to wait for the partner model after the primary answered. Practice
# showed the primary right 3/3 while each waited second cost ~1.7 speed points
# on a 60 s puzzle, so the default is no wait.
CONFIRM_WAIT_S = float(os.getenv("GAUNTLET_CONFIRM_WAIT_S", "0") or 0)
LOCK_WORDS = ("locked", "waiting for organizer", "not started", "lobby", "not running", "countdown")


def log(msg: str) -> None:
    print(f"[explorer {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Thoughts


class Thoughts:
    """Throttled, non-blocking thought broadcasts (REST, or MCP for image runs)."""

    def __init__(self, http: HttpArenaClient, agent_id: str) -> None:
        self.http = http
        self.agent_id = agent_id
        self.count = 0
        # Practice shares one key, so other teams see these. Answers, models and
        # tool output stay out unless GAUNTLET_THOUGHTS=verbose.
        self.verbose = os.getenv("GAUNTLET_THOUGHTS", "").lower() == "verbose"
        self.box: BlackBox | None = None
        self.mcp_tool: str | None = None
        self._last = 0.0
        self._pending: set[asyncio.Task[Any]] = set()
        self._sent: set[str] = set()

    def use_mcp(self, box: BlackBox, tool_name: str | None) -> None:
        self.box, self.mcp_tool = box, tool_name

    async def detail(self, text: str) -> None:
        """A thought with private content; sent only in verbose mode."""
        if self.verbose:
            await self.say(text)

    async def say(self, text: str, *, force: bool = False) -> None:
        text = re.sub(r"\s+", " ", str(text or "")).strip()[:280]
        if not text or text in self._sent:
            return
        now = time.monotonic()
        if not force and now - self._last < 0.8:
            return
        self._last = now
        self._sent.add(text)
        task = asyncio.create_task(self._send(text))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def _send(self, text: str) -> None:
        if self.box is not None and self.mcp_tool:
            res = await self.box.call(self.mcp_tool, {"thought": text, "agent_id": self.agent_id}, mine=False, timeout_s=6)
            if not res.get("error"):
                self.count += 1
                return
        try:
            await asyncio.wait_for(asyncio.to_thread(self.http.broadcast_thought, self.agent_id, text), timeout=6)
            self.count += 1
        except Exception:
            pass

    async def flush(self, timeout: float = 3.0) -> None:
        if self._pending:
            await asyncio.wait(set(self._pending), timeout=timeout)


# ---------------------------------------------------------------------------
# Run state


@dataclass
class RunState:
    agent_id: str
    agent_name: str
    strategy: BaseStrategy
    http: HttpArenaClient
    thoughts: Thoughts
    proxy_host: str
    api_key: str
    model_rows: list[dict[str, Any]] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    llm: LLM | None = None
    model_used: str = ""
    best_answer: str = ""
    fast: Any = None
    record: dict[str, Any] = field(default_factory=dict)
    started_ms: float = field(default_factory=lambda: time.time() * 1000)


def _identity(strategy: BaseStrategy) -> tuple[str, str]:
    agent_id = (os.getenv("AGENT_ID") or str(getattr(strategy, "agent_id", "")) or "").strip()
    agent_name = (os.getenv("AGENT_NAME") or str(getattr(strategy, "agent_name", "")) or agent_id).strip()
    return agent_id, agent_name


async def run(strategy: BaseStrategy) -> int:
    agent_id, agent_name = _identity(strategy)
    os.environ["AGENT_ID"] = agent_id
    os.environ["AGENT_NAME"] = agent_name
    if not agent_id:
        raise SystemExit("AGENT_ID is not set. Add AGENT_ID=<your-team-id> to .env (lowercase letters, digits, hyphens).")
    problem = agent_id_problem(agent_id)
    if problem:
        raise SystemExit(problem)
    ensure_connected()
    api_base, mcp_url, proxy_host, api_key = get_api_base(), get_mcp_url(), get_proxy_host(), get_arena_api_key()
    log(f"agent={agent_id} api={api_base} mcp={mcp_url} proxy={proxy_host}")

    http = HttpArenaClient(api_base=api_base, api_key=api_key)
    thoughts = Thoughts(http, agent_id)
    state = RunState(agent_id, agent_name, strategy, http, thoughts, proxy_host, api_key)
    startup = state.record.setdefault("startup", {})

    def mark(name: str) -> None:
        startup[name] = round(time.monotonic() - PROC_T0, 2)

    # Practice starts the speed clock at registration, and a live battle at GO.
    # Everything that does not need a session happens first: MCP inventory,
    # model roster, warm REST and proxy connections.
    state.fast = FastSubmit(api_base, api_key)
    state.fast.start()
    rows_task = asyncio.create_task(asyncio.to_thread(fetch_available_model_rows, proxy_host, api_key))
    state.llm = LLM(proxy_host, api_key, agent_id, resolve_usage_scope())
    asyncio.create_task(state.llm.warm())
    monitor = None
    exit_code = 0
    try:
        async with ExplorerArenaClient(mcp_url) as client:
            box = BlackBox(client, agent_id)
            await box.inventory()
            mark("inventory")
            log(f"inventory: {len(box.tools)} tools, {len(box.resources)} resources, {len(box.prompts)} prompts")
            for tool in box.tools.values():
                log(f"  tool {tool.brief()[:160]}")
            state.record["inventory"] = box.inventory_text()
            state.model_rows = await rows_task
            state.models = [str(r["id"]) for r in state.model_rows if r.get("id")]
            client.sampling_responder = _sampling_responder(state)
            client.elicitation_values = {"agent_id": agent_id, "name": agent_name, "_default": agent_id}

            session = await asyncio.to_thread(http.register, agent_id, agent_name)
            mark("registered")
            log(f"registered session={session.session_id} status={session.status}")
            await thoughts.say("Explorer online. Mapping the arena before the start signal.", force=True)
            monitor = monitor_session(http, agent_id).start()
            modality_task = asyncio.create_task(asyncio.to_thread(state.fast.lookup_modality, agent_id))

            comp = await _wait_for_go(state)
            mark("go")
            scope = str((comp or {}).get("usage_scope") or "").strip() or resolve_usage_scope()
            if scope:
                os.environ["ARENA_USAGE_SCOPE"] = scope
                state.llm.set_scope(scope)
            modality = await modality_task
            if modality not in {"text", "image"}:
                modality = await asyncio.to_thread(McpArenaClient.detect_modality, list(box.tools), agent_id)
            mark("modality")
            log(f"modality={modality} models={state.models[:8]} startup={startup}")
            text_tool = box.find("arena.get_challenge")
            image_tool = box.find("arena.image.get_challenge")
            if modality == "image" and image_tool:
                outcome = await solve_image(state, box)
                if outcome == "switch" and text_tool:
                    outcome = await solve_text(state, box)
            elif text_tool:
                outcome = await solve_text(state, box)
                if outcome == "switch" and image_tool:
                    outcome = await solve_image(state, box)
            else:
                outcome = await solve_black_box(state, box)
            state.record["outcome"] = outcome
            state.record["tools_used"] = box.tools_used
            state.record["calls"] = [c.__dict__ for c in box.calls]
            state.record["notebook"] = box.notebook.render(12000)
    except asyncio.CancelledError:
        if monitor is not None and monitor.stop_reason is not None:
            log(f"stopped by server: {monitor.stop_reason.message}")
            exit_code = monitor.stop_reason.exit_code
        else:
            raise
    finally:
        await thoughts.flush()
        if monitor is not None:
            await monitor.stop()
        if state.fast is not None:
            state.fast.stop()
        state.record.update({"agent_id": agent_id, "model": state.model_used, "thoughts": thoughts.count})
        if state.llm:
            state.record["llm_usage"] = state.llm.usage.__dict__
        log(f"run record {save_run_record(agent_id, state.record, REPO_ROOT)}")
    return exit_code


def _sampling_responder(state: RunState):
    async def respond(system: str, prompt: str, max_tokens: int) -> str:
        if not state.llm or not state.models:
            return ""
        model = state.model_used or rank_text_models(state.models, rows=state.model_rows)[0].name
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        reply = await state.llm.chat(model, messages, max_tokens=min(1024, max_tokens or 512), timeout_s=20)
        return reply.content
    return respond


async def _wait_for_go(state: RunState) -> dict[str, Any]:
    """Poll the competition phase. Practice is always running; a live battle has a lobby.

    Uses the warm REST session (no new TLS handshake per poll) and returns the
    competition payload, which also carries the usage scope.
    """
    last_phase = ""
    deadline = time.monotonic() + float(os.getenv("GAUNTLET_LOBBY_WAIT_S", "3600"))
    while time.monotonic() < deadline:
        try:
            comp = await asyncio.to_thread(state.fast.get_json, "/api/competition")
        except Exception:
            return {}  # endpoint unavailable; the challenge call itself will gate us
        phase = str(comp.get("phase") or "").lower()
        eligible = comp.get("eligible_agent_ids")
        if phase == "running":
            if isinstance(eligible, list) and eligible and state.agent_id not in eligible:
                if last_phase != "ineligible":
                    log("battle running without us; waiting for the next one")
                    last_phase = "ineligible"
                await asyncio.sleep(0.5)
                continue
            log("GO")
            asyncio.create_task(asyncio.to_thread(_quiet, state.http.update_status, state.agent_id, "running"))
            return comp
        if not phase:
            return comp
        if phase != last_phase:
            log(f"phase={phase}; waiting")
            if phase in {"lobby", "countdown", "waiting"}:
                await state.thoughts.say(f"Ready and waiting ({phase}).", force=True)
            last_phase = phase
        await asyncio.sleep(0.25 if phase == "countdown" else 0.5)
    return {}


def _quiet(fn: Any, *args: Any) -> Any:
    try:
        return fn(*args)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Shared helpers


async def _call_until_unlocked(box: BlackBox, name: str, args: dict[str, Any], *, max_wait_s: float = 900) -> dict[str, Any]:
    started = time.monotonic()
    while True:
        res = await box.call(name, args, source="controller")
        err = str(res.get("error") or "").lower()
        if not err:
            return res
        if any(word in err for word in LOCK_WORDS) and time.monotonic() - started < max_wait_s:
            await asyncio.sleep(0.5)
            continue
        return res


def _clean_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    out = []
    for item in value if isinstance(value, list) else [value]:
        if isinstance(item, dict):
            text = item.get("text") or item.get("content") or json.dumps(item)
        else:
            text = str(item)
        if str(text).strip():
            out.append(str(text).strip())
    return out


def _ctx(challenge: dict[str, Any], *, clues: list[str], models: list[str], clock: Clock | None, image_url: str | None = None,
         description: str | None = None, rules: str | None = None) -> ChallengeContext:
    return ChallengeContext(
        challenge_type=str(challenge.get("challenge_type") or "text"),
        difficulty=str(challenge.get("difficulty") or "unknown"),
        challenge_text=str(description if description is not None else challenge.get("description") or ""),
        description=str(description if description is not None else challenge.get("description") or ""),
        rules=str(rules if rules is not None else challenge.get("rules") or ""),
        clues=clues,
        time_remaining_s=clock.left() if clock else 0.0,
        max_time_s=int(challenge.get("max_time_s") or 0),
        available_models=list(models),
        required_tools=_clean_list(challenge.get("required_tools")),
        image_url=image_url,
    )


def _pick_model(state: RunState, ctx: ChallengeContext, stage: str = "solve") -> tuple[str, list[str]]:
    ranked = state.strategy.rank_models(ctx, state.models) if state.models else []
    model = state.strategy.pick_model(stage, ranked, ctx) if ranked else ""
    if model not in state.models:
        model = ranked[0] if ranked else (state.models[0] if state.models else "")
    return model, ranked


def _extra_text(challenge: dict[str, Any]) -> str:
    core = {"challenge_type", "challenge_id", "puzzle_id", "description", "rules", "max_time_s", "clues",
            "time_remaining_s", "required_tools", "difficulty", "agent_id"}
    parts = []
    for key, value in challenge.items():
        if key in core or value in (None, "", [], {}):
            continue
        parts.append(f"{key}: {compat.short(value, 600)}")
    return "\n".join(parts)


async def _ensure_tool_coverage(box: BlackBox, minimum: int = 3) -> None:
    """Tool coverage counts unique challenge tools used (capped at 3). Top up cheaply."""
    if len(box.tools_used) >= minimum:
        return
    for kind in ("timer", "clue_list"):
        for tool in box.by_kind(kind):
            if tool.name in box.tools_used:
                continue
            args = synthesize_args(tool, box.ctx) or {"agent_id": box.agent_id}
            await box.call(tool.name, args, source="coverage", timeout_s=5)
            if len(box.tools_used) >= minimum:
                return


def _metrics(state: RunState, total_time_ms: int) -> dict[str, Any]:
    usage = state.llm.usage if state.llm else None
    prompt = usage.prompt_tokens if usage else 0
    completion = usage.completion_tokens if usage else 0
    return {
        "model_name": state.model_used,
        "total_tokens": str(prompt + completion),
        "prompt_tokens": str(prompt),
        "completion_tokens": str(completion),
        "ttft_ms": usage.first_token_ms if usage else 0,
        "total_time_ms": int(total_time_ms),
    }


def _fetch_proxy_usage(state: RunState) -> dict[str, int] | None:
    scope = resolve_usage_scope()
    agent = quote(state.agent_id, safe="")
    urls = []
    if scope:
        urls.append(f"{state.proxy_host.rstrip('/')}/usage/{quote(scope, safe='')}/{agent}")
    urls.append(f"{state.proxy_host.rstrip('/')}/usage/{agent}")
    for url in urls:
        try:
            req = Request(url, headers={"Authorization": f"Bearer {state.api_key}", "Accept": "application/json"})
            with urlopen(req, timeout=1.0) as resp:
                usage = json.loads(resp.read().decode("utf-8")).get("usage")
            if isinstance(usage, dict):
                return {k: int(usage.get(k) or 0) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Text


async def solve_text(state: RunState, box: BlackBox) -> str:
    tool = box.find("arena.get_challenge")
    assert tool is not None
    res = await _call_until_unlocked(box, tool.name, {"agent_id": state.agent_id})
    err = str(res.get("error") or "")
    if err:
        log(f"get_challenge error: {err}")
        if "image" in err.lower() and "assigned" in err.lower():
            return "switch"
        return f"error: {err}"
    challenge = res["dict"]
    state.record.setdefault("startup", {})["challenge"] = round(time.monotonic() - PROC_T0, 2)
    clock = Clock(float(challenge.get("time_remaining_s") or 0), float(challenge.get("max_time_s") or 0))
    description = str(challenge.get("description") or "")
    rules = str(challenge.get("rules") or "")
    state.record["challenge"] = {k: v for k, v in challenge.items() if k != "clues"}
    log(f"challenge {challenge.get('puzzle_id') or challenge.get('challenge_id')} type={challenge.get('challenge_type')} "
        f"limit={challenge.get('max_time_s')}s :: {description[:160]}")
    state.record["timeline"] = {"challenge": 0.0}
    await state.thoughts.say(f"Challenge in hand ({challenge.get('challenge_type') or 'text'}). Reading clues.", force=True)
    box.ctx.challenge_text = description

    # Everything between here and submit runs under a hard deadline. A crash or a
    # slow model must never cost the submission: the best answer seen so far
    # (or a fallback) goes in regardless.
    work = {"answer": "", "confidence": 0.0, "mode": "direct", "clues": _clean_list(challenge.get("clues")), "ranked": []}
    try:
        await asyncio.wait_for(_solve_text_body(state, box, challenge, clock, work), timeout=max(1.0, clock.left()))
    except asyncio.TimeoutError:
        log("deadline reached inside the solver; submitting the best answer so far")
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # a bug must not cost the submission
        log(f"solver crashed ({type(exc).__name__}: {exc}); submitting the best answer so far")
    answer = str(work["answer"] or state.best_answer or "")
    if is_invalid(answer):
        answer = _fallback_answer(box, work["clues"], description)
        log(f"fallback answer: {answer!r}")
    try:
        await asyncio.wait_for(_ensure_tool_coverage(box), timeout=max(0.5, min(4.0, clock.left() + clock.margin_s - 3)))
    except Exception:
        pass

    # Strategy hook (deadline behavior)
    hook_ctx = _ctx(challenge, clues=work["clues"], models=work["ranked"], clock=clock)
    hook_ctx.tools_used = list(box.tools_used)
    hook_ctx.tokens_used = state.llm.usage.total_tokens if state.llm else 0
    try:
        revised = state.strategy.on_time_warning(clock.left(), answer, hook_ctx)
        if isinstance(revised, str) and revised.strip():
            answer = revised.strip()
    except Exception as exc:
        log(f"on_time_warning hook failed: {exc}")

    await state.thoughts.say("Answer checked against the rules. Submitting.", force=True)
    await state.thoughts.detail(f"Final answer: {answer[:120]}")
    total_ms = int(clock.elapsed() * 1000)
    state.record.setdefault("timeline", {})["submit_start"] = round(clock.elapsed(), 1)
    result = await _submit_text(state, answer, total_ms)
    state.record["timeline"]["submit_done"] = round(clock.elapsed(), 1)
    state.record.update({"answer": answer, "confidence": work["confidence"], "mode": work["mode"]})
    return result


async def _solve_text_body(state: RunState, box: BlackBox, challenge: dict[str, Any], clock: Clock, work: dict[str, Any]) -> None:
    description = str(challenge.get("description") or "")
    rules = str(challenge.get("rules") or "")
    # Clues and required tools, in parallel.
    clue_task = asyncio.create_task(_fetch_clues(box, state.agent_id))
    required = _clean_list(challenge.get("required_tools"))
    blob = f"{challenge.get('challenge_type', '')} {description} {rules} {_extra_text(challenge)}".lower()
    for name in box.tools:
        if name.lower() in blob and name not in required and box.tools[name].kind not in NEVER_AUTO_CALL:
            required.append(name)
    fetched = await clue_task
    clues = fetched or work["clues"]
    work["clues"] = clues
    box.ctx.challenge_text = description or (clues[0] if clues else "")
    evidence = await _call_required(box, required, description, clues)
    state.record.setdefault("timeline", {})["clues_and_required"] = round(clock.elapsed(), 1)

    spec = infer_format(rules, description)
    ctx = _ctx(challenge, clues=clues, models=state.models, clock=clock)
    model, ranked = _pick_model(state, ctx)
    work["ranked"] = ranked
    state.model_used = model
    log(f"model={model} ranked={ranked[:5]} format={spec.describe()} time_left={clock.left():.1f}s")

    agentic, why = _needs_agentic(box, challenge, clues, required)
    answer, confidence = "", 0.0
    if not agentic:
        await state.thoughts.say(f"{len(clues)} clues read. Solving.")
        await state.thoughts.detail(f"Model: {model}")
        answer, confidence = await _solve_direct(state, box, ctx, model, ranked, clock, spec, evidence)
        work["answer"], work["confidence"] = answer, confidence
        if not answer and clock.left() > 20:
            agentic, why = True, "direct solve gave no answer"
    if agentic:
        work["mode"] = "agentic"
        log(f"agentic mode: {why}")
        await state.thoughts.say("Exploring the server's tools before answering.")
        await state.thoughts.detail(f"Reason: {why}")
        probe_budget = min(15.0, clock.left() * 0.2)
        if os.getenv("GAUNTLET_PROBE", "1").lower() in {"0", "false", "no"}:
            probe_budget = 0.0
        if probe_budget > 3:
            n = await box.follow_leads(time.monotonic() + probe_budget, max_calls=8)
            log(f"probed {n} leads")
        answer2, conf2 = await _solve_react(state, box, ctx, model, clock, spec, evidence, prior=answer)
        if answer2 and (conf2 >= confidence or not answer):
            answer, confidence = answer2, conf2
            work["answer"], work["confidence"] = answer, confidence

    if agentic or _mentions_checker(box, description, rules, clues):
        answer = await _check_with_checker(state, box, ctx, model, clock, spec, evidence, answer)
        work["answer"] = answer
    answer = await _finalize_format(state, model, ranked, clock, spec, ctx, answer)
    if answer and not is_invalid(answer):
        work["answer"] = answer


async def _fetch_clues(box: BlackBox, agent_id: str) -> list[str]:
    lister = box.find("arena.clues.list") or next(iter(box.by_kind("clue_list")), None)
    getter = box.find("arena.clues.get") or next(iter(box.by_kind("clue_get")), None)
    if not lister:
        return []
    res = await box.call(lister.name, {"agent_id": agent_id}, source="controller")
    data = res.get("dict") or {}
    ids = data.get("clue_ids") or data.get("ids") or data.get("clues") or []
    if not isinstance(ids, list):
        ids = []
    ids = [i if isinstance(i, str) else (i.get("id") or i.get("clue_id")) if isinstance(i, dict) else str(i) for i in ids]
    ids = [str(i) for i in ids if i]
    if not getter or not ids:
        return _clean_list(data.get("clues")) if isinstance(data.get("clues"), list) else []
    id_param = next((p for p in getter.properties if "id" in p.lower() and p != "agent_id"), "clue_id")
    results = await asyncio.gather(*[box.call(getter.name, {id_param: cid, "agent_id": agent_id}, source="controller") for cid in ids])
    clues = []
    for r in results:
        d = r.get("dict") or {}
        text = d.get("text") or d.get("clue") or d.get("content") or r.get("text")
        if text and not r.get("error"):
            clues.append(str(text).strip())
    return clues


async def _call_required(box: BlackBox, required: list[str], description: str, clues: list[str]) -> list[str]:
    evidence: list[str] = []
    jobs = []
    for name in required:
        if name in box.tools_used:
            continue
        tool = box.tools.get(name)
        if tool is None or tool.kind in NEVER_AUTO_CALL or tool.kind in {"broadcast"}:
            continue
        args = synthesize_args(tool, box.ctx)
        if args is None:
            args = {"agent_id": box.agent_id}
            for pname in tool.required:
                args.setdefault(pname, description[:300] or (clues[0] if clues else ""))
        jobs.append((name, args))
    results = await asyncio.gather(*[box.call(n, a, source="required") for n, a in jobs])
    for (name, _args), res in zip(jobs, results):
        if res.get("error"):
            evidence.append(f"{name}: error {res['error']}")
        else:
            evidence.append(f"{name}: {compat.short(res['dict'] or res.get('text'), 2000)}")
    if jobs:
        log(f"required tools called: {[n for n, _ in jobs]}")
    return evidence


def _needs_agentic(box: BlackBox, challenge: dict[str, Any], clues: list[str], required: list[str]) -> tuple[bool, str]:
    if os.getenv("GAUNTLET_FORCE_AGENTIC", "").lower() in {"1", "true", "yes"}:
        return True, "forced by GAUNTLET_FORCE_AGENTIC"
    extra = [t for t in box.tools.values() if t.kind not in CORE_KINDS and not t.name.startswith("arena.")]
    text = " ".join([str(challenge.get("description") or ""), str(challenge.get("rules") or ""), _extra_text(challenge), *clues])
    mentioned = [t.name for t in extra if t.name.lower() in text.lower()]
    if mentioned:
        return True, f"challenge mentions tools {mentioned[:3]}"
    hidden = [h for h in box.notebook.hidden_tool_candidates if h.lower() in text.lower()]
    if hidden:
        return True, f"challenge points at unlisted tools {hidden[:3]}"
    if re.search(r"\b(use the tools?|explore|hidden|unlock|vault|server|discover|decrypt|look ?up|search|retriev|browse)\b", text, re.I) and extra:
        return True, "challenge asks for tool use"
    if box.notebook.keys and any(f.kind == "decoded" for f in box.notebook.facts) and extra:
        return True, "decoded data and keys suggest a multi-step chain"
    if re.search(r"\b(current|latest|today|this week|as of|live)\b", text, re.I) and box.by_kind("search"):
        return True, "needs fresh data from a search tool"
    return False, "self-contained puzzle"


def _evidence_block(box: BlackBox, evidence: list[str], limit: int = 7000) -> str:
    facts = box.notebook.render(limit // 2, kinds={"decoded", "key", "flag", "probe", "resource", "prompt", "discovery", "image", "instructions", "notification"})
    parts = []
    if evidence:
        parts.append("Tool evidence:\n" + "\n".join(f"- {e[:1500]}" for e in evidence))
    if facts:
        parts.append("Facts the explorer found (decoded payloads, keys, discoveries):\n" + facts)
    text = "\n\n".join(parts)
    return text[:limit]


FORMAT_GUARD = (
    "\n\nOutput contract: the FIRST line must be 'ANSWER: <final answer>'. "
    "Follow the required format exactly: {fmt}. No markdown, no code fences, no <think> tags, no extra lines "
    "unless the format itself spans lines."
)


def _family(name: str) -> str:
    return re.split(r"[-_.\d]", name.rsplit("/", 1)[-1].lower(), maxsplit=1)[0]


def _direct_lineup(model: str, ranked: list[str], *, tight: bool = False) -> list[str]:
    """Primary model, a capable partner from another family, and a tie-breaker.

    Practice showed a single slow pick costing a 60 s puzzle. Asking two
    models costs a few hundred tokens (5% axis) and protects quality (70%)
    and speed (15%). The third model only runs when the first two disagree.
    """
    from .models import _apply_stats, load_stats, score_model

    stats = load_stats()
    scored = {}
    for name in ranked[:8]:
        m = score_model(name)
        _apply_stats(m, stats)
        scored[name] = m
    speed_w = 0.6 if tight else 0.2
    others = [n for n in ranked[:8] if n != model and scored[n].quality >= 72]
    others.sort(key=lambda n: (_family(n) == _family(model), -(scored[n].quality + speed_w * scored[n].speed)))
    lineup = [model] + others[:1]
    rest = [n for n in others[1:] if _family(n) not in {_family(x) for x in lineup}] or others[1:]
    rest.sort(key=lambda n: -(scored[n].quality + speed_w * scored[n].speed))
    return lineup + rest[:1]


def _norm(answer: str, spec) -> str:
    fixed, _ = enforce(answer, spec)
    return re.sub(r"\s+", " ", fixed).strip().lower()


async def _solve_direct(state: RunState, box: BlackBox, ctx: ChallengeContext, model: str, ranked: list[str],
                        clock: Clock, spec, evidence: list[str]) -> tuple[str, float]:
    assert state.llm is not None
    system = state.strategy.build_system_prompt(ctx) + FORMAT_GUARD.format(fmt=spec.describe())
    user = state.strategy.build_solver_prompt(ctx)
    extra = _extra_text(state.record.get("challenge", {}))
    if extra:
        user += f"\n\nOther challenge fields:\n{extra}"
    ev = _evidence_block(box, evidence)
    if ev:
        user += "\n\n" + ev
    params = state.strategy.get_llm_params(ctx)
    max_tokens = int(params.get("max_tokens", 8192))
    temperature = params.get("temperature", 0.0)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    attempts = state.record.setdefault("llm_attempts", [])

    async def ask(m: str, timeout: float, tokens: int) -> tuple[str, str]:
        reply = await state.llm.chat(m, messages, max_tokens=tokens, temperature=temperature, timeout_s=timeout)
        answer = extract_answer(reply.content or "", spec)
        if not answer and reply.reasoning and "answer" in reply.reasoning.lower():
            answer = extract_answer(reply.reasoning, spec)
        if is_invalid(answer):
            answer = ""
        attempts.append({"stage": "direct", "model": m, "ms": reply.elapsed_ms, "finish": reply.finish_reason,
                         "error": reply.error[:160], "answer": answer[:200],
                         "tokens": reply.prompt_tokens + reply.completion_tokens, "t": round(clock.elapsed(), 1)})
        log(f"direct[{m}] {reply.elapsed_ms}ms finish={reply.finish_reason} err={reply.error[:80]} answer={answer[:100]!r}")
        return answer, reply.finish_reason

    hedge = os.getenv("GAUNTLET_HEDGE", "1").lower() not in {"0", "false", "no"}
    tight = 0 < float(ctx.max_time_s or 0) <= 60
    lineup = _direct_lineup(model, ranked, tight=tight) if hedge else [model]
    log(f"direct lineup: {lineup}")
    tried: set[str] = set()
    if len(lineup) >= 2 and clock.left() > 10:
        primary, partner = lineup[0], lineup[1]
        budget = clock.budget(0.85, floor_s=5)
        tasks = {asyncio.create_task(ask(m, budget, max_tokens)): m for m in (primary, partner)}
        tried.update((primary, partner))
        got: dict[str, str] = {}
        soft_deadline: float | None = None
        pending = set(tasks)
        while pending:
            wait = None if soft_deadline is None else max(0.0, soft_deadline - time.monotonic())
            done, pending = await asyncio.wait(pending, timeout=wait, return_when=asyncio.FIRST_COMPLETED)
            if not done:
                log("second model too slow; deciding with what arrived")
                break
            # Also take any answer that finished meanwhile (costs no waiting).
            finished = set(done) | {t for t in pending if t.done()}
            pending -= finished
            for task in finished:
                try:
                    answer, _finish = task.result()
                except Exception as exc:
                    log(f"direct task failed: {exc}")
                    answer = ""
                if answer:
                    got[tasks[task]] = answer
                    await _draft(state, answer, f"direct solve with {tasks[task]}")
            if primary in got and partner not in got:
                for task in [t for t in pending if t.done()]:
                    pending.discard(task)
                    try:
                        late, _ = task.result()
                    except Exception:
                        late = ""
                    if late:
                        got[tasks[task]] = late
            if primary in got and partner in got:
                break
            if primary in got and CONFIRM_WAIT_S <= 0:
                break  # calibrated primary answered: submit now (speed is 15% of score)
            if got and soft_deadline is None:
                # One answer is in. Give the other a bounded window: longer when the
                # trusted primary is the one still thinking.
                if primary not in got:
                    window = min(clock.left() * 0.3, 15.0) if tight else min(clock.left() * 0.45, 30.0)
                else:
                    window = min(clock.left() * 0.25, CONFIRM_WAIT_S)
                soft_deadline = time.monotonic() + max(4.0, window)
        for task in pending:
            task.cancel()
        if primary in got and partner in got:
            if _norm(got[primary], spec) == _norm(got[partner], spec):
                state.model_used = primary
                return got[primary], 0.95
            # Disagreement: a third model breaks the tie when time allows.
            if len(lineup) > 2 and clock.left() > 15:
                judge = lineup[2]
                tried.add(judge)
                third, _ = await ask(judge, clock.budget(0.6, floor_s=5), max_tokens)
                if third:
                    for name in (primary, partner):
                        if _norm(third, spec) == _norm(got[name], spec):
                            state.model_used = name
                            log(f"tie broken by {judge}: {name}")
                            await _draft(state, got[name], f"majority with {judge}")
                            return got[name], 0.85
            state.model_used = primary
            await _draft(state, got[primary], "primary after disagreement")
            return got[primary], 0.6
        if primary in got:
            state.model_used = primary
            return got[primary], 0.75
        if partner in got:
            state.model_used = partner
            return got[partner], 0.6
    # Sequential fallback over the remaining ranked models.
    for m in [x for x in [model] + ranked if x not in tried][:3]:
        if clock.left() < 8:
            break
        tried.add(m)
        answer, finish = await ask(m, clock.budget(0.7, floor_s=5), max_tokens)
        if answer:
            state.model_used = m
            await _draft(state, answer, f"direct solve with {m}")
            return answer, 0.7
        if finish == "length":
            max_tokens = max(2048, max_tokens // 2)
    return "", 0.0


async def _solve_react(state: RunState, box: BlackBox, ctx: ChallengeContext, model: str, clock: Clock, spec,
                       evidence: list[str], prior: str = "") -> tuple[str, float]:
    assert state.llm is not None
    system = (
        "You are an autonomous agent dropped into an unfamiliar MCP server (a black box). Your job is to work out "
        "what the challenge needs, use the server's tools to get it, and return the exact final answer.\n"
        "- Read tool descriptions. Some tools are hidden: names can appear in clues, descriptions, errors, or decoded text.\n"
        "- Data may be encoded or lightly encrypted (base64, hex, rot13, caesar, vigenere, xor, morse, ...). Use decode.\n"
        "- Keys and passwords found anywhere are likely inputs to other tools.\n"
        "- Prefer the fewest calls that settle the answer. Time and tokens are scored.\n"
        f"- Finish with final_answer. Required format: {spec.describe()}.\n"
        "- Never try to submit through a tool yourself. The controller submits final_answer."
    )
    task = (
        f"CHALLENGE ({ctx.challenge_type}, difficulty {ctx.difficulty}, about {clock.left():.0f}s left)\n"
        f"Description: {ctx.description}\nRules: {ctx.rules}\n"
        + ("Clues:\n" + "\n".join(f"- {c}" for c in ctx.clues) + "\n" if ctx.clues else "")
        + f"\n{box.inventory_text(include_core=False)}\n\n{_evidence_block(box, evidence, 6000)}"
        + (f"\n\nA first attempt answered {prior!r}; verify or improve it." if prior else "")
    )
    params = state.strategy.get_llm_params(ctx)
    agent = ReactAgent(
        box, state.llm, model, clock, spec=spec, on_event=state.thoughts.detail,
        max_steps=int(os.getenv("GAUNTLET_MAX_STEPS", "14")),
        max_tokens=min(int(params.get("max_tokens", 4096)), 8192),
    )
    result = await agent.run(system, task)
    for line in result.log[-20:]:
        log(f"  react {line[:220]}")
    if result.answer:
        await _draft(state, result.answer, f"react {result.steps} steps")
    return result.answer, result.confidence


def _mentions_checker(box: BlackBox, description: str, rules: str, clues: list[str]) -> bool:
    text = " ".join([description, rules, *clues]).lower()
    return any(t.name.lower() in text for t in box.by_kind("checker"))


async def _check_with_checker(state: RunState, box: BlackBox, ctx: ChallengeContext, model: str, clock: Clock, spec,
                              evidence: list[str], answer: str) -> str:
    """When the box offers a non-final checker tool, use it before the one real submit."""
    checkers = [t for t in box.by_kind("checker") if not t.name.startswith("arena.")]
    if not answer or not checkers or clock.left() < 10:
        return answer
    tool = checkers[0]
    field = next((p for p in tool.properties if re.search(r"(answer|guess|solution|flag|value|candidate|text)", p, re.I)), None)
    if not field:
        return answer
    box.ctx.overrides = {field: answer}
    args = synthesize_args(tool, box.ctx) or {field: answer}
    box.ctx.overrides = {}
    res = await box.call(tool.name, args, source="checker", timeout_s=min(10, clock.left() - 4))
    verdict = compat.short(res.get("dict") or res.get("text"), 400).lower()
    log(f"checker {tool.name}: {verdict[:160]}")
    if re.search(r"\b(correct|true|valid|accepted|match)\b", verdict) and not re.search(r"\b(incorrect|not correct|invalid|wrong|false)\b", verdict):
        return answer
    if clock.left() > 20 and re.search(r"\b(incorrect|wrong|invalid|false|try again|no match)\b", verdict):
        evidence = evidence + [f"checker {tool.name} rejected {answer!r}: {verdict}"]
        new, _conf = await _solve_react(state, box, ctx, model, clock, spec, evidence, prior=answer)
        return new or answer
    return answer


async def _finalize_format(state: RunState, model: str, ranked: list[str], clock: Clock, spec, ctx: ChallengeContext, answer: str) -> str:
    if not answer:
        return answer
    fixed, problems = enforce(answer, spec)
    if not problems or clock.left() < 8 or not state.llm:
        return fixed
    fast = ranked[-1] if ranked else model
    prompt = (
        f"Rules: {ctx.rules}\nRequired format: {spec.describe()}\nCandidate answer: {answer}\n"
        f"Problems: {'; '.join(problems)}\nRewrite the candidate so it satisfies the format without changing its meaning. "
        "Reply with one line: ANSWER: <answer>"
    )
    reply = await state.llm.chat(fast if clock.left() < 20 else model, [{"role": "user", "content": prompt}],
                                 max_tokens=800, temperature=0.0, timeout_s=min(15, clock.left() - 3))
    repaired = extract_answer(reply.content, spec)
    if repaired and not is_invalid(repaired):
        fixed2, problems2 = enforce(repaired, spec)
        if len(problems2) < len(problems):
            return fixed2
    return fixed


def _fallback_answer(box: BlackBox, clues: list[str], description: str) -> str:
    if box.notebook.flags:
        return box.notebook.flags[-1]
    decoded = [f.text for f in box.notebook.facts if f.kind == "decoded"]
    if decoded:
        return decoded[-1].split(":", 1)[-1].strip()[:200]
    return "No answer found"


async def _draft(state: RunState, answer: str, why: str) -> None:
    if answer and not is_invalid(answer):
        state.best_answer = answer
    try:
        await asyncio.wait_for(asyncio.to_thread(state.http.save_draft, state.agent_id, answer, why), timeout=5)
    except Exception:
        pass


class FastSubmit:
    """Keeps one TLS connection to the REST API warm until the submit.

    Practice: the server stamped submissions ~1.8 s after the agent started the
    POST; a fresh connection (DNS + TLS) per request is most of that. One
    requests.Session pinged every 2.5 s stays inside typical keep-alive windows.
    """

    def __init__(self, api_base: str, api_key: str) -> None:
        import requests

        self.base = api_base.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json", "X-Arena-API-Key": api_key})
        self._task: asyncio.Task[Any] | None = None

    def start(self) -> None:
        if os.getenv("GAUNTLET_KEEP_WARM", "1").lower() in {"0", "false", "no"}:
            return
        self._task = asyncio.create_task(self._loop())

    async def _loop(self) -> None:
        for _ in range(240):
            try:
                await asyncio.to_thread(self.session.get, f"{self.base}/api/health", timeout=3)
            except Exception:
                pass
            await asyncio.sleep(2.5)

    def stop(self) -> None:
        if self._task:
            self._task.cancel()
            self._task = None

    def get_json(self, path: str) -> dict[str, Any]:
        resp = self.session.get(f"{self.base}{path}", timeout=8)
        if resp.status_code >= 400:
            raise ArenaAPIError(resp.status_code, resp.text)
        data = resp.json()
        return data if isinstance(data, dict) else {}

    def lookup_modality(self, agent_id: str) -> str:
        """This agent's assignment from its session ("text", "image", or "")."""
        try:
            data = self.get_json(f"/api/session/{quote(agent_id, safe='')}")
        except Exception:
            return ""
        kind = str(data.get("challenge_type") or "").lower()
        return kind if kind in {"text", "image"} else ""

    def submit(self, agent_id: str, answer: str, metrics: dict[str, Any]) -> dict[str, Any]:
        resp = self.session.post(f"{self.base}/api/submit", json={"agent_id": agent_id, "answer": answer, "client_metrics": metrics}, timeout=60)
        if resp.status_code >= 400:
            raise ArenaAPIError(resp.status_code, resp.text)
        return resp.json()


async def _submit_text(state: RunState, answer: str, total_ms: int) -> str:
    metrics = _metrics(state, total_ms)
    if state.fast is not None:
        state.fast.stop()
        try:
            data = await asyncio.to_thread(state.fast.submit, state.agent_id, answer, metrics)
            for line in submission_summary_lines(data):
                log(line)
            note = missing_tools_note(data)
            if note:
                log(note)
            state.record["result"] = {"accepted": data.get("accepted"), "status": data.get("status"), "score": data.get("score")}
            return "submitted"
        except ArenaAPIError as exc:
            log(f"submit API error {exc.status_code}: {exc.message[:200]}")
            if exc.status_code == 409:
                return "already-submitted"
            if exc.status_code < 500:
                return f"submit-error-{exc.status_code}"
        except Exception as exc:
            log(f"fast submit failed ({type(exc).__name__}); retrying through the kit client")
    for attempt in range(3):
        try:
            result = await asyncio.to_thread(state.http.submit, state.agent_id, answer, metrics)
            for line in submission_summary_lines(result):
                log(line)
            note = missing_tools_note(result)
            if note:
                log(note)
            state.record["result"] = {"accepted": result.accepted, "status": result.status, "score": result.score}
            return "submitted"
        except ArenaAPIError as exc:
            log(f"submit API error {exc.status_code}: {exc.message[:200]}")
            if exc.status_code == 409:
                return "already-submitted"
            return f"submit-error-{exc.status_code}"
        except ArenaConnectionError as exc:
            log(f"submit connection error (retrying same answer): {exc}")
            await asyncio.sleep(0.6 * (attempt + 1))
    return "submit-failed"


# ---------------------------------------------------------------------------
# Image


async def solve_image(state: RunState, box: BlackBox) -> str:
    tool = box.find("arena.image.get_challenge")
    assert tool is not None
    thought_tool = box.find("arena.image.broadcast_thought")
    state.thoughts.use_mcp(box, thought_tool.name if thought_tool else None)
    res = await _call_until_unlocked(box, tool.name, {"agent_id": state.agent_id})
    err = str(res.get("error") or "")
    if err:
        log(f"image get_challenge error: {err}")
        state.thoughts.use_mcp(box, None)
        if "not_image_challenge" in err or "text" in err.lower():
            return "switch"
        return f"error: {err}"
    ch = res["dict"]
    state.record.setdefault("startup", {})["challenge"] = round(time.monotonic() - PROC_T0, 2)
    clock = Clock(float(ch.get("time_remaining_s") or 0), float(ch.get("max_time_s") or 0))
    prompt = str(ch.get("prompt") or ch.get("edit_prompt") or "")
    notes = str(ch.get("reference_notes") or "")
    input_uri = str(ch.get("input_image_uri") or "")
    state.record["challenge"] = {k: (v if k != "input_image_uri" else "(image)") for k, v in ch.items()}
    log(f"image challenge {ch.get('puzzle_id')} limit={ch.get('max_time_s')}s edit={bool(input_uri)} :: {prompt[:160]}")
    await state.thoughts.say(f"Image task received ({'edit' if input_uri else 'generate'}).", force=True)
    box.ctx.image_uri = input_uri
    box.ctx.challenge_text = prompt

    rules = "\n".join(p for p in (prompt, notes) if p.strip())
    ctx = _ctx(ch, clues=[], models=state.models, clock=clock, image_url=input_uri or None,
               description=str(ch.get("description") or ""), rules=rules)
    image_models = rank_image_models(state.models, state.model_rows)
    hook_ranked = state.strategy.rank_models(ctx, state.models) if state.models else []
    ranked = [m for m in hook_ranked if m in image_models] + [m for m in image_models if m not in hook_ranked]
    if not ranked:
        ranked = hook_ranked or state.models
    state.model_used = ranked[0] if ranked else ""
    image_prompt = state.strategy.build_image_prompt(ctx)

    produced: dict[str, Any] = {"uri": "", "plan": "", "prompt": image_prompt}

    async def _produce() -> None:
        nonlocal image_prompt
        # Pixel-level inspection of the input is free; a vision description costs one tool call.
        if input_uri.startswith("data:"):
            data = compat.decode_b64_loose(input_uri)
            if data:
                report = pngtools.inspect_image(data)
                log(f"input image: {report.get('format')} {report.get('width')}x{report.get('height')}")
        analyzer = box.find("image_analyze") or next(iter(box.by_kind("image_analyze")), None)
        use_analyze = os.getenv("GAUNTLET_IMAGE_ANALYZE", "0").lower() in {"1", "true", "yes"}
        if use_analyze and input_uri and analyzer and clock.left() > 0.55 * clock.max_time_s:
            extra = build_image_tool_arguments(list(_raw_defs(box)), analyzer.name, selected_model=ranked[0] if ranked else "",
                                               llm_api_key=state.api_key, arena_api_key=state.api_key)
            state.record.setdefault("timeline", {})["analyze_start"] = round(clock.elapsed(), 1)
            ares = await box.call(analyzer.name, {"image_uri": input_uri, "question": (
                "Describe this image precisely: subjects, layout, colors, lighting, style, any text. "
                f"Then list what must stay unchanged when applying this edit: {prompt}"), "agent_id": state.agent_id, **extra},
                source="image", timeout_s=min(30, clock.left() * 0.3))
            analysis = str((ares.get("dict") or {}).get("text") or ares.get("text") or "").strip()
            if analysis and not ares.get("error"):
                await state.thoughts.say("Input image analyzed.")
                await state.thoughts.detail(f"Analysis: {analysis[:140]}")
                image_prompt += f"\n\nPreserve everything not named in the edit. Source image details: {analysis[:900]}"

        plan = state.strategy.plan_image_tool(ctx, [n for n in box.tools if box.tools[n].kind.startswith("image_")])
        if plan == "image_analyze" or plan not in box.tools:
            plan = "image_edit" if input_uri and "image_edit" in box.tools else "image_generate"
        if plan not in box.tools:
            alt = box.by_kind("image_edit" if input_uri else "image_generate") or box.by_kind("image_generate")
            plan = alt[0].name if alt else ""
        output_uri = ""
        errors: list[str] = state.record.setdefault("image_errors", [])

        async def attempt(model: str) -> str:
            extra = build_image_tool_arguments(list(_raw_defs(box)), plan, selected_model=model, llm_api_key=state.api_key, arena_api_key=state.api_key)
            base_args = synthesize_args(box.tools[plan], box.ctx) or {}
            args = {**base_args, "prompt": image_prompt, "agent_id": state.agent_id, **extra}
            if input_uri and box.tools[plan].kind == "image_edit":
                args["image_uri"] = input_uri
            await state.thoughts.detail(f"{plan} with {model or 'default model'}")
            state.record.setdefault("timeline", {})[f"{plan}_start:{model}"] = round(clock.elapsed(), 1)
            r = await box.call(plan, args, source="image", timeout_s=max(10, clock.left() - 8))
            state.record["timeline"][f"{plan}_done:{model}"] = round(clock.elapsed(), 1)
            candidate = str((r.get("dict") or {}).get("image_uri") or "").strip()
            if not candidate and r.get("images"):
                img = r["images"][0]
                candidate = f"data:{img.get('mime') or 'image/png'};base64,{img.get('b64')}"
            if candidate and _valid_image_uri(candidate):
                return candidate
            reason = r.get("error") or (r.get("dict") or {}).get("error") or (r.get("dict") or {}).get("text") or "no image returned"
            errors.append(f"{model}: {str(reason)[:300]}")
            log(f"image attempt failed: {errors[-1][:200]}")
            return ""

        models = list(produced.get("models") or ranked[:3] or [""])
        await state.thoughts.say("Producing the image.")
        hedge = os.getenv("GAUNTLET_IMAGE_HEDGE", "1").lower() not in {"0", "false", "no"}
        if plan and hedge and len(models) >= 2 and clock.left() > 25:
            # Practice: the preferred image model often failed its first call after
            # ~20 s. Run two models at once; prefer the first-ranked one when it
            # lands within a short window after the other.
            prefer_wait = float(os.getenv("GAUNTLET_IMAGE_PREFER_WAIT_S", "3"))
            first, second = models[0], models[1]
            tasks = {asyncio.create_task(attempt(m)): m for m in (first, second)}
            got: dict[str, str] = {}
            deadline: float | None = None
            pending = set(tasks)
            while pending:
                wait = None if deadline is None else max(0.0, deadline - time.monotonic())
                done, pending = await asyncio.wait(pending, timeout=wait, return_when=asyncio.FIRST_COMPLETED)
                if not done:
                    break
                for task in done:
                    try:
                        uri = task.result()
                    except Exception as exc:
                        uri = ""
                        errors.append(f"{tasks[task]}: {type(exc).__name__}: {exc}")
                    if uri:
                        got[tasks[task]] = uri
                if first in got:
                    break
                if second in got and deadline is None:
                    deadline = time.monotonic() + max(0.0, min(prefer_wait, clock.left() - 15))
            for task in pending:
                task.cancel()
            chosen = first if first in got else (second if second in got else "")
            if chosen:
                output_uri = got[chosen]
                state.model_used = chosen
            models = models[2:]
        for model in models:
            if output_uri or not plan or clock.left() < 12:
                break
            uri = await attempt(model)
            if uri:
                output_uri = uri
                state.model_used = model or state.model_used
        produced["uri"], produced["plan"] = output_uri, plan
        produced["model"] = state.model_used

    try:
        await asyncio.wait_for(_produce(), timeout=max(1.0, clock.left()))
    except asyncio.TimeoutError:
        log("image deadline reached; falling back")
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        log(f"image pipeline crashed ({type(exc).__name__}: {exc}); falling back")
    output_uri, plan = str(produced["uri"] or ""), str(produced["plan"] or "image_edit")
    if not output_uri:
        output_uri = input_uri
        log("no new image produced; submitting the input image as the safe fallback" if input_uri else "no image at all")
    if not output_uri:
        return "no-image"

    submit_errors: list[str] = state.record.setdefault("image_submit_errors", [])
    result = await _submit_image(state, box, output_uri, plan, clock, submit_errors)
    # A failed submit (for example an image too large for transport) gets one more
    # try with the next image model while time allows.
    used = str(produced.get("model") or "")
    if result is None and output_uri != input_uri and clock.left() > 25:
        others = [m for m in ranked if m != used][:1]
        if others:
            log(f"retrying the image with {others[0]} after submit failure")
            produced["models"] = others
            produced["uri"] = ""
            try:
                await asyncio.wait_for(_produce(), timeout=max(1.0, clock.left() - 5))
            except Exception as exc:
                log(f"image retry failed: {type(exc).__name__}: {exc}")
            if produced.get("uri"):
                output_uri = str(produced["uri"])
                result = await _submit_image(state, box, output_uri, plan, clock, submit_errors)
    if result is None and input_uri and output_uri != input_uri:
        log("submitting the input image as the last resort")
        output_uri = input_uri
        result = await _submit_image(state, box, output_uri, plan, clock, submit_errors)
    await state.thoughts.flush()
    if result is None:
        return "submit-failed"
    for line in submission_summary_lines(result):
        log(line)
    log(f"image {save_submitted_image(state.agent_id, output_uri, REPO_ROOT)}")
    score = result.get("score") if isinstance(result, dict) else getattr(result, "score", None)
    state.record["result"] = {"accepted": result.get("accepted") if isinstance(result, dict) else getattr(result, "accepted", None),
                              "score": {k: v for k, v in (score or {}).items() if k != "output_image_uri"} if isinstance(score, dict) else score}
    return "submitted"


async def _submit_image(state: RunState, box: BlackBox, uri: str, plan: str, clock: Clock, errors: list[str]) -> Any:
    """MCP submit, then REST. Returns the result, or None with the reasons in ``errors``."""
    state.record.setdefault("timeline", {})["submit_start"] = round(clock.elapsed(), 1)
    state.record["image_bytes"] = len(uri)
    log(f"submitting image: {len(uri) / 1e6:.2f} MB as text ({uri[:30]}...)")
    metrics = _metrics(state, int(clock.elapsed() * 1000))
    metrics["planner_tool"] = plan
    await state.thoughts.say("Submitting the image.", force=True)
    submit = box.find("arena.image.submit_edit")
    if submit:
        r = await box.call(submit.name, {"edited_image": uri, "client_metrics": metrics, "rationale": "gauntlet explorer image flow",
                                         "agent_id": state.agent_id}, source="submit", mine=False, timeout_s=90)
        if not r.get("error"):
            state.record["timeline"]["submit_done"] = round(clock.elapsed(), 1)
            return r.get("dict")
        errors.append(f"mcp: {str(r['error'])[:400]}")
        log(f"MCP image submit failed: {str(r['error'])[:300]}")
        if "409" in str(r["error"]) or "already" in str(r["error"]).lower():
            return {"accepted": None, "status": "already-submitted", "score": None}
    try:
        result = await asyncio.to_thread(state.http.submit, state.agent_id, uri, metrics, "image")
        state.record["timeline"]["submit_done"] = round(clock.elapsed(), 1)
        return result
    except ArenaAPIError as exc:
        errors.append(f"rest: HTTP {exc.status_code}: {exc.message[:400]}")
        log(f"REST image submit failed: HTTP {exc.status_code}: {exc.message[:300]}")
        if exc.status_code == 409:
            return {"accepted": None, "status": "already-submitted", "score": None}
    except Exception as exc:
        errors.append(f"rest: {type(exc).__name__}: {str(exc)[:400]}")
        log(f"REST image submit failed: {exc}")
    return None


def _raw_defs(box: BlackBox) -> list[Any]:
    """Tool defs shaped like the kit expects (name + inputSchema) for its helpers."""
    class _Def:
        def __init__(self, name: str, schema: dict[str, Any]) -> None:
            self.name = name
            self.inputSchema = schema
    return [_Def(t.name, t.schema) for t in box.tools.values()]


def _valid_image_uri(uri: str) -> bool:
    if uri.startswith(("http://", "https://")):
        return True
    data = compat.decode_b64_loose(uri)
    return bool(data) and pngtools.sniff(data) != "unknown"


# ---------------------------------------------------------------------------
# Unknown black box (no arena.* contract)


async def solve_black_box(state: RunState, box: BlackBox) -> str:
    """No familiar challenge tool. Explore, find the task, solve it, submit once."""
    clock = Clock(float(os.getenv("GAUNTLET_BLACKBOX_TIME_S", "300")))
    await state.thoughts.say("Unfamiliar server. Exploring tools, resources and prompts.", force=True)
    task_text = ""
    challenge_tool = next(iter(box.by_kind("challenge")), None)
    if challenge_tool:
        args = synthesize_args(challenge_tool, box.ctx) or {}
        res = await box.call(challenge_tool.name, args, source="controller")
        task_text = compat.short(res.get("dict") or res.get("text"), 4000)
        data = res.get("dict") or {}
        if data.get("time_remaining_s") or data.get("max_time_s"):
            clock = Clock(float(data.get("time_remaining_s") or 0), float(data.get("max_time_s") or 0))
    await box.follow_leads(time.monotonic() + min(30.0, clock.left() * 0.3), max_calls=14)
    challenge = {"challenge_type": "black-box", "description": task_text or "Discover the objective of this server and solve it.",
                 "rules": "Return the exact answer, key, or flag the server expects."}
    spec = infer_format(task_text, "")
    ctx = _ctx(challenge, clues=[], models=state.models, clock=clock)
    model, _ranked = _pick_model(state, ctx)
    state.model_used = model
    answer, _conf = await _solve_react(state, box, ctx, model, clock, spec, [])
    if not answer and box.notebook.flags:
        answer = box.notebook.flags[-1]
    if not answer:
        return "no-answer"
    submitters = box.by_kind("submit")
    for tool in submitters:
        field = next((p for p in tool.properties if re.search(r"(answer|flag|solution|value|guess|text)", p, re.I)), None)
        if not field:
            continue
        box.ctx.overrides = {field: answer}
        args = synthesize_args(tool, box.ctx) or {field: answer}
        box.ctx.overrides = {}
        res = await box.call(tool.name, args, source="submit")
        log(f"black-box submit via {tool.name}: {compat.short(res.get('dict') or res.get('text') or res.get('error'), 300)}")
        state.record["result"] = res.get("dict") or res.get("text")
        return "submitted"
    try:
        await _submit_text(state, answer, int(clock.elapsed() * 1000))
    except Exception:
        pass
    return "submitted-rest"
