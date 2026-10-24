#!/usr/bin/env python3
"""Measure every text model on the proxy roster and save gauntlet/model_stats.json.

Run this against Practice before the event (tokens there do not count):
    python tools/calibrate_models.py            # all text models, 6 probes each
    python tools/calibrate_models.py --probes 3 --timeout 60

The ranker in gauntlet/models.py blends the measured median latency and
accuracy into its built-in priors. Package the JSON with the agent:
    python tools/package_explorer.py ... (it includes gauntlet/model_stats.json when present)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from arena_clients import get_arena_api_key, get_proxy_host  # noqa: E402
from gauntlet.answer import extract_answer  # noqa: E402
from gauntlet.llm import LLM  # noqa: E402
from gauntlet.models import is_image_model  # noqa: E402
from model_selector import fetch_available_model_rows  # noqa: E402

PROBES = [
    ("Five houses in a row are painted red, blue, green, white, yellow from left to right. "
     "Which color is in the middle? Answer with one lowercase word.", "green"),
    ("If 3 workers build 3 walls in 3 hours, how many hours do 6 workers need to build 6 walls? Answer with an integer.", "3"),
    ("Decode this base64 and answer with the decoded word in uppercase: T1JJT04=", "ORION"),
    ("Alice is taller than Bob. Bob is taller than Cara. Cara is taller than Dan. Who is the second shortest? One name.", "Cara"),
    ("Return a JSON object with keys \"a\" and \"b\" where a is 2+2 and b is the word \"ok\". No spaces.", '{"a":4,"b":"ok"}'),
    ("A clock shows 3:15. What is the angle in degrees between the hour and minute hands? Answer with a number.", "7.5"),
]


def correct(answer: str, expected: str) -> bool:
    a = answer.strip().strip(".").replace(" ", "")
    e = expected.replace(" ", "")
    if e.startswith("{"):
        try:
            return json.loads(a) == json.loads(e)
        except ValueError:
            return False
    return a.lower() == e.lower()


async def calibrate(model: str, llm: LLM, probes: int, timeout: float) -> dict:
    latencies, hits, errors, tokens = [], 0, 0, 0
    for question, expected in PROBES[:probes]:
        reply = await llm.chat(model, [
            {"role": "system", "content": "Answer the question. First line: ANSWER: <answer>."},
            {"role": "user", "content": question},
        ], max_tokens=4096, temperature=0.0, timeout_s=timeout)
        if reply.error:
            errors += 1
            continue
        latencies.append(reply.elapsed_ms / 1000)
        tokens += reply.prompt_tokens + reply.completion_tokens
        if correct(extract_answer(reply.content), expected):
            hits += 1
    trials = probes
    return {
        "trials": trials,
        "errors": errors,
        "accuracy": round(hits / max(1, trials - errors), 3) if trials > errors else 0.0,
        "median_latency_s": round(statistics.median(latencies), 2) if latencies else None,
        "avg_tokens": round(tokens / max(1, trials - errors)),
    }


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probes", type=int, default=len(PROBES))
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--out", default=str(ROOT / "gauntlet" / "model_stats.json"))
    parser.add_argument("--only", default="", help="comma-separated model ids")
    args = parser.parse_args()
    proxy, key = get_proxy_host(), get_arena_api_key()
    rows = fetch_available_model_rows(proxy, key)
    models = [r["id"] for r in rows if not is_image_model(r["id"], rows)]
    if args.only:
        models = [m for m in models if m in set(args.only.split(","))]
    agent = (os.getenv("AGENT_ID") or "explorer") + "-calibration"
    llm = LLM(proxy, key, agent)
    results = {}
    for model in models:
        started = time.monotonic()
        results[model] = await calibrate(model, llm, args.probes, args.timeout)
        print(f"{model:45s} {results[model]}  ({time.monotonic() - started:.0f}s)", flush=True)
    payload = {"measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "proxy": proxy, "models": results}
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
