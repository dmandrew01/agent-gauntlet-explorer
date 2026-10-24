#!/usr/bin/env python3
"""Run the explorer against every mock arena scenario and report pass/fail.

    python tools/run_mock_suite.py                 # all scenarios
    python tools/run_mock_suite.py --only blackbox
    python tools/run_mock_suite.py --deadline      # adds a slow-LLM run that must still submit

Needs: mcp (1.x or 2.x), starlette, uvicorn, requests in the interpreter.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent
# Practice rotation shape: logic easy/medium/hard, image edit, image generate,
# plus black-box cases the live event may add.
SCENARIOS = ["text_logic", "text_order", "text_encoded", "blackbox", "image_edit", "image_generate"]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_mock(port: int, env_extra: dict[str, str]) -> subprocess.Popen:
    env = {**os.environ, **env_extra}
    proc = subprocess.Popen([sys.executable, str(ROOT / "mock_arena" / "server.py"), "--port", str(port)],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for _ in range(50):
        try:
            urlopen(f"http://127.0.0.1:{port}/api/health", timeout=0.5)
            return proc
        except Exception:
            time.sleep(0.2)
    proc.kill()
    raise RuntimeError("mock arena did not start")


def run_agent(port: int, agent_id: str, env_extra: dict[str, str], timeout: float) -> tuple[dict, float, str]:
    base = f"http://127.0.0.1:{port}"
    env = {**os.environ, "ARENA_API_BASE": base, "ARENA_MCP_URL": base, "LLM_PROXY_HOST": f"{base}/proxy",
           "ARENA_API_KEY": "practice-key", "AGENT_ID": agent_id, "AGENT_NAME": agent_id, **env_extra}
    env.pop("ARENA_SERVER", None)
    started = time.monotonic()
    proc = subprocess.run([sys.executable, str(ROOT / "examples" / "explorer" / "agent.py")], env=env,
                          capture_output=True, text=True, timeout=timeout)
    elapsed = time.monotonic() - started
    record_path = ROOT / "outputs" / f"{agent_id}-run.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    return record, elapsed, proc.stdout[-3000:] + proc.stderr[-2000:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="")
    parser.add_argument("--deadline", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    plan = [(s, {}, {}) for s in (args.only.split(",") if args.only else SCENARIOS)]
    if args.deadline:
        plan.append(("text_logic", {"MOCK_LLM_DELAY_S": "40", "MOCK_LOGIC_TIME_S": "30"}, {}))
        plan.append(("blackbox", {}, {"GAUNTLET_PROBE": "0", "GAUNTLET_MODEL_PREFERENCE": "mock/fast-mini"}))
    failures = 0
    for i, (scenario, mock_env, agent_env) in enumerate(plan):
        port = free_port()
        mock = start_mock(port, {"MOCK_SCENARIO": scenario, **mock_env})
        agent_id = f"suite-{scenario.replace('_', '-')}-{i}"
        try:
            record, elapsed, out = run_agent(port, agent_id, agent_env, timeout=200)
        finally:
            mock.terminate()
        result = record.get("result") or {}
        accepted = bool(result.get("accepted"))
        submitted = record.get("outcome") in {"submitted", "already-submitted"}
        expect_accept = "MOCK_LLM_DELAY_S" not in mock_env
        ok = submitted and (accepted or not expect_accept)
        failures += 0 if ok else 1
        score = (result.get("score") or {}).get("final_score")
        label = scenario + (" (slow LLM, must still submit)" if not expect_accept else "") + (" (JSON protocol)" if agent_env else "")
        print(f"{'PASS' if ok else 'FAIL'}  {label:48s} accepted={accepted} score={score} "
              f"mode={record.get('mode', '-')} tools={len(record.get('tools_used', []))} thoughts={record.get('thoughts')} {elapsed:.1f}s")
        if not ok or args.verbose:
            print(out)
    print(f"\n{len(plan) - failures}/{len(plan)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
