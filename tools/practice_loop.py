#!/usr/bin/env python3
"""Run the explorer several times on Practice and print one score line per run.

Each registration gets the next Practice puzzle (the rotation is shared by all
teams, so runs may skip ahead). Five runs usually cover text and image.

    python tools/practice_loop.py            # 5 runs
    python tools/practice_loop.py --runs 8

Every run record is kept as outputs/<AGENT_ID>-run-<n>.json for later study.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--pause", type=float, default=2.0, help="seconds between runs")
    args = parser.parse_args()
    try:
        from dotenv import dotenv_values

        env_file = {k: v for k, v in dotenv_values(ROOT / ".env").items() if v}
    except ImportError:
        env_file = {}
    env = {**env_file, **os.environ}
    agent_id = env.get("AGENT_ID", "")
    if not agent_id:
        print("Set AGENT_ID in .env first.", file=sys.stderr)
        return 2
    record = ROOT / "outputs" / f"{agent_id}-run.json"
    print(f"{'#':>2}  {'puzzle':36s} {'ok':5s} {'score':>6s} {'qual':>5s} {'speed':>5s} {'tools':>5s} {'tok':>5s} {'secs':>5s}  notes")
    totals = []
    for n in range(1, args.runs + 1):
        started = time.monotonic()
        proc = subprocess.run([sys.executable, str(ROOT / "examples" / "explorer" / "agent.py")], env=env,
                              capture_output=True, text=True, timeout=600)
        secs = time.monotonic() - started
        data = json.loads(record.read_text()) if record.exists() else {}
        if record.exists():
            shutil.copy(record, ROOT / "outputs" / f"{agent_id}-run-{n}.json")
        (ROOT / "outputs" / f"{agent_id}-run-{n}.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
        challenge = data.get("challenge") or {}
        puzzle = str(challenge.get("puzzle_id") or challenge.get("challenge_type") or "?")[:36]
        result = data.get("result") or {}
        score = result.get("score") or {}
        notes = []
        if score.get("missing_required_tools"):
            notes.append("missing " + ",".join(score["missing_required_tools"]))
        if score.get("broadcast_penalty_applied"):
            notes.append("thought penalty")
        if score.get("judge_error"):
            notes.append("judge error")
        if proc.returncode:
            notes.append(f"exit {proc.returncode}")
        final = score.get("final_score")
        if isinstance(final, (int, float)):
            totals.append(final)
        print(f"{n:>2}  {puzzle:36s} {str(result.get('accepted')):5s} {final if final is not None else '-':>6} "
              f"{score.get('quality_score', '-'):>5} {score.get('speed_score', '-'):>5} {score.get('tools_score', '-'):>5} "
              f"{score.get('tokens_score', '-'):>5} {secs:5.0f}  {'; '.join(notes)}")
        if proc.returncode and not data:
            print(proc.stdout[-1500:], proc.stderr[-1500:])
        time.sleep(args.pause)
    if totals:
        print(f"\naverage final score over {len(totals)} scored runs: {sum(totals) / len(totals):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
