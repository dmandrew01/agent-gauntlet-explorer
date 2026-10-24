#!/usr/bin/env python3
"""Print a compact report of one or more Practice runs, ready to paste.

    python tools/show_run.py 3          # outputs/<AGENT_ID>-run-3.json
    python tools/show_run.py 1 2 3 4 5
    python tools/show_run.py            # latest outputs/<AGENT_ID>-run.json
    python tools/show_run.py 2 --inventory   # also print the server's tool list
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def agent_id() -> str:
    if os.getenv("AGENT_ID"):
        return os.environ["AGENT_ID"]
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("AGENT_ID="):
                return line.split("=", 1)[1].strip().strip('"')
    return ""


def show(path: Path) -> None:
    r = json.loads(path.read_text())
    ch = r.get("challenge") or {}
    res = r.get("result") or {}
    score = res.get("score") or {}
    print(f"=== {path.name}")
    print(f"puzzle={ch.get('puzzle_id')} type={ch.get('challenge_type')} difficulty={ch.get('difficulty')} limit={ch.get('max_time_s')}s")
    for key in ("description", "rules", "prompt", "reference_notes"):
        if ch.get(key):
            print(f"{key}: {str(ch[key])[:400]}")
    print(f"mode={r.get('mode', '-')} model={r.get('model')} answer={str(r.get('answer', ''))[:200]!r}")
    print(f"accepted={res.get('accepted')} final={score.get('final_score')} quality={score.get('quality_score')} "
          f"speed={score.get('speed_score')} tools={score.get('tools_score')} tokens={score.get('tokens_score')}")
    extra = {k: v for k, v in score.items() if k not in {"final_score", "quality_score", "speed_score", "tools_score",
                                                          "tokens_score", "models_score", "output_image_uri"}}
    if extra:
        print("score detail:", json.dumps(extra)[:600])
    for a in r.get("llm_attempts", []):
        print(f"  llm t={a.get('t')}s {a.get('model')} {a.get('ms')}ms finish={a.get('finish')} "
              f"tokens={a.get('tokens')} err={a.get('error') or '-'} answer={a.get('answer')!r}")
    if r.get("startup"):
        print("startup (s from process start):", json.dumps(r["startup"]))
    if r.get("timeline"):
        print("timeline (s from challenge):", json.dumps(r["timeline"]))
    for c in r.get("calls", [])[:30]:
        line = f"  tool {c.get('name')} {c.get('elapsed_ms')}ms ok={c.get('ok')}"
        if not c.get("ok"):
            line += f" args={json.dumps(c.get('args'))[:160]} -> {str(c.get('summary'))[:300]}"
        print(line)
    if r.get("catalog") is not None:
        print("catalog (arena.tools.list etc.):", json.dumps(r.get("catalog"))[:1500])
        print("catalog revealed:", r.get("catalog_revealed"))
    if r.get("image_bytes"):
        print(f"submitted image size: {r['image_bytes'] / 1e6:.2f} MB as text")
    for err in r.get("image_submit_errors", []):
        print(f"  image submit error: {err[:400]}")
    if "--inventory" in sys.argv:
        print(r.get("inventory", ""))
    print(f"thoughts={r.get('thoughts')} llm_usage={r.get('llm_usage')}")


def main() -> int:
    aid = agent_id()
    out = ROOT / "outputs"
    nums = [a for a in sys.argv[1:] if not a.startswith("--")]
    paths = [out / f"{aid}-run-{n}.json" for n in nums] or [out / f"{aid}-run.json"]
    for path in paths:
        if path.exists():
            show(path)
        else:
            print(f"missing {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
