#!/usr/bin/env python3
"""Package the Gauntlet Explorer with the kit's own packager and self-check.

    set -a; . ./.env; set +a
    python tools/package_explorer.py --agent-name "AHGroup"          # first build
    python tools/package_explorer.py --agent-name "AHGroup" --force  # rebuild

It passes every file of the gauntlet package with --include (the kit's
import check only sees top-level packages, so submodules must be listed),
adds gauntlet/model_stats.json when calibration produced one, and then runs
`--check` on the result. Outputs land in dist/ exactly like the kit's flow.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arena_clients import package  # noqa: E402

ENTRYPOINT = "examples/explorer/agent.py"


def gauntlet_files() -> list[str]:
    files = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "gauntlet").glob("*.py"))
    stats = ROOT / "gauntlet" / "model_stats.json"
    if stats.is_file():
        files.append(stats.relative_to(ROOT).as_posix())
    return files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-id", default=os.getenv("AGENT_ID", ""))
    parser.add_argument("--agent-name", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output", default="dist")
    args = parser.parse_args()
    if not args.agent_id:
        print("Set AGENT_ID in .env (or pass --agent-id).", file=sys.stderr)
        return 2
    argv = ["--agent-id", args.agent_id, "--agent-name", args.agent_name, "--entrypoint", ENTRYPOINT, "--output", args.output]
    for rel in gauntlet_files():
        argv += ["--include", rel]
    if args.force:
        argv.append("--force")
    os.chdir(ROOT)
    code = package.main(argv)
    if code:
        return code
    return package.main(["--check", str(Path(args.output) / package.SUBMISSION_DIR_NAME)])


if __name__ == "__main__":
    raise SystemExit(main())
