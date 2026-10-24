#!/usr/bin/env python3
"""Gauntlet Explorer entrypoint.

Usage (from the repository root or this folder):
    python examples/explorer/agent.py

Reads ARENA_SERVER, ARENA_API_KEY and AGENT_ID from the environment or the
repository-root .env file. Works when no .env exists (the organizer's runner
sets the environment).
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from gauntlet.runner import run  # noqa: E402
from my_strategy import MyStrategy  # noqa: E402


def main() -> int:
    try:
        return asyncio.run(run(MyStrategy()))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    code = main()
    # The answer is submitted and the run record written by now. Exit at once
    # instead of waiting on an abandoned slow model request to time out.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code if isinstance(code, int) else 1)
