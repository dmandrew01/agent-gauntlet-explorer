"""A default AGENT_ID is refused on a shared server, where teams would collide."""

from __future__ import annotations

import os
import re
import unittest
from pathlib import Path
from unittest import mock

from arena_clients.config import KIT_DEFAULT_AGENT_IDS, agent_id_problem

REMOTE = {"ARENA_SERVER": "https://practice.example.com"}
LOCAL = {"ARENA_SERVER": "127.0.0.1"}
EXAMPLES = sorted((Path(__file__).resolve().parents[1] / "examples").glob("*/agent.py"))


def _env(values: dict[str, str]):
    cleared = {name: "" for name in ("ARENA_API_BASE", "ARENA_USAGE_SCOPE")}
    return mock.patch.dict(os.environ, {**cleared, **values})


class AgentIdProblemTests(unittest.TestCase):
    def test_default_ids_are_refused_on_a_remote_server(self) -> None:
        with _env(REMOTE):
            for agent_id in ("my-agent", "My-Agent ", "", "simple-agent", "langgraph-agent"):
                with self.subTest(agent_id=agent_id):
                    problem = agent_id_problem(agent_id)
                    self.assertIn("Set AGENT_ID in .env", problem)
                    self.assertIn("another team's run will overwrite yours", problem)

    def test_a_team_id_is_accepted(self) -> None:
        with _env(REMOTE):
            self.assertEqual(agent_id_problem("team-nemo"), "")

    def test_local_servers_allow_the_defaults(self) -> None:
        with _env(LOCAL):
            self.assertEqual(agent_id_problem("my-agent"), "")

    def test_organizer_launched_runs_are_never_blocked(self) -> None:
        # The runner sets ARENA_USAGE_SCOPE and the submission's AGENT_ID.
        with _env({**REMOTE, "ARENA_USAGE_SCOPE": "heat-7"}):
            self.assertEqual(agent_id_problem("my-agent"), "")

    def test_every_example_default_is_covered(self) -> None:
        for path in EXAMPLES:
            match = re.search(r'^DEFAULT_AGENT_ID = "([^"]+)"', path.read_text(encoding="utf-8"), re.M)
            if match:
                with self.subTest(example=path.parent.name):
                    self.assertIn(match.group(1), KIT_DEFAULT_AGENT_IDS)


class ExamplesCheckTheIdTests(unittest.TestCase):
    def test_each_example_checks_the_id_before_connecting(self) -> None:
        self.assertGreaterEqual(len(EXAMPLES), 4)
        for path in EXAMPLES:
            with self.subTest(example=path.parent.name):
                source = path.read_text(encoding="utf-8")
                guard = source.find('id_problem = agent_id_problem(os.environ["AGENT_ID"])')
                self.assertNotEqual(guard, -1)
                for call in ("ensure_connected()", ".register("):
                    first_use = source.find(call, source.find("os.environ[\"AGENT_ID\"] ="))
                    if first_use != -1:
                        self.assertLess(guard, first_use, call)


if __name__ == "__main__":
    unittest.main()
