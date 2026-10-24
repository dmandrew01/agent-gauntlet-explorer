"""Competitor clients must keep reading usage_scope from the public competition payload."""

from __future__ import annotations

import json
import unittest
from unittest import mock

from arena_clients.http_client import HttpArenaClient

MINTED_SCOPE = "battle-scope-w0-redaction"


class UsageScopeContractTests(unittest.TestCase):
    def test_fetch_usage_scope_resolves_top_level_field_from_redacted_payload(self) -> None:
        client = HttpArenaClient(
            api_base="https://arena.example",
            api_key="team-key",
        )
        payload = {
            "status": "running",
            "phase": "running",
            "challenge_type": "text",
            "puzzle_id": "breakfast_logic_001",
            "usage_scope": MINTED_SCOPE,
            "run_id": "text-abc123",
            "runs": {
                "text": {
                    "run_id": "text-abc123",
                    "run_type": "text",
                    "status": "running",
                    "started_at": 100.0,
                    "ended_at": None,
                    "exit_code": None,
                    "puzzle_id": "breakfast_logic_001",
                    "usage_scope": MINTED_SCOPE,
                }
            },
        }
        next_scope = "scope-b"
        next_payload = dict(payload)
        next_payload["usage_scope"] = next_scope
        next_runs = dict(payload["runs"])
        next_text_run = dict(payload["runs"]["text"])
        next_text_run["usage_scope"] = next_scope
        next_runs["text"] = next_text_run
        next_payload["runs"] = next_runs
        client._request = mock.Mock(side_effect=[payload, next_payload])

        first_scope = client.fetch_usage_scope()
        second_scope = client.fetch_usage_scope()

        self.assertEqual(first_scope, MINTED_SCOPE)
        self.assertEqual(second_scope, next_scope)
        self.assertEqual(client._request.call_count, 2)
        client._request.assert_has_calls(
            [
                mock.call("GET", "/api/competition"),
                mock.call("GET", "/api/competition"),
            ]
        )
        serialized = json.dumps(payload)
        self.assertNotIn("arena_api_key", serialized)
        self.assertNotIn("command", serialized)
        self.assertNotIn("log_path", serialized)
