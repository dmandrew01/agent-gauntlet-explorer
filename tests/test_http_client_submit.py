"""HttpArenaClient.submit leaves the challenge type to the server by default."""

from __future__ import annotations

import unittest
from unittest import mock

from arena_clients.http_client import HttpArenaClient

RECEIPT = {"accepted": True, "agent_id": "alpha", "answer": "A", "score": {}, "status": "submitted"}


class SubmitChallengeTypeTests(unittest.TestCase):
    def _submit(self, **kwargs) -> dict:
        client = HttpArenaClient(api_base="https://arena.example", api_key="k")
        with mock.patch.object(client, "_request", return_value=RECEIPT) as request:
            client.submit("alpha", "A", **kwargs)
        method, path, payload = request.call_args.args
        self.assertEqual((method, path), ("POST", "/api/submit"))
        return payload

    def test_default_omits_challenge_type_so_the_server_uses_the_assignment(self) -> None:
        # Practice assigns text or image per agent; sending "text" for an
        # image-assigned agent is rejected with 409.
        self.assertNotIn("challenge_type", self._submit())

    def test_explicit_challenge_type_is_still_sent(self) -> None:
        self.assertEqual(self._submit(challenge_type="image")["challenge_type"], "image")


if __name__ == "__main__":
    unittest.main()
