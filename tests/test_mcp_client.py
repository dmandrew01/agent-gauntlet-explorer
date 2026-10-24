from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest import mock
from urllib.error import HTTPError, URLError

from arena_clients.mcp_client import McpArenaClient, McpArenaError, build_image_tool_arguments


class MCPAuthenticationTransportTests(unittest.TestCase):
    def test_credential_is_carried_in_header_not_url(self) -> None:
        client = McpArenaClient(
            "https://practice.example",
            api_key="secret/key",
        )

        self.assertEqual(client.sse_url, "https://practice.example/sse")
        self.assertEqual(
            client._headers,
            {"X-Arena-API-Key": "secret/key"},
        )


class ImageToolArgumentTests(unittest.TestCase):
    def test_builds_selected_model_and_byo_proxy_key_from_live_schema(self) -> None:
        tool_defs = [
            SimpleNamespace(
                name="image_generate",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "model": {"type": ["string", "null"]},
                        "proxy_api_key": {"type": ["string", "null"]},
                    },
                },
            )
        ]

        arguments = build_image_tool_arguments(
            tool_defs,
            "image_generate",
            selected_model="gpt-5-image-mini",
            llm_api_key="provider-key",
            arena_api_key="team-key",
        )

        self.assertEqual(
            arguments,
            {
                "model": "gpt-5-image-mini",
                "proxy_api_key": "provider-key",
            },
        )


class ClueHydrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetches_every_listed_clue_in_server_order(self) -> None:
        client = McpArenaClient("https://practice.example", api_key="team-key")
        client.list_clues = mock.AsyncMock(return_value=["clue_0", "clue_1"])
        client.get_clue = mock.AsyncMock(
            side_effect=[
                SimpleNamespace(text="First clue"),
                SimpleNamespace(text="Second clue"),
            ]
        )

        clues = await client.get_all_clue_texts("agent-one")

        self.assertEqual(clues, ["First clue", "Second clue"])
        self.assertEqual(
            client.get_clue.await_args_list,
            [
                mock.call("clue_0", "agent-one"),
                mock.call("clue_1", "agent-one"),
            ],
        )


class NotImageChallengeClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_image_challenge_raises_terminal_not_image_challenge(self) -> None:
        client = McpArenaClient("https://practice.example", api_key="team-key")
        client._session = mock.Mock()
        client._session.call_tool = mock.AsyncMock(return_value=object())
        client._parse_result = mock.Mock(
            return_value={
                "error": "Authoritative assignment is not an image challenge.",
                "code": "not_image_challenge",
                "accepted": False,
                "challenge_type": "text",
            }
        )

        with self.assertRaises(McpArenaError) as ctx:
            await client.get_image_challenge("alpha")

        self.assertEqual(ctx.exception.code, "not_image_challenge")
        self.assertTrue(ctx.exception.is_terminal)
        self.assertNotIn("locked", str(ctx.exception).lower())

    def test_locked_error_is_not_terminal(self) -> None:
        exc = McpArenaError("Challenge locked. Waiting for organizer to open lobby.")
        self.assertIsNone(exc.code)
        self.assertFalse(exc.is_terminal)


class DetectModalityTests(unittest.TestCase):
    TOOLS = ["arena.get_challenge", "arena.image.get_challenge"]

    def _detect(self, responses: dict[str, object], **kwargs) -> tuple[str, list]:
        """Run detect_modality with urlopen answering by URL path.

        A dict value is returned as the JSON body; an exception is raised; a
        list gives one outcome per call, in order.
        """
        requests = []
        self.timeouts = []

        def fake_urlopen(request, timeout):
            requests.append(request)
            self.timeouts.append(timeout)
            path = request.full_url.removeprefix("https://arena.example.com")
            outcome = responses.get(path, HTTPError(request.full_url, 404, "nf", {}, None))
            if isinstance(outcome, list):
                outcome = outcome.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            body = mock.MagicMock()
            body.read.return_value = json.dumps(outcome).encode("utf-8")
            context = mock.MagicMock()
            context.__enter__.return_value = body
            return context

        with (
            mock.patch("arena_clients.mcp_client.get_api_base", return_value="https://arena.example.com"),
            mock.patch("arena_clients.mcp_client.get_arena_api_key", return_value="battle-key"),
            mock.patch("arena_clients.mcp_client.urlopen", side_effect=fake_urlopen),
        ):
            modality = McpArenaClient.detect_modality(self.TOOLS, **kwargs)
        return modality, requests

    def test_own_session_wins_over_competition(self) -> None:
        # Practice mixed mode: this agent got image, the competition already
        # describes the next registration's text challenge.
        modality, requests = self._detect(
            {
                "/api/session/team-nemo": {"agent_id": "team-nemo", "challenge_type": "image"},
                "/api/competition": {"challenge_type": "logic-puzzle"},
            },
            agent_id="team-nemo",
        )

        self.assertEqual(modality, "image")
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].get_header("X-arena-api-key"), "battle-key")

    def test_session_text_assignment_is_text(self) -> None:
        modality, _ = self._detect(
            {
                "/api/session/team-nemo": {"challenge_type": "text"},
                "/api/competition": {"challenge_type": "image-generate"},
            },
            agent_id="team-nemo",
        )
        self.assertEqual(modality, "text")

    def test_agent_id_is_url_encoded(self) -> None:
        _, requests = self._detect(
            {"/api/session/team%2Fnemo": {"challenge_type": "text"}},
            agent_id="team/nemo",
        )
        self.assertTrue(requests[0].full_url.endswith("/api/session/team%2Fnemo"))

    def test_missing_session_falls_back_to_competition(self) -> None:
        modality, requests = self._detect(
            {"/api/competition": {"challenge_type": "image"}},
            agent_id="team-nemo",
        )
        self.assertEqual(modality, "image")
        self.assertEqual(len(requests), 2)

    def test_without_agent_id_uses_competition_only(self) -> None:
        modality, requests = self._detect({"/api/competition": {"challenge_type": "image"}})
        self.assertEqual(modality, "image")
        self.assertEqual([r.full_url for r in requests], ["https://arena.example.com/api/competition"])

    def test_slow_first_session_lookup_is_retried_not_guessed(self) -> None:
        # The first request also sets up TLS; on Practice it took 1.5s, which
        # used to time out and fall back to the competition's (wrong) type.
        modality, requests = self._detect(
            {
                "/api/session/team-nemo": [TimeoutError("timed out"), {"challenge_type": "image"}],
                "/api/competition": {"challenge_type": "logic-puzzle"},
            },
            agent_id="team-nemo",
        )
        self.assertEqual(modality, "image")
        self.assertEqual(len(requests), 2)
        self.assertTrue(all(r.full_url.endswith("/api/session/team-nemo") for r in requests))

    def test_session_lookup_allows_a_slow_connection(self) -> None:
        self._detect({"/api/session/team-nemo": {"challenge_type": "image"}}, agent_id="team-nemo")
        self.assertGreaterEqual(self.timeouts[0], 10)

    def test_unregistered_session_is_not_retried(self) -> None:
        _, requests = self._detect({"/api/competition": {"challenge_type": "text"}}, agent_id="team-nemo")
        self.assertEqual(
            [r.full_url.rsplit("/api/", 1)[1] for r in requests],
            ["session/team-nemo", "competition"],
        )

    def test_unreachable_api_falls_back_to_tools(self) -> None:
        modality, _ = self._detect(
            {
                "/api/session/team-nemo": URLError("down"),
                "/api/competition": URLError("down"),
            },
            agent_id="team-nemo",
        )
        self.assertEqual(modality, "text")


if __name__ == "__main__":
    unittest.main()
