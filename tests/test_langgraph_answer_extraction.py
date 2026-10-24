from __future__ import annotations

from pathlib import Path
import sys
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
LANGGRAPH_DIR = ROOT / "examples" / "langgraph"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(LANGGRAPH_DIR) not in sys.path:
    sys.path.insert(0, str(LANGGRAPH_DIR))

from arena_clients.proxy_headers import build_proxy_headers  # noqa: E402
import agent as langgraph_agent  # noqa: E402
from base_strategy import BaseStrategy, ChallengeContext  # noqa: E402
from agent import (  # noqa: E402
    _build_mcp_sse_connection,
    _pick_react_model,
    _derive_answer_from_challenge,
    _derive_answer_from_evidence,
    _extract_ordered_answer_from_rules,
    _is_invalid_answer_candidate,
    extract_answer,
)


BREAKFAST_RULES = (
    "Find the unique chronological order (1st to 5th) that satisfies all "
    "constraints. Format: 'Name1, Name2, Name3, Name4, Name5'"
)
FAKE_LLM_HOST = "https://llm-proxy.example.test"
FAKE_LLM_API_KEY = "test-api-key"


class LangGraphAnswerExtractionTests(unittest.TestCase):
    def test_rejects_placeholder_format_from_rules(self) -> None:
        fallback = _extract_ordered_answer_from_rules(BREAKFAST_RULES)

        self.assertTrue(_is_invalid_answer_candidate(fallback))

    def test_extracts_nested_answer_line_from_reasoning(self) -> None:
        raw = "\n".join(
            [
                "Thus order: Eve, Bob, Alice, Charlie, David.",
                "Thus final answer: ANSWER: Eve, Bob, Alice, Charlie, David",
            ]
        )

        self.assertEqual(
            extract_answer(raw, BREAKFAST_RULES),
            "Eve, Bob, Alice, Charlie, David",
        )

    def test_placeholder_format_is_not_valid_final_answer(self) -> None:
        self.assertTrue(
            _is_invalid_answer_candidate("Name1, Name2, Name3, Name4, Name5")
        )

    def test_extracts_order_line_from_thought_stream(self) -> None:
        raw = "\n".join(
            [
                "Exactly one person between Eve and Alice: yes.",
                "Thus unique order: Eve, Bob, Alice, Charlie, David.",
                "Output format: 'Name1, Name2, Name3, Name4, Name5' exactly.",
            ]
        )

        self.assertEqual(
            extract_answer(raw, BREAKFAST_RULES),
            "Eve, Bob, Alice, Charlie, David",
        )


class LangGraphProxyHeaderTests(unittest.TestCase):
    def test_mcp_credential_is_carried_in_header_not_url(self) -> None:
        connection = _build_mcp_sse_connection(
            "https://arena.example.test/mcp",
            "arena-secret",
        )

        self.assertEqual(connection["url"], "https://arena.example.test/mcp/sse")
        self.assertEqual(
            connection["headers"],
            {"X-Arena-API-Key": "arena-secret"},
        )
        self.assertNotIn("arena-secret", connection["url"])

    def test_build_proxy_headers_uses_agent_and_round_scope(self) -> None:
        self.assertEqual(
            build_proxy_headers("team-beacon", "round-123"),
            {"X-Agent-ID": "team-beacon", "X-Round-ID": "round-123"},
        )

    def test_evidence_fallback_passes_proxy_headers_to_openai_client(self) -> None:
        captured_headers: dict[str, str] = {}

        class FakeOpenAI:
            def __init__(self, **kwargs):
                captured_headers.update(kwargs.get("default_headers") or {})
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(create=self._create)
                )

            def _create(self, **_kwargs):
                message = types.SimpleNamespace(content="Eve, Bob, Alice, Charlie, David")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_openai = types.ModuleType("openai")
        fake_openai.OpenAI = FakeOpenAI

        with mock.patch.dict(sys.modules, {"openai": fake_openai}):
            answer = _derive_answer_from_evidence(
                llm_host=FAKE_LLM_HOST,
                llm_api_key=FAKE_LLM_API_KEY,
                model_name="gpt-oss-20b",
                challenge_description="Find the order.",
                challenge_rules=BREAKFAST_RULES,
                evidence_lines=["Order: Eve, Bob, Alice, Charlie, David"],
                agent_id="team-beacon",
                usage_scope="round-123",
            )

        self.assertEqual(answer, "Eve, Bob, Alice, Charlie, David")
        self.assertEqual(
            captured_headers,
            {"X-Agent-ID": "team-beacon", "X-Round-ID": "round-123"},
        )

    def test_direct_fallback_passes_proxy_headers_to_openai_client(self) -> None:
        captured_headers: dict[str, str] = {}

        class FakeOpenAI:
            def __init__(self, **kwargs):
                captured_headers.update(kwargs.get("default_headers") or {})
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(create=self._create)
                )

            def _create(self, **_kwargs):
                message = types.SimpleNamespace(content="Eve, Bob, Alice, Charlie, David")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        fake_openai = types.ModuleType("openai")
        fake_openai.OpenAI = FakeOpenAI

        with mock.patch.dict(sys.modules, {"openai": fake_openai}):
            answer = _derive_answer_from_challenge(
                llm_host=FAKE_LLM_HOST,
                llm_api_key=FAKE_LLM_API_KEY,
                model_name="gpt-oss-20b",
                challenge_description="Find the order.",
                challenge_rules=BREAKFAST_RULES,
                clues=["Eve is first."],
                evidence_lines=["Order: Eve, Bob, Alice, Charlie, David"],
                agent_id="team-beacon",
                usage_scope="round-123",
            )

        self.assertEqual(answer, "Eve, Bob, Alice, Charlie, David")
        self.assertEqual(
            captured_headers,
            {"X-Agent-ID": "team-beacon", "X-Round-ID": "round-123"},
        )


class ReactModelForImageTests(unittest.TestCase):
    """The ReAct agent must call tools, which image models cannot do."""

    MODELS = ["gemini-3.1-flash-lite-image", "gemini-3.1-flash-image", "nemotron-3.5-lightning", "qwen3.7-flash"]

    class _ImageFirst(BaseStrategy):
        def rank_models(self, ctx, available_models):
            images = [m for m in available_models if "image" in m]
            others = [m for m in available_models if "image" not in m]
            return images + others if ctx.challenge_type.startswith("image") else others + images

    def _pick(self, modality: str, challenge_type: str, strategy_model: str) -> str:
        ctx = ChallengeContext(challenge_type=challenge_type, image_url="https://x/in.jpg")
        with mock.patch.object(langgraph_agent, "STRATEGY", self._ImageFirst()):
            return _pick_react_model(ctx, self.MODELS, modality, strategy_model)

    def test_image_challenge_drives_react_with_a_non_image_model(self) -> None:
        picked = self._pick("image", "image-generate", "gemini-3.1-flash-lite-image")
        self.assertEqual(picked, "nemotron-3.5-lightning")

    def test_text_challenge_keeps_the_strategy_model(self) -> None:
        self.assertEqual(self._pick("text", "logic-puzzle", "qwen3.7-flash"), "qwen3.7-flash")

    def test_roster_with_only_image_models_falls_back_to_the_strategy_model(self) -> None:
        ctx = ChallengeContext(challenge_type="image-edit")
        with mock.patch.object(langgraph_agent, "STRATEGY", self._ImageFirst()):
            picked = _pick_react_model(ctx, ["gemini-3.1-flash-image"], "image", "gemini-3.1-flash-image")
        self.assertEqual(picked, "gemini-3.1-flash-image")


if __name__ == "__main__":
    unittest.main()
