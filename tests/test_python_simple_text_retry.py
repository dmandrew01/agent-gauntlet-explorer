"""Python Simple retries with another model when a reasoning model runs out of tokens."""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import io
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from base_strategy import BaseStrategy, ChallengeContext

SIMPLE_AGENT = ROOT / "examples" / "python_simple" / "agent.py"


def _load_simple_agent_module():
    strategy_module = types.ModuleType("my_strategy")
    strategy_module.MyStrategy = type("MyStrategy", (BaseStrategy,), {})
    with mock.patch.dict(sys.modules, {"my_strategy": strategy_module}):
        spec = importlib.util.spec_from_file_location("python_simple_text_retry", SIMPLE_AGENT)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def _usage(completion: int, finish_reason: str) -> dict:
    return {
        "finish_reason": finish_reason,
        "prompt_tokens": 100,
        "completion_tokens": completion,
        "total_tokens": 100 + completion,
    }


class NextRankedModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load_simple_agent_module()

    def test_picks_the_model_after_the_current_one(self) -> None:
        self.assertEqual(self.module._next_ranked_model(["a", "b", "c"], "a"), "b")

    def test_wraps_to_an_earlier_model_at_the_end(self) -> None:
        self.assertEqual(self.module._next_ranked_model(["a", "b", "c"], "c"), "a")

    def test_unknown_current_takes_the_first(self) -> None:
        self.assertEqual(self.module._next_ranked_model(["a", "b"], "x"), "a")

    def test_no_other_model(self) -> None:
        self.assertEqual(self.module._next_ranked_model(["a"], "a"), "")


class RetryAfterLengthStopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load_simple_agent_module()
        self.solved_with: list[str] = []

    def _run(self, first, *, remaining: float = 90.0):
        async def remaining_s() -> float:
            return remaining

        async def solve_with(model: str):
            self.solved_with.append(model)
            return ("Ava, Ben", "ANSWER: Ava, Ben", model, _usage(600, "stop"), 50, 4000)

        with contextlib.redirect_stdout(io.StringIO()) as out:
            result = asyncio.run(
                self.module._retry_after_length_stop(
                    first,
                    ranked_models=["lightning", "super", "flash"],
                    remaining_s=remaining_s,
                    solve_with=solve_with,
                )
            )
        return result, out.getvalue()

    def test_out_of_tokens_retries_with_next_model_and_sums_cost(self) -> None:
        first = ("", "...thinking...", "lightning", _usage(8192, "length"), 0, 20000)

        (answer, _raw, model, usage, _ttft, total_ms), output = self._run(first)

        self.assertEqual(self.solved_with, ["super"])
        self.assertEqual((answer, model), ("Ava, Ben", "super"))
        self.assertEqual(usage["completion_tokens"], 8192 + 600)
        self.assertEqual(total_ms, 20000 + 4000)
        self.assertIn("lightning used all 8192 tokens thinking", output)
        self.assertIn("Retrying with super", output)

    def test_an_answer_is_never_retried(self) -> None:
        first = ("Ava, Ben", "ANSWER: Ava, Ben", "lightning", _usage(8192, "length"), 0, 1)
        result, _ = self._run(first)
        self.assertEqual(self.solved_with, [])
        self.assertIs(result, first)

    def test_empty_answer_that_finished_normally_is_not_retried(self) -> None:
        first = ("", "no answer line", "lightning", _usage(300, "stop"), 0, 1)
        self._run(first)
        self.assertEqual(self.solved_with, [])

    def test_too_little_time_left_skips_the_retry(self) -> None:
        first = ("", "...thinking...", "lightning", _usage(8192, "length"), 0, 1)
        result, output = self._run(first, remaining=10.0)
        self.assertEqual(self.solved_with, [])
        self.assertIs(result, first)
        self.assertIn("Not retrying (10s left)", output)


class _FakeLlm:
    """chat.completions.create returning queued (content, finish_reason) replies."""

    def __init__(self, replies: list[tuple[str, str]]) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        content, finish_reason = self.replies.pop(0)
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content=content),
                    finish_reason=finish_reason,
                )
            ],
            usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )


class SolveChallengeFinishReasonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load_simple_agent_module()
        self.ctx = ChallengeContext(challenge_type="logic-puzzle", description="Order them.")

    def _solve(self, llm: _FakeLlm):
        return asyncio.run(
            self.module.solve_challenge(None, [], llm, "lightning", BaseStrategy(), self.ctx)
        )

    def test_out_of_tokens_skips_the_same_model_strict_retry(self) -> None:
        llm = _FakeLlm([("", "length")])
        answer, _raw, _model, usage, _ttft, _ms = self._solve(llm)

        self.assertEqual(answer, "")
        self.assertEqual(usage["finish_reason"], "length")
        self.assertEqual(len(llm.calls), 1)

    def test_format_miss_still_gets_the_strict_retry(self) -> None:
        llm = _FakeLlm([("I think it is Ava then Ben", "stop"), ("ANSWER: Ava, Ben", "stop")])
        answer, *_ = self._solve(llm)

        self.assertEqual(len(llm.calls), 2)
        self.assertEqual(answer, "Ava, Ben")

    def test_default_token_limit_leaves_room_to_think(self) -> None:
        llm = _FakeLlm([("ANSWER: Ava, Ben", "stop")])
        self._solve(llm)
        self.assertEqual(llm.calls[0]["max_tokens"], 8192)


class _FakeStreamingLlm:
    """chat.completions.create(stream=True) yielding (reasoning, content) chunks."""

    def __init__(self, chunks: list[tuple[str, str]]) -> None:
        self.chunks = chunks
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        assert kwargs.get("stream") is True
        for reasoning, content in self.chunks:
            delta = types.SimpleNamespace(content=content, reasoning_content=reasoning)
            yield types.SimpleNamespace(
                choices=[types.SimpleNamespace(delta=delta, finish_reason=None)],
                usage=None,
            )


class _QuietHttp:
    def broadcast_thought(self, *args, **kwargs) -> None:
        del args, kwargs


class StreamedAnswerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load_simple_agent_module()
        self.ctx = ChallengeContext(challenge_type="logic-puzzle", description="Order them.")

    def _solve(self, chunks):
        llm = _FakeStreamingLlm(chunks)
        with mock.patch.dict("os.environ", {"LLM_STREAM": "1"}):
            return asyncio.run(
                self.module.solve_challenge(
                    None, [], llm, "lightning", BaseStrategy(), self.ctx,
                    http_client=_QuietHttp(), agent_id="alpha",
                )
            )

    def test_answer_comes_from_the_reply_not_a_draft_in_reasoning(self) -> None:
        # The reasoning ends with a draft answer and no newline; the reply
        # then starts. Joined together they read "...E4ANSWER: E3,...".
        answer, raw, *_ = self._solve(
            [
                ("Try E1 first. ANSWER: E1,E2,E3,E4", ""),
                ("", "ANSWER: E3,E2,E1,E4"),
            ]
        )
        self.assertEqual(answer, "E3,E2,E1,E4")
        self.assertNotIn("E4ANSWER", raw)

    def test_token_estimate_counts_streamed_reasoning(self) -> None:
        reasoning = "word " * 500
        *_, usage, _ttft, _ms = self._solve([(reasoning, ""), ("", "ANSWER: E3")])
        self.assertGreaterEqual(usage["total_tokens"], 1000)

    def test_empty_reply_falls_back_to_reasoning(self) -> None:
        answer, *_ = self._solve([("Worked it out.\nANSWER: E3,E2,E1,E4", "")])
        self.assertEqual(answer, "E3,E2,E1,E4")


if __name__ == "__main__":
    unittest.main()
