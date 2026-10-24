"""arena_clients.display: short, never a dump, never breaks a run."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from arena_clients.display import (
    ReasoningTicker,
    challenge_lines,
    challenge_record,
    save_run_record,
    summary_lines,
)
from arena_clients.mcp_client import ChallengeInfo, ImageChallengeInfo

LONG_QUESTION = "Six events happened in a single day. " * 20


def text_challenge(**overrides) -> ChallengeInfo:
    fields = dict(
        challenge_type="logic-puzzle",
        challenge_id="practice_logic_easy_001",
        puzzle_id="practice_logic_easy_001",
        description="Four friends stand in a line. Determine their order.",
        rules="Return only the names in order, comma-separated.",
        max_time_s=60,
        clues=["Ava is left of Ben.", "Maya is right of Ben.", "Leo is not first."],
        time_remaining_s=60.0,
        raw_data={"difficulty": "easy", "required_tools": ["arena.get_challenge", "arena.clues.list"]},
    )
    fields.update(overrides)
    return ChallengeInfo(**fields)


def image_challenge(**overrides) -> ImageChallengeInfo:
    fields = dict(
        challenge_type="image-generate",
        challenge_id="img",
        puzzle_id="practice_image_generate_medium_001",
        difficulty="medium",
        description="Generate an original image.",
        prompt="Create a cinematic poster of a futuristic electric motorcycle on a rainy neon street.",
        reference_notes="Single coherent scene.",
        max_time_s=150,
        input_image_uri="",
        time_remaining_s=150.0,
    )
    fields.update(overrides)
    return ImageChallengeInfo(**fields)


class ChallengeSummaryTests(unittest.TestCase):
    def test_text_summary_is_short_and_points_to_the_record_for_clues(self) -> None:
        lines = challenge_lines(text_challenge())
        joined = "\n".join(lines)

        self.assertLessEqual(len(lines), 6)
        self.assertIn("practice_logic_easy_001 · logic-puzzle · easy · time limit 60s", lines[0])
        self.assertIn("Four friends stand in a line.", joined)
        self.assertIn("Clues      3 (full text in the run record)", joined)
        self.assertNotIn("Ava is left of Ben.", joined)
        self.assertIn("Must call  arena.get_challenge, arena.clues.list", joined)

    def test_long_question_is_capped_at_two_lines(self) -> None:
        lines = challenge_lines(text_challenge(description=LONG_QUESTION))
        question = [line for line in lines if "Six events" in line or line.startswith("              ")]
        self.assertLessEqual(len(question), 2)
        self.assertTrue(question[-1].endswith("…"))

    def test_image_summary_shows_the_task_and_input(self) -> None:
        joined = "\n".join(challenge_lines(image_challenge()))
        self.assertIn("futuristic electric motorcycle", joined)
        self.assertIn("Input      none (generate from scratch)", joined)

        edit = "\n".join(challenge_lines(image_challenge(input_image_uri="https://x/in.jpg")))
        self.assertIn("Input      an image to edit", edit)

    def test_unexpected_object_does_not_raise(self) -> None:
        self.assertTrue(challenge_lines(object())[0].startswith("🎯 Challenge"))


class FakeTty(io.StringIO):
    def isatty(self) -> bool:
        return True


class ReasoningTickerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = [0.0]
        self.clock = lambda: self.now[0]

    def test_terminal_redraws_one_line_and_never_dumps_the_reasoning(self) -> None:
        out = FakeTty()
        ticker = ReasoningTicker("model-a", stream=out, enabled=True, clock=self.clock)
        reasoning = "Ben must be left of Maya so Ava goes first. " * 200
        for start in range(0, len(reasoning), 50):
            self.now[0] += 1.0
            ticker.feed(reasoning=reasoning[start : start + 50])
        ticker.finish()

        written = out.getvalue()
        self.assertNotIn("\n", written)
        frames = [frame for frame in written.split("\r") if frame.strip()]
        self.assertTrue(frames)
        self.assertTrue(all(len(frame) <= 120 for frame in frames))
        self.assertLess(max(frame.count("Ben must") for frame in frames), 3)

    def test_non_terminal_prints_a_progress_line_at_most_every_interval(self) -> None:
        out = io.StringIO()
        ticker = ReasoningTicker("model-a", stream=out, enabled=True, plain_interval_s=10, clock=self.clock)
        for _ in range(60):
            self.now[0] += 1.0
            ticker.feed(reasoning="thinking ")
        self.assertLessEqual(len(out.getvalue().splitlines()), 7)

    def test_show_reasoning_off_prints_nothing(self) -> None:
        out = FakeTty()
        with mock.patch.dict(os.environ, {"SHOW_REASONING": "off"}):
            ticker = ReasoningTicker("model-a", stream=out, clock=self.clock)
        self.now[0] += 5
        ticker.feed(reasoning="anything")
        ticker.finish()
        self.assertEqual(out.getvalue(), "")

    def test_bad_input_does_not_raise(self) -> None:
        ticker = ReasoningTicker("m", stream=io.StringIO(), enabled=True, clock=self.clock)
        ticker.feed(reasoning=None, answer=42)  # type: ignore[arg-type]
        ticker.finish()


class SummaryTests(unittest.TestCase):
    def test_summary_names_time_tokens_stop_reason_and_answer(self) -> None:
        lines = summary_lines(
            model="nemotron-3.5-lightning",
            elapsed_s=22.4,
            usage={"completion_tokens": 5148},
            finish_reason="length",
            answer="Ava, Ben, Maya, Leo",
        )
        self.assertEqual(
            lines[0], "🧠 nemotron-3.5-lightning done in 22.4s · 5,148 tokens · stopped: token limit"
        )
        self.assertIn("Ava, Ben, Maya, Leo", lines[1])

    def test_estimated_token_count_is_marked(self) -> None:
        lines = summary_lines(
            model="m", elapsed_s=2.0, usage={"completion_tokens": 0, "total_tokens": 10}, finish_reason="stop"
        )
        self.assertIn("~10 tokens", lines[0])


class RunRecordTests(unittest.TestCase):
    def test_record_holds_the_full_challenge_and_is_written(self) -> None:
        record = {"challenge": challenge_record(text_challenge()), "answer": "x"}
        self.assertEqual(record["challenge"]["clues"][0], "Ava is left of Ben.")
        self.assertEqual(record["challenge"]["required_tools"], ["arena.get_challenge", "arena.clues.list"])
        with tempfile.TemporaryDirectory() as tmp:
            status = save_run_record("../team/x", record, Path(tmp))
            files = list(Path(tmp).rglob("*-run.json"))
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].parent, Path(tmp) / "outputs")
            self.assertEqual(json.loads(files[0].read_text())["answer"], "x")
            self.assertTrue(status.startswith("saved to "))

    def test_inline_input_image_is_not_copied_into_the_record(self) -> None:
        record = challenge_record(image_challenge(input_image_uri="data:image/png;base64,AAAA"))
        self.assertEqual(record["input_image_uri"], "(inline image)")

    def test_unwritable_outputs_is_reported_not_raised(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "outputs").write_text("not a directory", encoding="utf-8")
            self.assertEqual(save_run_record("a", {}, Path(tmp)), "not saved (FileExistsError)")


if __name__ == "__main__":
    unittest.main()
