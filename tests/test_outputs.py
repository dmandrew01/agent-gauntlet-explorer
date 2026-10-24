"""arena_clients.outputs: never raises, never leaves outputs/, never prints images."""

from __future__ import annotations

import base64
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from arena_clients.outputs import missing_tools_note, save_submitted_image, submission_summary_lines

PNG_URI = "data:image/png;base64," + base64.b64encode(b"png-bytes").decode("ascii")
JPEG_URI = "data:image/jpeg;base64," + base64.b64encode(b"jpeg-bytes").decode("ascii")


class SaveSubmittedImageTests(unittest.TestCase):
    def setUp(self) -> None:
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)

    def test_writes_decoded_bytes_with_matching_extension(self) -> None:
        status = save_submitted_image("team-nemo", JPEG_URI, self.root)

        path = self.root / "outputs" / "team-nemo-submission.jpg"
        self.assertEqual(status, f"saved to {path}")
        self.assertEqual(path.read_bytes(), b"jpeg-bytes")

    def test_agent_id_cannot_escape_outputs(self) -> None:
        status = save_submitted_image("../../etc/x", PNG_URI, self.root)

        written = list(self.root.rglob("*.png"))
        self.assertEqual(len(written), 1)
        self.assertEqual(written[0].parent, self.root / "outputs")
        self.assertTrue(status.startswith("saved to "))

    def test_non_inline_values_are_skipped_without_raising(self) -> None:
        for value in ("https://example.com/out.png", "", None, 42, "data:text/plain;base64,aGk="):
            with self.subTest(value=value):
                self.assertEqual(
                    save_submitted_image("team-nemo", value, self.root),
                    "not saved (not an inline image)",
                )
        self.assertFalse((self.root / "outputs").exists())

    def test_url_is_never_fetched(self) -> None:
        with mock.patch("urllib.request.urlopen") as urlopen:
            save_submitted_image("team-nemo", "https://example.com/out.png", self.root)
        urlopen.assert_not_called()

    def test_corrupt_base64_is_reported_not_raised(self) -> None:
        status = save_submitted_image("team-nemo", "data:image/png;base64,***", self.root)
        self.assertEqual(status, "not saved (Error)")

    def test_outputs_path_taken_by_a_file_is_reported_not_raised(self) -> None:
        (self.root / "outputs").write_text("not a directory", encoding="utf-8")
        status = save_submitted_image("team-nemo", PNG_URI, self.root)
        self.assertEqual(status, "not saved (FileExistsError)")

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root ignores permissions")
    def test_read_only_directory_is_reported_not_raised(self) -> None:
        self.root.chmod(0o500)
        self.addCleanup(self.root.chmod, 0o700)
        status = save_submitted_image("team-nemo", PNG_URI, self.root)
        self.assertEqual(status, "not saved (PermissionError)")


class SubmissionSummaryTests(unittest.TestCase):
    def test_mcp_image_result_omits_image_data(self) -> None:
        result = {
            "accepted": True,
            "edited_image": PNG_URI,
            "score": {"final_score": 47.8, "output_image_uri": PNG_URI},
        }
        lines = submission_summary_lines(result)
        self.assertEqual(lines, ["Accepted: True", "Score: 47.8"])
        self.assertNotIn("data:image", "\n".join(lines))

    def test_rest_submit_result_includes_status(self) -> None:
        result = types.SimpleNamespace(
            accepted=True, status="submitted", score={"final_score": 99.35}, answer=PNG_URI
        )
        self.assertEqual(
            submission_summary_lines(result),
            ["Accepted: True", "Status: submitted", "Score: 99.35"],
        )

    def test_judge_error_is_explained(self) -> None:
        lines = submission_summary_lines(
            {
                "accepted": True,
                "score": {"final_score": 25.0, "judge_error": "Judge call failed: timeout", "pending_evaluation": True},
            }
        )
        self.assertIn("Judge error: Judge call failed: timeout (quality 0 until the image can be judged)", lines)

    def test_missing_required_tools_get_their_own_line(self) -> None:
        result = {
            "accepted": False,
            "score": {"final_score": 28.15, "missing_required_tools": ["image_generate"]},
        }
        self.assertEqual(
            submission_summary_lines(result),
            ["Accepted: False", "Score: 28.15", "Missing required tools: image_generate"],
        )

    def test_unexpected_result_does_not_raise(self) -> None:
        class Exploding:
            def __getattr__(self, name):
                raise RuntimeError("boom")

        self.assertEqual(
            submission_summary_lines(Exploding()),
            ["Submission result unavailable (RuntimeError)"],
        )


class MissingToolsNoteTests(unittest.TestCase):
    def test_rest_result_names_each_missing_tool(self) -> None:
        result = types.SimpleNamespace(
            score={"missing_required_tools": ["arena.clues.list", "arena.clues.get"]}
        )
        self.assertEqual(
            missing_tools_note(result),
            "Missing required tools: arena.clues.list, arena.clues.get",
        )

    def test_nothing_missing_is_empty(self) -> None:
        for result in (
            {"score": {"final_score": 90, "missing_required_tools": []}},
            types.SimpleNamespace(score=None),
            {"accepted": True},
            None,
        ):
            with self.subTest(result=result):
                self.assertEqual(missing_tools_note(result), "")

    def test_unexpected_result_does_not_raise(self) -> None:
        class Exploding:
            def __getattr__(self, name):
                raise RuntimeError("boom")

        self.assertEqual(missing_tools_note(Exploding()), "")


if __name__ == "__main__":
    unittest.main()
