"""Examples call the model the strategy picked, never a hard-coded name.

A literal model name goes stale when the roster changes: the proxy rejects
it, and the call it belongs to (often a fallback) silently never works.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

EXAMPLES = sorted((Path(__file__).resolve().parents[1] / "examples").glob("*/agent.py"))


def _literal_keyword_calls(path: Path, keyword: str) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg == keyword and isinstance(kw.value, ast.Constant):
                found.append(f"{path.parent.name}/agent.py:{kw.value.lineno} {keyword}={kw.value.value!r}")
    return found


class ExampleModelNameTests(unittest.TestCase):
    def test_examples_exist(self) -> None:
        self.assertGreaterEqual(len(EXAMPLES), 4)

    def test_no_example_hard_codes_a_model_name(self) -> None:
        for path in EXAMPLES:
            with self.subTest(example=path.parent.name):
                self.assertEqual(_literal_keyword_calls(path, "model"), [])

    def test_no_example_caps_a_text_call_at_a_fixed_small_budget(self) -> None:
        # Reasoning models spend a small budget thinking and return no answer.
        for path in EXAMPLES:
            with self.subTest(example=path.parent.name):
                small = [
                    hit
                    for hit in _literal_keyword_calls(path, "max_tokens")
                    if int(hit.rsplit("=", 1)[1]) < 1024
                ]
                self.assertEqual(small, [])


if __name__ == "__main__":
    unittest.main()
