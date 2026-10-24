"""BaseStrategy.build_image_prompt sends the challenge's own image instruction."""

from __future__ import annotations

import unittest

from base_strategy import BaseStrategy, ChallengeContext

GENERIC = "Generate an original image from a prompt suitable for a product launch poster draft."
INSTRUCTION = (
    "Create a cinematic launch poster of a futuristic electric motorcycle on a rainy "
    "neon street at night, high contrast, detailed reflections, no text."
)
NOTES = "Single coherent scene. Keep composition clean and avoid logo/watermark text."


class BuildImagePromptTests(unittest.TestCase):
    def test_instruction_in_rules_leads_the_prompt(self) -> None:
        # How python_simple, langgraph and crewai build the image context:
        # the generic description as challenge_text, prompt + notes in rules.
        ctx = ChallengeContext(
            challenge_type="image-generate",
            challenge_text=GENERIC,
            description=GENERIC,
            rules=f"{INSTRUCTION}\n{NOTES}",
        )

        prompt = BaseStrategy().build_image_prompt(ctx)

        self.assertTrue(prompt.startswith(INSTRUCTION))
        self.assertIn(NOTES, prompt)
        self.assertIn(f"Context: {GENERIC}", prompt)
        self.assertIn("Output constraints:", prompt)

    def test_instruction_as_challenge_text_is_kept(self) -> None:
        # How python_reference builds it: the instruction is challenge_text.
        ctx = ChallengeContext(
            challenge_type="image-edit",
            challenge_text=INSTRUCTION,
            description=GENERIC,
        )

        prompt = BaseStrategy().build_image_prompt(ctx)

        self.assertTrue(prompt.startswith(INSTRUCTION))
        self.assertNotIn("Context:", prompt)

    def test_rules_already_containing_the_headline_are_not_repeated(self) -> None:
        ctx = ChallengeContext(
            challenge_type="image-edit",
            challenge_text=INSTRUCTION,
            rules=f"{INSTRUCTION}\n{NOTES}",
        )

        prompt = BaseStrategy().build_image_prompt(ctx)

        self.assertEqual(prompt.count(INSTRUCTION), 1)

    def test_empty_context_falls_back_to_default(self) -> None:
        prompt = BaseStrategy().build_image_prompt(ChallengeContext(challenge_type="image"))
        self.assertTrue(prompt.startswith("Generate a clear image response for the task."))


if __name__ == "__main__":
    unittest.main()
