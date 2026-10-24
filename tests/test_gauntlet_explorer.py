"""Unit tests for the Gauntlet Explorer modules (stdlib unittest; no network)."""

from __future__ import annotations

import base64
import codecs
import json
import sys
import unittest
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from base_strategy import ChallengeContext  # noqa: E402
from gauntlet import answer, decoders, pngtools  # noqa: E402
from gauntlet.clock import Clock  # noqa: E402
from gauntlet.explorer import ArgContext, ToolInfo, classify, mine_text, synthesize_args  # noqa: E402
from gauntlet.llm import parse_json_loose, safe_fn_name  # noqa: E402
from gauntlet.models import rank_image_models, rank_text_models  # noqa: E402
from gauntlet.react import run_python  # noqa: E402
from my_strategy import MyStrategy  # noqa: E402

SECRET = "The vault key is ORION and the flag is FLAG{black_box_cracked}"


class DecoderTests(unittest.TestCase):
    def assertDecodes(self, encoded: str, expected_fragment: str = "FLAG{black_box_cracked}") -> None:
        best = decoders.auto_decode(encoded)[0]
        self.assertIn(expected_fragment, best.text, msg=f"chain={best.chain}")

    def test_layered_encodings(self) -> None:
        b64 = base64.b64encode(SECRET.encode()).decode()
        self.assertDecodes(b64)
        self.assertDecodes(b64.encode().hex())
        self.assertDecodes(base64.b64encode(codecs.encode(SECRET, "rot_13").encode()).decode())
        self.assertDecodes(base64.b32encode(SECRET.encode()).decode())
        self.assertDecodes(base64.b64encode(zlib.compress(SECRET.encode())).decode())
        self.assertDecodes(bytes(b ^ 0x2A for b in SECRET.encode()).hex())

    def test_classical_ciphers(self) -> None:
        self.assertDecodes(codecs.encode(SECRET, "rot_13"))
        self.assertDecodes(decoders.caesar(SECRET, 7))
        self.assertDecodes(decoders.t_atbash(SECRET))
        self.assertDecodes(decoders.t_rot47(SECRET))
        self.assertDecodes(SECRET[::-1])
        self.assertEqual(decoders.auto_decode("WECRLTEERDSOEEFEAOCAIVDEN")[0].text, "WEAREDISCOVEREDFLEEATONCE")

    def test_other_formats(self) -> None:
        self.assertEqual(decoders.auto_decode(".... .. -.. -.. . -. / -.. --- --- .-.")[0].text, "HIDDEN DOOR")
        self.assertEqual(decoders.auto_decode(" ".join(f"{b:08b}" for b in b"use the unlock tool"))[0].text, "use the unlock tool")
        self.assertEqual(decoders.auto_decode(" ".join(str(b) for b in b"call vault.open"))[0].text, "call vault.open")

    def test_keyed(self) -> None:
        enc = decoders.vigenere_encrypt(SECRET, "orion")
        self.assertIn("FLAG{", decoders.auto_decode(enc, keys=["orion"])[0].text)

    def test_plain_text_left_alone(self) -> None:
        self.assertEqual(decoders.auto_decode("Find the hidden tool and call it.")[0].chain, [])
        self.assertEqual(decoders.decode_findings("Find the hidden tool and call it."), [])

    def test_rotated_flag_shape_is_not_a_flag(self) -> None:
        self.assertEqual(decoders.find_flags("SYNT{oynpx}"), [])
        self.assertEqual(decoders.find_flags("x NVIDIA{abc_123} y"), ["NVIDIA{abc_123}"])

    def test_findings_inside_prose(self) -> None:
        text = "The guard whispered " + base64.b64encode(b"password is swordfish").decode() + " and left."
        found = decoders.decode_findings(text)
        self.assertTrue(any("swordfish" in f["text"] for f in found))


class PngTests(unittest.TestCase):
    def test_metadata_trailing_and_lsb(self) -> None:
        w, h = 40, 30
        rgb = bytearray(bytes((i * 37) % 256 for i in range(w * h * 3)))
        msg = b"FLAG{lsb_found_it}\x00"
        bits = [(b >> (7 - i)) & 1 for b in msg for i in range(8)]
        for i, bit in enumerate(bits):
            rgb[i] = (rgb[i] & 0xFE) | bit
        png = pngtools.make_png(w, h, bytes(rgb), {"Comment": "key=ORION"}) + b"appended: use vault.open"
        report = pngtools.inspect_image(png)
        self.assertEqual((report["width"], report["height"]), (40, 30))
        self.assertEqual(report["text_chunks"]["Comment"], "key=ORION")
        self.assertIn("FLAG{lsb_found_it}", report["flags"])
        self.assertTrue(any("vault.open" in s for s in report["trailing_strings"]))


class ExplorerTests(unittest.TestCase):
    def test_classify(self) -> None:
        cases = {
            "arena.get_challenge": "challenge", "arena.clues.list": "clue_list", "arena.clues.get": "clue_get",
            "arena.time_remaining": "timer", "arena.image.submit_edit": "submit", "arena.image.broadcast_thought": "broadcast",
            "image_edit": "image_edit", "image_generate": "image_generate", "image_analyze": "image_analyze",
            "reset_session": "danger", "checkAnswer": "checker", "vault.unlock": "decrypt", "web_search": "search",
        }
        for name, kind in cases.items():
            self.assertEqual(classify(name), kind, name)
        self.assertEqual(classify("oracle", "Submits your final answer. This cannot be undone."), "submit")

    def test_synthesize_args(self) -> None:
        tool = ToolInfo("vault.open", "", {"type": "object", "properties": {"key": {"type": "string"}, "agent_id": {"type": "string"},
                                                                              "mode": {"enum": ["fast", "slow"]}},
                                           "required": ["key", "agent_id"]}, "decrypt")
        ctx = ArgContext(agent_id="team-x", keys=["ORION"])
        self.assertEqual(synthesize_args(tool, ctx), {"key": "ORION", "agent_id": "team-x", "mode": "fast"})
        needs_id = ToolInfo("doors.open", "", {"type": "object", "properties": {"door_id": {"type": "string"}}, "required": ["door_id"]}, "unknown")
        self.assertIsNone(synthesize_args(needs_id, ArgContext(agent_id="a")))

    def test_mine_text(self) -> None:
        hints = mine_text('{"challenge_type": "x", "image_uri": "y"} Archivist: use vault.open with key=ORION.', set())
        self.assertIn("vault.open", hints.tool_names)
        self.assertNotIn("challenge_type", hints.tool_names)
        self.assertNotIn("image_uri", hints.tool_names)
        self.assertIn("ORION", hints.keys)


class CatalogTests(unittest.TestCase):
    def test_tool_entries(self) -> None:
        from gauntlet.explorer import _tool_entries

        payload = {"tools": [{"name": "vault.open", "description": "x", "input_schema": {}}, {"name": "a.b", "description": ""}],
                   "note": "ok"}
        self.assertEqual([t["name"] for t in _tool_entries(payload)], ["vault.open", "a.b"])
        self.assertEqual([t["name"] for t in _tool_entries({"available": ["arena.get_challenge", "hidden_tool"]})],
                         ["arena.get_challenge", "hidden_tool"])
        # Practice format: names only, under "tools"; bare words count there.
        self.assertEqual([t["name"] for t in _tool_entries({"tools": ["arena.clues.get", "oracle"], "count": 2})],
                         ["arena.clues.get", "oracle"])
        self.assertEqual(_tool_entries({"notes": ["oracle"]}), [])


class AnswerTests(unittest.TestCase):
    def test_extract(self) -> None:
        self.assertEqual(answer.extract_answer("<think>hmm ANSWER: no</think>\nANSWER: Bob\nbecause"), "Bob")
        self.assertEqual(answer.extract_answer("**Final answer:** `42`"), "42")
        spec = answer.infer_format('Return JSON with keys "a" and "b".')
        self.assertTrue(spec.json)
        self.assertEqual(json.loads(answer.extract_answer('ANSWER:\n```json\n{"a": 1, "b": 2}\n```', spec)), {"a": 1, "b": 2})

    def test_enforce(self) -> None:
        spec = answer.infer_format("Answer in UPPERCASE, single word.")
        self.assertEqual(answer.enforce("orion.", spec)[0], "ORION")
        spec = answer.infer_format("Respond with the integer only.")
        self.assertEqual(answer.enforce("The total is 1,204 apples", spec)[0], "1204")
        spec = answer.infer_format("Answer yes or no.")
        self.assertEqual(answer.enforce("Yes, because the clues agree", spec)[0], "Yes")

    def test_shared_prefix(self) -> None:
        spec = answer.infer_format("Return only the final order as comma-separated team names.")
        self.assertEqual(answer.enforce("Team Alpha, Team Bravo, Team Echo", spec)[0], "Alpha, Bravo, Echo")
        self.assertEqual(answer.enforce("E3, E2, E5", spec)[0], "E3, E2, E5")
        self.assertEqual(answer.enforce("New York, New Jersey", spec)[0], "New York, New Jersey")

    def test_invalid(self) -> None:
        for bad in ("", "unknown", "<toolcall>x", "Reasoning: x"):
            self.assertTrue(answer.is_invalid(bad))
        self.assertFalse(answer.is_invalid("Bob"))


class ModelTests(unittest.TestCase):
    ROSTER = ["gpt-oss-20b", "nemotron-3-super", "nemotron-3.5-lightning", "qwen3.7-flash", "gpt-5.2", "gpt-5.4",
              "gemini-2.5-pro", "gemini-3.1-flash-image", "gpt-5-image-mini"]

    def test_text_ranking_excludes_image_models(self) -> None:
        ranked = [m.name for m in rank_text_models(self.ROSTER, max_time_s=120)]
        self.assertNotIn("gemini-3.1-flash-image", ranked)
        self.assertEqual(ranked[0], "gpt-5.4")

    def test_image_ranking(self) -> None:
        self.assertEqual(set(rank_image_models(self.ROSTER)), {"gemini-3.1-flash-image", "gpt-5-image-mini"})

    def test_strategy_hooks(self) -> None:
        strategy = MyStrategy()
        text_ctx = ChallengeContext(challenge_type="logic", max_time_s=120, difficulty="hard")
        self.assertEqual(strategy.pick_model("solve", strategy.rank_models(text_ctx, self.ROSTER), text_ctx), "gpt-5.4")
        img_ctx = ChallengeContext(challenge_type="image_edit", image_url="data:image/png;base64,AA==")
        self.assertIn(strategy.rank_models(img_ctx, self.ROSTER)[0], {"gemini-3.1-flash-image", "gpt-5-image-mini"})


class MiscTests(unittest.TestCase):
    def test_safe_fn_name(self) -> None:
        taken: set[str] = set()
        self.assertEqual(safe_fn_name("arena.clues.get", taken), "arena__clues__get")
        self.assertEqual(safe_fn_name("arena.clues.get", taken), "arena__clues__get_2")

    def test_parse_json_loose(self) -> None:
        self.assertEqual(parse_json_loose('Sure! {"tool": "x", "args": {"a": "}"}} done'), {"tool": "x", "args": {"a": "}"}})

    def test_run_python(self) -> None:
        self.assertEqual(run_python("print(sum(range(10)))").strip(), "45")
        self.assertIn("not allowed", run_python("import os; print(os.environ)"))

    def test_clock(self) -> None:
        clock = Clock(60, 60)
        self.assertLess(clock.left(), 60)
        clock.sync(10)
        self.assertLess(clock.left(), 10)


if __name__ == "__main__":
    unittest.main()
