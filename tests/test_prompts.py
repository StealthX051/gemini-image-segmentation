import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from gemini_segmentation.prompts import (
    PROMPTS_DESC,
    PROMPTS_NEGATION,
    PromptFamily,
    SCHEMA_PREAMBLE,
    build_prompt,
)


class BuildPromptTests(unittest.TestCase):
    def test_builds_desc_with_negation(self) -> None:
        prompt = build_prompt("polyp", PromptFamily.DESC_NEG_V1)
        self.assertTrue(prompt.startswith(SCHEMA_PREAMBLE))
        self.assertIn(PROMPTS_DESC["polyp"], prompt)
        self.assertIn(PROMPTS_NEGATION["polyp"], prompt)

    def test_rejects_unknown_family(self) -> None:
        with self.assertRaises(ValueError):
            build_prompt("polyp", "unknown_family")

    def test_rejects_unknown_task(self) -> None:
        with self.assertRaises(KeyError):
            build_prompt("unknown_task", PromptFamily.DESC_V1)


if __name__ == "__main__":
    unittest.main()
