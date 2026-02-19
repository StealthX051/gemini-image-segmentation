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
    build_prompt_for_provider,
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

    def test_provider_prompt_preserves_gemini_schema(self) -> None:
        prompt = build_prompt_for_provider("polyp", PromptFamily.DESC_NEG_V1, "gemini")
        self.assertTrue(prompt.prompt.startswith(SCHEMA_PREAMBLE))

    def test_provider_prompt_for_moondream_is_target_only(self) -> None:
        prompt = build_prompt_for_provider("optic_disc_cup", PromptFamily.LABEL_V1, "moondream")
        self.assertEqual(prompt.prompt, "optic disc")
        self.assertEqual(list(prompt.targets or ()), ["optic disc", "optic cup"])
        self.assertNotIn("JSON", prompt.prompt)

    def test_provider_prompt_for_replicate_uses_instructions(self) -> None:
        prompt = build_prompt_for_provider("polyp", PromptFamily.LABEL_V1, "replicate")
        self.assertIn("colorectal polyp", prompt.instructions)
        self.assertEqual(prompt.prompt, "Segment the colorectal polyp.")
        self.assertNotIn("JSON", prompt.prompt)

    def test_provider_prompt_for_replicate_varies_by_family(self) -> None:
        label_prompt = build_prompt_for_provider("polyp", PromptFamily.LABEL_V1, "replicate")
        desc_prompt = build_prompt_for_provider("polyp", PromptFamily.DESC_V1, "replicate")
        desc_neg_prompt = build_prompt_for_provider("polyp", PromptFamily.DESC_NEG_V1, "replicate")

        label_instruction = label_prompt.instructions["colorectal polyp"]
        desc_instruction = desc_prompt.instructions["colorectal polyp"]
        desc_neg_instruction = desc_neg_prompt.instructions["colorectal polyp"]

        self.assertNotEqual(label_instruction, desc_instruction)
        self.assertTrue(desc_neg_instruction.startswith(desc_instruction))
        self.assertIn("Exclude:", desc_neg_instruction)


if __name__ == "__main__":
    unittest.main()
