import sys
import tempfile
import textwrap
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from gemini_segmentation.config import load_preset, resolve_preset_name


class ResolvePresetNameTests(unittest.TestCase):
    def test_returns_base_for_legacy(self) -> None:
        self.assertEqual(resolve_preset_name("polyp", None), "polyp")
        self.assertEqual(resolve_preset_name("polyp", "legacy"), "polyp")

    def test_appends_branch_suffix(self) -> None:
        self.assertEqual(resolve_preset_name("polyp", "hybrid"), "polyp_hybrid")

    def test_does_not_duplicate_suffix(self) -> None:
        self.assertEqual(resolve_preset_name("polyp_hybrid", "hybrid"), "polyp_hybrid")


class LoadPresetWithBranchTests(unittest.TestCase):
    def test_loads_branch_specific_entry(self) -> None:
        yaml_content = textwrap.dedent(
            """
            base:
              prompt_text: base text
            base_hybrid:
              prompt_text: hybrid text
            """
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_path = Path(tmp_dir) / "presets.yaml"
            cfg_path.write_text(yaml_content)
            preset_name = resolve_preset_name("base", "hybrid")
            preset_cfg = load_preset(cfg_path, preset_name)
            self.assertEqual(preset_cfg["prompt_text"], "hybrid text")


if __name__ == "__main__":
    unittest.main()
