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


if __name__ == "__main__":
    unittest.main()
