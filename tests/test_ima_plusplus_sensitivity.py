import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "analyze_ima_plusplus_sensitivity.py"

spec = importlib.util.spec_from_file_location("analyze_ima_plusplus_sensitivity", SCRIPT_PATH)
analyze_module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = analyze_module
spec.loader.exec_module(analyze_module)


def _write_mask(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array.astype(np.uint8), mode="L").save(path)


def test_ima_plusplus_sensitivity_outputs(tmp_path):
    dataset_root = tmp_path / "dataset"
    run_dir = tmp_path / "results" / "ima_plusplus" / "gemini-2.5-flash" / "label_v1-aaaa" / "run_001"

    (dataset_root / "images").mkdir(parents=True)
    (dataset_root / "masks").mkdir(parents=True)
    (dataset_root / "masks_all").mkdir(parents=True)
    (dataset_root / "metadata").mkdir(parents=True)
    (run_dir / "masks").mkdir(parents=True)

    # Image 1 artifacts
    _write_mask(dataset_root / "masks_all" / "ISIC_001_MV_MV_MV_MV.png", np.array([[255, 0], [0, 0]], dtype=np.uint8))
    _write_mask(dataset_root / "masks_all" / "ISIC_001_ann1.png", np.array([[255, 0], [0, 0]], dtype=np.uint8))
    _write_mask(dataset_root / "masks_all" / "ISIC_001_ann2.png", np.array([[255, 255], [255, 255]], dtype=np.uint8))
    _write_mask(run_dir / "masks" / "ISIC_001.jpg", np.array([[255, 0], [0, 0]], dtype=np.uint8))

    # Image 2 artifacts
    _write_mask(dataset_root / "masks_all" / "ISIC_002_MV_MV_MV_MV.png", np.array([[0, 0], [0, 0]], dtype=np.uint8))
    _write_mask(dataset_root / "masks_all" / "ISIC_002_ann1.png", np.array([[255, 0], [0, 0]], dtype=np.uint8))
    _write_mask(run_dir / "masks" / "ISIC_002.jpg", np.array([[0, 0], [0, 0]], dtype=np.uint8))

    index_rows = [
        {
            "ISIC_id": "ISIC_001",
            "image_path": "images/ISIC_001.jpg",
            "gt_mask_path": "masks/ISIC_001.jpg",
            "gt_policy": "consensus_staple",
            "all_mask_paths": [
                "masks_all/ISIC_001_MV_MV_MV_MV.png",
                "masks_all/ISIC_001_ann1.png",
                "masks_all/ISIC_001_ann2.png",
            ],
            "all_mask_metadata": [
                {"mask_path": "masks_all/ISIC_001_MV_MV_MV_MV.png", "mask_kind": "consensus_mv"},
                {
                    "mask_path": "masks_all/ISIC_001_ann1.png",
                    "mask_kind": "annotator",
                    "annotator": "a1",
                    "tool": "tool_a",
                    "skill_level": "expert",
                },
                {
                    "mask_path": "masks_all/ISIC_001_ann2.png",
                    "mask_kind": "annotator",
                    "annotator": "a2",
                    "tool": "tool_b",
                    "skill_level": "novice",
                },
            ],
            "staple_mask_path": None,
            "mv_mask_path": "masks_all/ISIC_001_MV_MV_MV_MV.png",
            "n_masks": 3,
            "n_annotator_masks": 2,
        },
        {
            "ISIC_id": "ISIC_002",
            "image_path": "images/ISIC_002.jpg",
            "gt_mask_path": "masks/ISIC_002.jpg",
            "gt_policy": "consensus_mv",
            "all_mask_paths": [
                "masks_all/ISIC_002_MV_MV_MV_MV.png",
                "masks_all/ISIC_002_ann1.png",
            ],
            "all_mask_metadata": [
                {"mask_path": "masks_all/ISIC_002_MV_MV_MV_MV.png", "mask_kind": "consensus_mv"},
                {
                    "mask_path": "masks_all/ISIC_002_ann1.png",
                    "mask_kind": "annotator",
                    "annotator": "a3",
                    "tool": "tool_a",
                    "skill_level": "intermediate",
                },
            ],
            "staple_mask_path": None,
            "mv_mask_path": "masks_all/ISIC_002_MV_MV_MV_MV.png",
            "n_masks": 2,
            "n_annotator_masks": 1,
        },
    ]

    index_path = dataset_root / "metadata" / "ima_plusplus_index.jsonl"
    index_path.write_text("\n".join(json.dumps(row) for row in index_rows) + "\n", encoding="utf-8")

    run_config = {
        "dataset_root": str(dataset_root),
        "success_threshold": 0.5,
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config), encoding="utf-8")

    args = argparse.Namespace(
        run_dir=str(run_dir),
        dataset_root=str(dataset_root),
        index_path=None,
        success_threshold=None,
        log_level="INFO",
    )

    analyze_module.analyze_sensitivity(args)

    out_dir = run_dir / "ima_plusplus_sensitivity"
    metrics_mv = pd.read_csv(out_dir / "metrics_mv.csv")
    metrics_ann = pd.read_csv(out_dir / "metrics_annotators.csv")
    per_image = pd.read_csv(out_dir / "per_image_annotator_summary.csv")
    summary_overall = pd.read_csv(out_dir / "summary_overall.csv")
    summary_tool = pd.read_csv(out_dir / "summary_by_tool.csv")
    summary_skill = pd.read_csv(out_dir / "summary_by_skill_level.csv")

    assert len(metrics_mv) == 2
    assert len(metrics_ann) == 3
    assert len(per_image) == 2
    assert set(summary_overall["comparison_set"]) == {"mv_consensus", "annotators"}
    assert set(summary_tool["tool"]) == {"tool_a", "tool_b"}
    assert set(summary_skill["skill_level"]) == {"expert", "intermediate", "novice"}

    img1_ann = metrics_ann[metrics_ann["image_name"] == "ISIC_001.jpg"]
    assert img1_ann["iou"].max() == 1.0
    assert img1_ann["iou"].min() < 1.0
