import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "prepare_ima_plusplus.py"

spec = importlib.util.spec_from_file_location("prepare_ima_plusplus", SCRIPT_PATH)
prepare_module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = prepare_module
spec.loader.exec_module(prepare_module)


DEFAULTS = {
    "segs_zip": None,
    "seg_metadata": None,
    "seg_metadata_multiannotator": None,
    "img_metadata": None,
    "train_csv": None,
    "val_csv": None,
    "test_csv": None,
    "download_zenodo": False,
    "download_split_csvs": False,
    "download_multiannotator_metadata": True,
    "overwrite_downloads": False,
    "segs_zip_url": prepare_module.DEFAULT_SEGS_URL,
    "seg_metadata_url": prepare_module.DEFAULT_SEG_METADATA_URL,
    "seg_metadata_multiannotator_url": prepare_module.DEFAULT_SEG_METADATA_MULTI_URL,
    "img_metadata_url": prepare_module.DEFAULT_IMG_METADATA_URL,
    "train_csv_url": prepare_module.DEFAULT_TRAIN_URL,
    "val_csv_url": prepare_module.DEFAULT_VAL_URL,
    "test_csv_url": prepare_module.DEFAULT_TEST_URL,
    "extract_masks": False,
    "force_extract": False,
    "download_images": False,
    "download_images_mode": "api",
    "isic_api_url_template": prepare_module.DEFAULT_ISIC_API_URL_TEMPLATE,
    "isic_api_workers": 4,
    "isic_api_retries": 2,
    "isic_api_timeout_sec": 5.0,
    "isic_api_backoff_sec": 0.01,
    "isic_api_skip_existing": True,
    "isic_api_fail_on_errors": True,
    "isic_download_template": "isic image download --search isic_id:{isic_id} {output_dir}",
    "write_split_manifests": False,
    "log_level": "INFO",
}


def _args(*, raw_root: Path, dataset_root: Path, **overrides):
    payload = dict(DEFAULTS)
    payload.update(overrides)
    payload["raw_root"] = str(raw_root)
    payload["dataset_root"] = str(dataset_root)
    return argparse.Namespace(**payload)


def _write_mask(path: Path, value: int) -> None:
    arr = np.full((4, 4), value, dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def _write_image(path: Path, value: int) -> None:
    arr = np.full((4, 4, 3), value, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def test_prepare_ima_plusplus_staple_first_and_mask_retention(tmp_path):
    raw_root = tmp_path / "raw"
    dataset_root = tmp_path / "dataset"

    metadata_dir = raw_root / "metadata"
    masks_raw_dir = raw_root / "masks_raw"
    images_raw_dir = raw_root / "images_raw"
    metadata_dir.mkdir(parents=True)
    masks_raw_dir.mkdir(parents=True)
    images_raw_dir.mkdir(parents=True)

    _write_mask(masks_raw_dir / "ISIC_001_ann1.png", 10)
    _write_mask(masks_raw_dir / "ISIC_001_ann2.png", 20)
    _write_mask(masks_raw_dir / "ISIC_001_ST_ST_ST_ST.png", 30)
    _write_mask(masks_raw_dir / "ISIC_001_MV_MV_MV_MV.png", 40)
    _write_mask(masks_raw_dir / "ISIC_002_ann1.png", 50)

    _write_image(images_raw_dir / "ISIC_001.jpg", 200)
    _write_image(images_raw_dir / "ISIC_002.jpg", 210)

    seg_df = pd.DataFrame(
        [
            {
                "ISIC_id": "ISIC_001",
                "seg_filename": "ISIC_001_ann1.png",
                "annotator": "a1",
                "tool": "tool_a",
                "skill_level": "expert",
            },
            {
                "ISIC_id": "ISIC_001",
                "seg_filename": "ISIC_001_ann2.png",
                "annotator": "a2",
                "tool": "tool_b",
                "skill_level": "novice",
            },
            {
                "ISIC_id": "ISIC_001",
                "seg_filename": "ISIC_001_ST_ST_ST_ST.png",
                "annotator": "consensus",
                "tool": "staple",
                "skill_level": "n/a",
            },
            {
                "ISIC_id": "ISIC_001",
                "seg_filename": "ISIC_001_MV_MV_MV_MV.png",
                "annotator": "consensus",
                "tool": "mv",
                "skill_level": "n/a",
            },
            {
                "ISIC_id": "ISIC_002",
                "seg_filename": "ISIC_002_ann1.png",
                "annotator": "a3",
                "tool": "tool_c",
                "skill_level": "intermediate",
            },
        ]
    )
    seg_df.to_csv(metadata_dir / "seg_metadata.csv", index=False)

    img_df = pd.DataFrame(
        [
            {"ISIC_id": "ISIC_001", "img_filename": "ISIC_001.jpg"},
            {"ISIC_id": "ISIC_002", "img_filename": "ISIC_002.jpg"},
        ]
    )
    img_df.to_csv(metadata_dir / "img_metadata.csv", index=False)

    args = _args(raw_root=raw_root, dataset_root=dataset_root)
    prepare_module.prepare_ima_plusplus(args)

    manifest = dataset_root / "master_imagelist_ima_plusplus.txt"
    assert manifest.exists()
    assert manifest.read_text(encoding="utf-8").splitlines() == ["ISIC_001.jpg", "ISIC_002.jpg"]

    canonical_1 = np.asarray(Image.open(dataset_root / "masks" / "ISIC_001.jpg"))
    canonical_2 = np.asarray(Image.open(dataset_root / "masks" / "ISIC_002.jpg"))
    assert int(canonical_1[0, 0]) == 30  # STAPLE selected for ISIC_001
    assert int(canonical_2[0, 0]) == 50  # single annotator fallback for ISIC_002

    index_jsonl = dataset_root / "metadata" / "ima_plusplus_index.jsonl"
    rows = [json.loads(line) for line in index_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 2

    row_1 = next(row for row in rows if row["ISIC_id"] == "ISIC_001")
    assert row_1["gt_policy"] == "consensus_staple"
    assert row_1["n_masks"] == 4
    assert row_1["n_annotator_masks"] == 2
    assert row_1["staple_mask_path"] is not None
    assert row_1["mv_mask_path"] is not None
    assert len(row_1["all_mask_paths"]) == 4
    assert len(row_1["all_mask_metadata"]) == 4


def test_prepare_ima_plusplus_fails_for_multi_annotator_without_consensus(tmp_path):
    raw_root = tmp_path / "raw"
    dataset_root = tmp_path / "dataset"

    metadata_dir = raw_root / "metadata"
    masks_raw_dir = raw_root / "masks_raw"
    images_raw_dir = raw_root / "images_raw"
    metadata_dir.mkdir(parents=True)
    masks_raw_dir.mkdir(parents=True)
    images_raw_dir.mkdir(parents=True)

    _write_mask(masks_raw_dir / "ISIC_100_ann1.png", 10)
    _write_mask(masks_raw_dir / "ISIC_100_ann2.png", 20)
    _write_image(images_raw_dir / "ISIC_100.jpg", 120)

    seg_df = pd.DataFrame(
        [
            {
                "ISIC_id": "ISIC_100",
                "seg_filename": "ISIC_100_ann1.png",
                "annotator": "a1",
                "tool": "tool_x",
                "skill_level": "expert",
            },
            {
                "ISIC_id": "ISIC_100",
                "seg_filename": "ISIC_100_ann2.png",
                "annotator": "a2",
                "tool": "tool_y",
                "skill_level": "novice",
            },
        ]
    )
    seg_df.to_csv(metadata_dir / "seg_metadata.csv", index=False)

    img_df = pd.DataFrame([{"ISIC_id": "ISIC_100", "img_filename": "ISIC_100.jpg"}])
    img_df.to_csv(metadata_dir / "img_metadata.csv", index=False)

    args = _args(raw_root=raw_root, dataset_root=dataset_root)
    with pytest.raises(ValueError, match="Integrity error: multi-annotator image missing consensus masks"):
        prepare_module.prepare_ima_plusplus(args)


def test_prepare_ima_plusplus_split_image_column_and_multiannotator_metadata_copy(tmp_path):
    raw_root = tmp_path / "raw"
    dataset_root = tmp_path / "dataset"

    metadata_dir = raw_root / "metadata"
    masks_raw_dir = raw_root / "masks_raw"
    images_raw_dir = raw_root / "images_raw"
    metadata_dir.mkdir(parents=True)
    masks_raw_dir.mkdir(parents=True)
    images_raw_dir.mkdir(parents=True)

    _write_mask(masks_raw_dir / "ISIC_123_ann1.png", 7)
    _write_image(images_raw_dir / "ISIC_123.jpg", 190)

    pd.DataFrame(
        [
            {
                "ISIC_id": "ISIC_123",
                "seg_filename": "ISIC_123_ann1.png",
                "annotator": "a1",
                "tool": "tool_a",
                "skill_level": "expert",
            }
        ]
    ).to_csv(metadata_dir / "seg_metadata.csv", index=False)

    pd.DataFrame([{"ISIC_id": "ISIC_123", "img_filename": "ISIC_123.jpg"}]).to_csv(
        metadata_dir / "img_metadata.csv", index=False
    )
    pd.DataFrame([{"ISIC_id": "ISIC_123", "seg_filename": "ISIC_123_ann1.png"}]).to_csv(
        metadata_dir / "seg_metadata_multiannotator_subset.csv", index=False
    )

    split_df = pd.DataFrame([{"image": "ISIC_123.JPG"}])
    split_df.to_csv(metadata_dir / "train.csv", index=False)
    split_df.to_csv(metadata_dir / "val.csv", index=False)
    split_df.to_csv(metadata_dir / "test.csv", index=False)

    args = _args(raw_root=raw_root, dataset_root=dataset_root, write_split_manifests=True)
    prepare_module.prepare_ima_plusplus(args)

    for split_name in ("train", "val", "test"):
        split_manifest = dataset_root / f"{split_name}_ima_plusplus.txt"
        assert split_manifest.exists()
        assert split_manifest.read_text(encoding="utf-8").splitlines() == ["ISIC_123.jpg"]

    assert (dataset_root / "metadata" / "seg_metadata_multiannotator_subset.csv").exists()
