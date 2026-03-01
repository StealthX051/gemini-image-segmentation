from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Set

import pandas as pd

from .config import EnhancedFairnessConfig
from .dedup import sha256_file


def _image_dir(root: Path) -> Path:
    for name in ("images", "image"):
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return root / "images"


def _iter_image_paths(root: Path) -> Iterable[Path]:
    img_dir = _image_dir(root)
    if not img_dir.is_dir():
        return []
    return sorted(
        p
        for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    )


def _read_manifest_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    ids: Set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip()
        if not token:
            continue
        ids.add(Path(token).stem)
    return ids


def _isic2017_splits(root: Path) -> Dict[str, str]:
    image_dir = root / "image"
    mapping: Dict[str, str] = {}
    for split, fname in (
        ("train", "ISIC-2017_Training_Data_metadata.csv"),
        ("val", "ISIC-2017_Validation_Data_metadata.csv"),
        ("test", "ISIC-2017_Test_v2_Data_metadata.csv"),
    ):
        fpath = image_dir / fname
        if not fpath.exists():
            continue
        with fpath.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                image_id = (row.get("image_id") or "").strip()
                if image_id:
                    mapping[image_id] = split
    return mapping


def _ima_splits(root: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for split, fname in (
        ("train", "train_ima_plusplus.txt"),
        ("val", "val_ima_plusplus.txt"),
        ("test", "test_ima_plusplus.txt"),
    ):
        for image_id in _read_manifest_ids(root / fname):
            mapping[image_id] = split
    return mapping


def _ima_mask_source_map(root: Path) -> Dict[str, str]:
    index_csv = root / "metadata" / "ima_plusplus_index.csv"
    if not index_csv.exists():
        return {}
    try:
        df = pd.read_csv(index_csv)
    except Exception:
        return {}
    out: Dict[str, str] = {}
    if "ISIC_id" not in df.columns or "gt_policy" not in df.columns:
        return out
    for _, row in df.iterrows():
        image_id = str(row.get("ISIC_id", "")).strip()
        policy = str(row.get("gt_policy", "")).strip()
        if image_id:
            out[image_id] = policy or "unknown"
    return out


def _records_for_source(
    *,
    source_name: str,
    root: Path,
    split_map: Dict[str, str] | None = None,
    mask_source_map: Dict[str, str] | None = None,
    default_mask_source: str = "challenge_gt",
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    split_map = split_map or {}
    mask_source_map = mask_source_map or {}

    for image_path in _iter_image_paths(root):
        image_id = image_path.stem
        try:
            digest = sha256_file(image_path)
        except Exception as exc:
            logging.warning("Failed to hash %s: %s", image_path, exc)
            continue
        records.append(
            {
                "dataset_source": source_name,
                "image_id": image_id,
                "image_name": image_path.name,
                "image_path": str(image_path.resolve()),
                "sha256": digest,
                "split": split_map.get(image_id, "unknown"),
                "mask_source": mask_source_map.get(image_id, default_mask_source),
            }
        )
    return records


def _build_source_index(cfg: EnhancedFairnessConfig) -> pd.DataFrame:
    src = cfg.sources

    records: List[Dict[str, object]] = []

    records.extend(
        _records_for_source(
            source_name="interop4074",
            root=src.interop_root,
            default_mask_source="challenge_gt",
        )
    )

    records.extend(
        _records_for_source(
            source_name="isic2016",
            root=src.isic2016_root,
            default_mask_source="challenge_gt",
        )
    )

    records.extend(
        _records_for_source(
            source_name="isic2017",
            root=src.isic2017_root,
            split_map=_isic2017_splits(src.isic2017_root),
            default_mask_source="challenge_gt",
        )
    )

    records.extend(
        _records_for_source(
            source_name="isic2018",
            root=src.isic2018_root,
            default_mask_source="challenge_gt",
        )
    )

    records.extend(
        _records_for_source(
            source_name="ima_plusplus",
            root=src.ima_plusplus_root,
            split_map=_ima_splits(src.ima_plusplus_root),
            mask_source_map=_ima_mask_source_map(src.ima_plusplus_root),
            default_mask_source="single_annotator_only",
        )
    )

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Primary source per SHA: prefer non-interop sources only for provenance if present.
    source_priority = {
        "ima_plusplus": 0,
        "isic2017": 1,
        "isic2018": 2,
        "isic2016": 3,
        "interop4074": 4,
    }
    df["source_priority"] = df["dataset_source"].map(source_priority).fillna(99)

    members = (
        df.groupby("sha256", dropna=False)["dataset_source"]
        .apply(lambda s: sorted(set(str(v) for v in s.tolist())))
        .to_dict()
    )
    primary = (
        df.sort_values(["sha256", "source_priority"])
        .drop_duplicates(subset=["sha256"], keep="first")
        .set_index("sha256")["dataset_source"]
        .to_dict()
    )

    df["dataset_source_primary"] = df["sha256"].map(primary)
    df["dataset_source_memberships_json"] = df["sha256"].map(
        lambda sha: json.dumps(members.get(str(sha), []), sort_keys=True)
    )

    return df.drop(columns=["source_priority"])


def build_or_load_source_index(
    *,
    cache_path: Path,
    cfg: EnhancedFairnessConfig,
) -> pd.DataFrame:
    if cache_path.exists() and not cfg.refresh_source_index:
        try:
            return pd.read_parquet(cache_path)
        except Exception as exc:
            logging.warning("Failed to load cached source index (%s), rebuilding.", exc)

    df = _build_source_index(cfg)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df
