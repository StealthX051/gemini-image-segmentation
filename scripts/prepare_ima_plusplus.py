#!/usr/bin/env python3
"""Prepare IMA++ into CLI-compatible dataset layout.

This script materializes:
- canonical dataset root with images/, masks/, master_imagelist_ima_plusplus.txt
- full-mask retention under masks_all/
- metadata index files for downstream sensitivity analyses

Canonical GT selection is deterministic and consensus-first:
1) STAPLE consensus mask
2) Majority-vote consensus mask
3) Single annotator mask (only when exactly one annotator mask exists)
"""

from __future__ import annotations

import argparse
import filecmp
import json
import logging
import re
import shlex
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen, urlretrieve

import pandas as pd


STAPLE_PATTERN = re.compile(r"_ST_ST_ST_ST\.png$", re.IGNORECASE)
MV_PATTERN = re.compile(r"_MV_MV_MV_MV\.png$", re.IGNORECASE)

MASK_FILENAME_COLUMNS = (
    "seg_filename",
    "seg_file_name",
    "mask_filename",
    "filename",
    "seg_name",
    "seg_path",
    "mask_path",
    "path",
)
IMAGE_FILENAME_COLUMNS = (
    "image",
    "img_filename",
    "image_filename",
    "filename",
    "img_name",
    "image_name",
    "file_name",
)

# Zenodo DOI 10.5281/zenodo.14201692 currently resolves to record 14201693.
DEFAULT_SEGS_URL = "https://zenodo.org/records/14201693/files/segs.zip?download=1"
DEFAULT_SEG_METADATA_URL = "https://zenodo.org/records/14201693/files/seg_metadata.csv?download=1"
DEFAULT_IMG_METADATA_URL = "https://zenodo.org/records/14201693/files/img_metadata.csv?download=1"
DEFAULT_SEG_METADATA_MULTI_URL = (
    "https://zenodo.org/records/14201693/files/seg_metadata_multiannotator_subset.csv?download=1"
)
DEFAULT_TRAIN_URL = "https://zenodo.org/records/14201693/files/train.csv?download=1"
DEFAULT_VAL_URL = "https://zenodo.org/records/14201693/files/val.csv?download=1"
DEFAULT_TEST_URL = "https://zenodo.org/records/14201693/files/test.csv?download=1"
DEFAULT_ISIC_API_URL_TEMPLATE = "https://api.isic-archive.com/api/v2/images/{isic_id}"


@dataclass
class MaskEntry:
    isic_id: str
    mask_filename: str
    source_path: Path
    mask_kind: str
    metadata: Dict[str, Any]


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _normalize_isic_token(value: str) -> str:
    token = value.strip().upper().replace("-", "_")
    return token


def _normalize_isic_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    return _normalize_isic_token(str(value))


def _sanitize_for_json(payload: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    for key, value in payload.items():
        if pd.isna(value):
            sanitized[str(key)] = None
        elif isinstance(value, (str, int, float, bool)) or value is None:
            sanitized[str(key)] = value
        else:
            sanitized[str(key)] = str(value)
    return sanitized


def _find_column(df: pd.DataFrame, candidates: Sequence[str], *, required: bool) -> Optional[str]:
    by_lower = {str(col).strip().lower(): str(col) for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    if required:
        raise ValueError(
            f"Missing required column. Tried candidates: {', '.join(candidates)}. "
            f"Available columns: {', '.join(map(str, df.columns))}"
        )
    return None


def _classify_mask(mask_filename: str) -> str:
    name = Path(mask_filename).name
    if STAPLE_PATTERN.search(name):
        return "consensus_staple"
    if MV_PATTERN.search(name):
        return "consensus_mv"
    return "annotator"


def _download_file(url: str, destination: Path, *, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        logging.info("Skipping download (already exists): %s", destination)
        return
    logging.info("Downloading %s -> %s", url, destination)
    urlretrieve(url, destination)


def _extract_zip(archive_path: Path, output_dir: Path, *, force: bool) -> None:
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    existing_files = list(output_dir.rglob("*")) if output_dir.exists() else []
    if existing_files and not force:
        logging.info("Skipping extraction; files already exist in %s", output_dir)
        return

    if force and output_dir.exists():
        logging.info("Clearing extracted mask directory before extraction: %s", output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Extracting %s -> %s", archive_path, output_dir)
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(output_dir)


def _build_mask_source_lookup(masks_raw_dir: Path) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    collisions: Dict[str, List[Path]] = {}
    for path in masks_raw_dir.rglob("*"):
        if not path.is_file():
            continue
        key = path.name
        if key in lookup and lookup[key] != path:
            collisions.setdefault(key, [lookup[key]]).append(path)
            continue
        lookup[key] = path

    if collisions:
        samples = []
        for key, values in list(collisions.items())[:5]:
            samples.append(f"{key}: {', '.join(str(v) for v in values)}")
        raise ValueError(
            "Duplicate mask basenames found under masks_raw_dir; cannot resolve uniquely. "
            + " | ".join(samples)
        )
    return lookup


def _build_image_lookup(images_raw_dir: Path) -> Dict[str, List[Path]]:
    lookup: Dict[str, List[Path]] = {}
    for path in images_raw_dir.rglob("*"):
        if not path.is_file():
            continue
        stem = _normalize_isic_token(path.stem)
        lookup.setdefault(stem, []).append(path)
    return lookup


def _resolve_image_path(
    isic_id: str,
    image_lookup: Dict[str, List[Path]],
    image_filename_hint: Optional[str],
    images_raw_dir: Path,
) -> Optional[Path]:
    if image_filename_hint:
        hinted = images_raw_dir / Path(image_filename_hint).name
        if hinted.exists():
            return hinted

    candidates = image_lookup.get(_normalize_isic_token(isic_id), [])
    if not candidates:
        return None

    # Prefer common image extensions deterministically.
    def _score(path: Path) -> tuple[int, str]:
        suffix = path.suffix.lower()
        order = {".jpg": 0, ".jpeg": 1, ".png": 2}
        return (order.get(suffix, 99), path.name)

    return sorted(candidates, key=_score)[0]


def _choose_consensus_entry(entries: Sequence[MaskEntry], mask_kind: str) -> Optional[MaskEntry]:
    candidates = [entry for entry in entries if entry.mask_kind == mask_kind]
    if not candidates:
        return None
    return sorted(candidates, key=lambda e: e.mask_filename)[0]


def _copy_mask_with_collision_handling(source: Path, destination_dir: Path, preferred_name: str) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    preferred = destination_dir / preferred_name
    if not preferred.exists():
        shutil.copy2(source, preferred)
        return preferred
    if filecmp.cmp(source, preferred, shallow=False):
        return preferred

    stem = Path(preferred_name).stem
    suffix = Path(preferred_name).suffix
    idx = 1
    while True:
        candidate = destination_dir / f"{stem}__dup{idx}{suffix}"
        if not candidate.exists():
            shutil.copy2(source, candidate)
            return candidate
        idx += 1


def _run_isic_download(
    *,
    isic_ids: Sequence[str],
    ids_file: Path,
    output_dir: Path,
    command_template: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if "{isic_id}" in command_template:
        for isic_id in isic_ids:
            cmd = [
                token.format(
                    output_dir=str(output_dir),
                    ids_file=str(ids_file),
                    isic_id=isic_id,
                    count=len(isic_ids),
                )
                for token in shlex.split(command_template)
            ]
            logging.info("Running ISIC download command: %s", " ".join(cmd))
            subprocess.run(cmd, check=True)
        return

    cmd = [
        token.format(
            output_dir=str(output_dir),
            ids_file=str(ids_file),
            count=len(isic_ids),
        )
        for token in shlex.split(command_template)
    ]
    logging.info("Running ISIC download command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _find_existing_isic_image(output_dir: Path, isic_id: str) -> Optional[Path]:
    direct_candidates = [
        output_dir / f"{isic_id}.jpg",
        output_dir / f"{isic_id}.jpeg",
        output_dir / f"{isic_id}.png",
    ]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate

    for candidate in output_dir.glob(f"{isic_id}.*"):
        if candidate.is_file():
            return candidate
    return None


def _request_json(url: str, *, timeout_sec: float) -> Dict[str, Any]:
    request = Request(url, headers={"User-Agent": "gemini-segmentation/ima-prep"})
    with urlopen(request, timeout=timeout_sec) as response:
        body = response.read().decode("utf-8")
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected JSON payload at {url}")
    return payload


def _download_binary(url: str, destination: Path, *, timeout_sec: float) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "gemini-segmentation/ima-prep"})
    try:
        with urlopen(request, timeout=timeout_sec) as response, tmp_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        tmp_path.replace(destination)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _pick_isic_image_destination(output_dir: Path, isic_id: str, file_url: str) -> Path:
    suffix = Path(urlparse(file_url).path).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
        suffix = ".jpg"
    return output_dir / f"{isic_id}{suffix}"


def _download_single_isic_api_image(
    *,
    isic_id: str,
    output_dir: Path,
    api_url_template: str,
    retries: int,
    timeout_sec: float,
    backoff_sec: float,
    skip_existing: bool,
) -> tuple[str, Optional[str]]:
    if skip_existing and _find_existing_isic_image(output_dir, isic_id) is not None:
        return ("skipped", None)

    metadata_url = api_url_template.format(isic_id=isic_id)
    last_error: Optional[str] = None

    for attempt in range(1, retries + 1):
        try:
            payload = _request_json(metadata_url, timeout_sec=timeout_sec)
            full_entry = payload.get("files", {}).get("full", {})
            if not isinstance(full_entry, dict):
                raise ValueError(f"Invalid files.full payload for {isic_id}")
            file_url = full_entry.get("url")
            if not file_url:
                raise ValueError(f"Missing full image URL in API response for {isic_id}")

            destination = _pick_isic_image_destination(output_dir, isic_id, str(file_url))
            if skip_existing and destination.exists():
                return ("skipped", None)

            _download_binary(str(file_url), destination, timeout_sec=timeout_sec)
            return ("downloaded", None)
        except HTTPError as exc:
            last_error = f"HTTP {exc.code}: {exc.reason}"
            # Retry transient server/rate-limit errors only.
            if exc.code not in {408, 425, 429, 500, 502, 503, 504}:
                break
        except (TimeoutError, URLError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt < retries:
            sleep_sec = max(0.0, backoff_sec) * (2 ** (attempt - 1))
            if sleep_sec:
                time.sleep(sleep_sec)

    return ("failed", last_error)


def _run_isic_api_download(
    *,
    isic_ids: Sequence[str],
    output_dir: Path,
    api_url_template: str,
    workers: int,
    retries: int,
    timeout_sec: float,
    backoff_sec: float,
    skip_existing: bool,
    fail_on_errors: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    unique_ids = sorted(set(isic_ids))

    max_workers = max(1, int(workers))
    max_retries = max(1, int(retries))

    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    failures: List[str] = []

    def _task(identifier: str) -> tuple[str, str, Optional[str]]:
        status, error_message = _download_single_isic_api_image(
            isic_id=identifier,
            output_dir=output_dir,
            api_url_template=api_url_template,
            retries=max_retries,
            timeout_sec=timeout_sec,
            backoff_sec=backoff_sec,
            skip_existing=skip_existing,
        )
        return (identifier, status, error_message)

    logging.info(
        "Starting ISIC API downloads for %d IDs (workers=%d, retries=%d, skip_existing=%s)",
        len(unique_ids),
        max_workers,
        max_retries,
        skip_existing,
    )

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_task, isic_id): isic_id for isic_id in unique_ids}
        for future in as_completed(future_map):
            isic_id, status, error_message = future.result()
            counts[status] = counts.get(status, 0) + 1
            completed += 1

            if status == "failed":
                failures.append(f"{isic_id}: {error_message or 'unknown error'}")

            if completed % 100 == 0 or completed == len(unique_ids):
                logging.info(
                    "ISIC API progress %d/%d (downloaded=%d skipped=%d failed=%d)",
                    completed,
                    len(unique_ids),
                    counts.get("downloaded", 0),
                    counts.get("skipped", 0),
                    counts.get("failed", 0),
                )

    if failures and fail_on_errors:
        sample = " | ".join(failures[:10])
        raise RuntimeError(f"Failed ISIC image downloads ({len(failures)} IDs). Sample: {sample}")

def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def _write_manifest(path: Path, image_names: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(image_names) + ("\n" if image_names else ""), encoding="utf-8")


def prepare_ima_plusplus(args: argparse.Namespace) -> None:
    raw_root = Path(args.raw_root).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()

    metadata_raw_dir = raw_root / "metadata"
    masks_raw_dir = raw_root / "masks_raw"
    images_raw_dir = raw_root / "images_raw"
    tmp_dir = raw_root / "tmp"

    segs_zip_path = Path(args.segs_zip).expanduser().resolve() if args.segs_zip else tmp_dir / "segs.zip"
    seg_metadata_path = (
        Path(args.seg_metadata).expanduser().resolve()
        if args.seg_metadata
        else metadata_raw_dir / "seg_metadata.csv"
    )
    seg_metadata_multi_path = (
        Path(args.seg_metadata_multiannotator).expanduser().resolve()
        if args.seg_metadata_multiannotator
        else metadata_raw_dir / "seg_metadata_multiannotator_subset.csv"
    )
    img_metadata_path = (
        Path(args.img_metadata).expanduser().resolve()
        if args.img_metadata
        else metadata_raw_dir / "img_metadata.csv"
    )

    train_csv_path = Path(args.train_csv).expanduser().resolve() if args.train_csv else metadata_raw_dir / "train.csv"
    val_csv_path = Path(args.val_csv).expanduser().resolve() if args.val_csv else metadata_raw_dir / "val.csv"
    test_csv_path = Path(args.test_csv).expanduser().resolve() if args.test_csv else metadata_raw_dir / "test.csv"

    if args.download_zenodo:
        _download_file(args.segs_zip_url, segs_zip_path, overwrite=args.overwrite_downloads)
        _download_file(args.seg_metadata_url, seg_metadata_path, overwrite=args.overwrite_downloads)
        _download_file(args.img_metadata_url, img_metadata_path, overwrite=args.overwrite_downloads)
        if args.download_multiannotator_metadata:
            _download_file(
                args.seg_metadata_multiannotator_url,
                seg_metadata_multi_path,
                overwrite=args.overwrite_downloads,
            )
        if args.download_split_csvs:
            _download_file(args.train_csv_url, train_csv_path, overwrite=args.overwrite_downloads)
            _download_file(args.val_csv_url, val_csv_path, overwrite=args.overwrite_downloads)
            _download_file(args.test_csv_url, test_csv_path, overwrite=args.overwrite_downloads)

    if not seg_metadata_path.exists():
        raise FileNotFoundError(f"seg_metadata.csv not found at {seg_metadata_path}")
    if not img_metadata_path.exists():
        raise FileNotFoundError(f"img_metadata.csv not found at {img_metadata_path}")

    if args.extract_masks:
        _extract_zip(segs_zip_path, masks_raw_dir, force=args.force_extract)

    if not masks_raw_dir.exists():
        raise FileNotFoundError(
            f"masks_raw directory missing: {masks_raw_dir}. Provide extracted segs or enable --extract-masks."
        )

    seg_df = pd.read_csv(seg_metadata_path)
    img_df = pd.read_csv(img_metadata_path)

    isic_col = _find_column(seg_df, ("ISIC_id",), required=True)
    mask_filename_col = _find_column(seg_df, MASK_FILENAME_COLUMNS, required=True)

    img_isic_col = _find_column(img_df, ("ISIC_id",), required=True)
    img_filename_col = _find_column(img_df, IMAGE_FILENAME_COLUMNS, required=False)

    mask_lookup = _build_mask_source_lookup(masks_raw_dir)

    image_filename_map: Dict[str, str] = {}
    if img_filename_col:
        for _, row in img_df.iterrows():
            isic_id = _normalize_isic_id(row[img_isic_col])
            if not isic_id:
                continue
            filename = row[img_filename_col]
            if pd.isna(filename):
                continue
            image_filename_map.setdefault(isic_id, Path(str(filename)).name)

    grouped_entries: Dict[str, List[MaskEntry]] = {}
    for _, row in seg_df.iterrows():
        isic_id = _normalize_isic_id(row[isic_col])
        if not isic_id:
            continue

        raw_mask_value = row[mask_filename_col]
        if pd.isna(raw_mask_value):
            raise ValueError(f"Row for ISIC_id={isic_id} has empty mask filename in column {mask_filename_col}")
        mask_filename = Path(str(raw_mask_value)).name

        source_path = mask_lookup.get(mask_filename)
        if source_path is None:
            raise FileNotFoundError(
                f"Mask file '{mask_filename}' referenced by ISIC_id={isic_id} not found under {masks_raw_dir}"
            )

        metadata = _sanitize_for_json({str(col): row[col] for col in seg_df.columns})
        metadata["ISIC_id"] = isic_id

        entry = MaskEntry(
            isic_id=isic_id,
            mask_filename=mask_filename,
            source_path=source_path,
            mask_kind=_classify_mask(mask_filename),
            metadata=metadata,
        )
        grouped_entries.setdefault(isic_id, []).append(entry)

    isic_ids = sorted(grouped_entries)

    metadata_out_dir = dataset_root / "metadata"
    ids_file = metadata_out_dir / "isic_ids.txt"
    metadata_out_dir.mkdir(parents=True, exist_ok=True)
    ids_file.write_text("\n".join(isic_ids) + ("\n" if isic_ids else ""), encoding="utf-8")

    if args.download_images:
        if args.download_images_mode == "api":
            _run_isic_api_download(
                isic_ids=isic_ids,
                output_dir=images_raw_dir,
                api_url_template=args.isic_api_url_template,
                workers=args.isic_api_workers,
                retries=args.isic_api_retries,
                timeout_sec=args.isic_api_timeout_sec,
                backoff_sec=args.isic_api_backoff_sec,
                skip_existing=args.isic_api_skip_existing,
                fail_on_errors=args.isic_api_fail_on_errors,
            )
        else:
            _run_isic_download(
                isic_ids=isic_ids,
                ids_file=ids_file,
                output_dir=images_raw_dir,
                command_template=args.isic_download_template,
            )

    if not images_raw_dir.exists():
        raise FileNotFoundError(
            f"images_raw directory missing: {images_raw_dir}. Provide images or enable --download-images."
        )

    image_lookup = _build_image_lookup(images_raw_dir)

    images_dir = dataset_root / "images"
    masks_dir = dataset_root / "masks"
    masks_all_dir = dataset_root / "masks_all"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    masks_all_dir.mkdir(parents=True, exist_ok=True)

    index_rows: List[Dict[str, Any]] = []
    missing_images: List[str] = []
    policy_counts: Dict[str, int] = {
        "consensus_staple": 0,
        "consensus_mv": 0,
        "single_annotator_only": 0,
    }

    for isic_id in isic_ids:
        entries = grouped_entries[isic_id]
        entries_sorted = sorted(entries, key=lambda item: item.mask_filename)

        staple_entry = _choose_consensus_entry(entries_sorted, "consensus_staple")
        mv_entry = _choose_consensus_entry(entries_sorted, "consensus_mv")
        annotator_entries = [entry for entry in entries_sorted if entry.mask_kind == "annotator"]

        if staple_entry is not None:
            canonical_entry = staple_entry
            gt_policy = "consensus_staple"
        elif mv_entry is not None:
            canonical_entry = mv_entry
            gt_policy = "consensus_mv"
        elif len(annotator_entries) == 1:
            canonical_entry = annotator_entries[0]
            gt_policy = "single_annotator_only"
        elif len(annotator_entries) > 1:
            raise ValueError(
                "Integrity error: multi-annotator image missing consensus masks "
                f"for ISIC_id={isic_id} (annotator masks={len(annotator_entries)})."
            )
        else:
            raise ValueError(f"No usable masks found for ISIC_id={isic_id}")

        image_filename_hint = image_filename_map.get(isic_id)
        image_source = _resolve_image_path(isic_id, image_lookup, image_filename_hint, images_raw_dir)
        if image_source is None:
            missing_images.append(isic_id)
            logging.warning("Skipping %s due to missing source image", isic_id)
            continue

        image_dest = images_dir / image_source.name
        shutil.copy2(image_source, image_dest)

        gt_mask_dest = masks_dir / image_dest.name
        shutil.copy2(canonical_entry.source_path, gt_mask_dest)

        copied_mask_paths: List[str] = []
        all_mask_metadata: List[Dict[str, Any]] = []
        staple_mask_rel: Optional[str] = None
        mv_mask_rel: Optional[str] = None

        for entry in entries_sorted:
            copied_path = _copy_mask_with_collision_handling(
                entry.source_path,
                masks_all_dir,
                entry.mask_filename,
            )
            rel_mask_path = str(copied_path.relative_to(dataset_root))
            copied_mask_paths.append(rel_mask_path)

            mask_meta = dict(entry.metadata)
            mask_meta["mask_kind"] = entry.mask_kind
            mask_meta["mask_path"] = rel_mask_path
            all_mask_metadata.append(mask_meta)

            if entry.mask_kind == "consensus_staple" and staple_mask_rel is None:
                staple_mask_rel = rel_mask_path
            if entry.mask_kind == "consensus_mv" and mv_mask_rel is None:
                mv_mask_rel = rel_mask_path

        policy_counts[gt_policy] += 1

        index_rows.append(
            {
                "ISIC_id": isic_id,
                "image_path": str(image_dest.relative_to(dataset_root)),
                "gt_mask_path": str(gt_mask_dest.relative_to(dataset_root)),
                "gt_policy": gt_policy,
                "all_mask_paths": copied_mask_paths,
                "all_mask_metadata": all_mask_metadata,
                "staple_mask_path": staple_mask_rel,
                "mv_mask_path": mv_mask_rel,
                "n_masks": len(entries_sorted),
                "n_annotator_masks": len(annotator_entries),
            }
        )

    if not index_rows:
        raise RuntimeError("No images were prepared. Check metadata/mask/image inputs.")

    index_rows = sorted(index_rows, key=lambda row: str(row["image_path"]))
    image_names = [Path(str(row["image_path"])).name for row in index_rows]

    manifest_path = dataset_root / "master_imagelist_ima_plusplus.txt"
    _write_manifest(manifest_path, image_names)

    index_jsonl = metadata_out_dir / "ima_plusplus_index.jsonl"
    _write_jsonl(index_jsonl, index_rows)

    index_csv = metadata_out_dir / "ima_plusplus_index.csv"
    csv_rows = []
    for row in index_rows:
        csv_row = dict(row)
        csv_row["all_mask_paths"] = json.dumps(row["all_mask_paths"], sort_keys=True)
        csv_row["all_mask_metadata"] = json.dumps(row["all_mask_metadata"], sort_keys=True)
        csv_rows.append(csv_row)
    pd.DataFrame(csv_rows).to_csv(index_csv, index=False)

    # Copy metadata files into dataset-root metadata for downstream reproducibility.
    shutil.copy2(seg_metadata_path, metadata_out_dir / "seg_metadata.csv")
    shutil.copy2(img_metadata_path, metadata_out_dir / "img_metadata.csv")
    if seg_metadata_multi_path.exists():
        shutil.copy2(seg_metadata_multi_path, metadata_out_dir / "seg_metadata_multiannotator_subset.csv")
    else:
        logging.warning("Optional metadata not found (multi-annotator subset): %s", seg_metadata_multi_path)

    if args.write_split_manifests:
        id_to_image = {row["ISIC_id"]: Path(str(row["image_path"])).name for row in index_rows}
        token_to_image = {_normalize_isic_token(isic_id): image for isic_id, image in id_to_image.items()}
        lowercase_name_to_image = {name.lower(): name for name in image_names}
        for split_name, split_path in (
            ("train", train_csv_path),
            ("val", val_csv_path),
            ("test", test_csv_path),
        ):
            if not split_path.exists():
                continue
            split_df = pd.read_csv(split_path)
            split_isic_col = _find_column(split_df, ("ISIC_id",), required=False)
            split_img_col = _find_column(split_df, IMAGE_FILENAME_COLUMNS, required=False)

            names: List[str] = []
            if split_isic_col:
                for value in split_df[split_isic_col]:
                    isic_id = _normalize_isic_id(value)
                    if isic_id in id_to_image:
                        names.append(id_to_image[isic_id])
            elif split_img_col:
                for value in split_df[split_img_col]:
                    if pd.isna(value):
                        continue
                    name = Path(str(value)).name
                    normalized_token = _normalize_isic_token(Path(name).stem)
                    resolved = token_to_image.get(normalized_token)
                    if resolved:
                        names.append(resolved)
                        continue
                    lowered = lowercase_name_to_image.get(name.lower())
                    if lowered:
                        names.append(lowered)
            else:
                logging.warning(
                    "Skipping %s split manifest; no ISIC/image filename columns found in %s",
                    split_name,
                    split_path,
                )
                continue

            names = sorted(set(names))
            split_manifest = dataset_root / f"{split_name}_ima_plusplus.txt"
            _write_manifest(split_manifest, names)

    prep_report = {
        "dataset_root": str(dataset_root),
        "raw_root": str(raw_root),
        "manifest_path": str(manifest_path),
        "index_jsonl": str(index_jsonl),
        "index_csv": str(index_csv),
        "total_seg_rows": int(len(seg_df)),
        "total_isic_ids": int(len(isic_ids)),
        "prepared_images": int(len(index_rows)),
        "missing_images": missing_images,
        "policy_counts": policy_counts,
        "notes": [
            "Canonical GT policy: STAPLE -> majority-vote -> single annotator only.",
            "All masks and per-mask metadata retained under masks_all and index files.",
        ],
    }

    report_path = metadata_out_dir / "prep_report.json"
    report_path.write_text(json.dumps(prep_report, indent=2, sort_keys=True), encoding="utf-8")

    logging.info("Prepared %s images into %s", len(index_rows), dataset_root)
    logging.info("Manifest: %s", manifest_path)
    logging.info("Index JSONL: %s", index_jsonl)
    logging.info("Prep report: %s", report_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare IMA++ into CLI-compatible dataset layout.")

    parser.add_argument("--raw-root", default="data/IMAplusplus_raw", help="Raw staging root")
    parser.add_argument("--dataset-root", default="data/IMAplusplus_cli", help="Output dataset root")

    parser.add_argument("--segs-zip", help="Path to segs.zip")
    parser.add_argument("--seg-metadata", help="Path to seg_metadata.csv")
    parser.add_argument(
        "--seg-metadata-multiannotator",
        help="Optional path to seg_metadata_multiannotator_subset.csv",
    )
    parser.add_argument("--img-metadata", help="Path to img_metadata.csv")
    parser.add_argument("--train-csv", help="Optional path to train.csv")
    parser.add_argument("--val-csv", help="Optional path to val.csv")
    parser.add_argument("--test-csv", help="Optional path to test.csv")

    parser.add_argument("--download-zenodo", action="store_true", help="Download Zenodo files before preparing")
    parser.add_argument("--download-split-csvs", action="store_true", help="Download train/val/test split CSVs")
    parser.add_argument(
        "--download-multiannotator-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download seg_metadata_multiannotator_subset.csv when --download-zenodo is enabled",
    )
    parser.add_argument("--overwrite-downloads", action="store_true", help="Overwrite existing downloaded files")

    parser.add_argument("--segs-zip-url", default=DEFAULT_SEGS_URL, help="Zenodo URL for segs.zip")
    parser.add_argument("--seg-metadata-url", default=DEFAULT_SEG_METADATA_URL, help="Zenodo URL for seg_metadata.csv")
    parser.add_argument("--img-metadata-url", default=DEFAULT_IMG_METADATA_URL, help="Zenodo URL for img_metadata.csv")
    parser.add_argument(
        "--seg-metadata-multiannotator-url",
        default=DEFAULT_SEG_METADATA_MULTI_URL,
        help="Zenodo URL for seg_metadata_multiannotator_subset.csv",
    )
    parser.add_argument("--train-csv-url", default=DEFAULT_TRAIN_URL, help="Zenodo URL for train.csv")
    parser.add_argument("--val-csv-url", default=DEFAULT_VAL_URL, help="Zenodo URL for val.csv")
    parser.add_argument("--test-csv-url", default=DEFAULT_TEST_URL, help="Zenodo URL for test.csv")

    parser.add_argument(
        "--extract-masks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Extract segs.zip into masks_raw",
    )
    parser.add_argument("--force-extract", action="store_true", help="Force re-extraction of segs.zip")

    parser.add_argument("--download-images", action="store_true", help="Download ISIC images")
    parser.add_argument(
        "--download-images-mode",
        choices=("api", "template"),
        default="api",
        help=(
            "Image download backend. 'api' uses threaded ISIC API v2 downloads with retries; "
            "'template' runs --isic-download-template commands."
        ),
    )
    parser.add_argument(
        "--isic-api-url-template",
        default=DEFAULT_ISIC_API_URL_TEMPLATE,
        help="ISIC image metadata API URL template (must include {isic_id})",
    )
    parser.add_argument(
        "--isic-api-workers",
        type=int,
        default=12,
        help="Worker count for --download-images-mode api",
    )
    parser.add_argument(
        "--isic-api-retries",
        type=int,
        default=5,
        help="Retries per image for --download-images-mode api",
    )
    parser.add_argument(
        "--isic-api-timeout-sec",
        type=float,
        default=45.0,
        help="HTTP timeout in seconds for --download-images-mode api",
    )
    parser.add_argument(
        "--isic-api-backoff-sec",
        type=float,
        default=1.5,
        help="Exponential backoff base in seconds for API download retries",
    )
    parser.add_argument(
        "--isic-api-skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip download when an image already exists for the ISIC ID",
    )
    parser.add_argument(
        "--isic-api-fail-on-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail preparation when one or more API image downloads still fail after retries",
    )
    parser.add_argument(
        "--isic-download-template",
        default="isic image download --search isic_id:{isic_id} {output_dir}",
        help=(
            "Command template for isic-cli downloads. Available placeholders: "
            "{output_dir}, {ids_file}, {isic_id}, {count}"
        ),
    )

    parser.add_argument(
        "--write-split-manifests",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write optional split manifests when split CSVs are available",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.log_level)

    try:
        prepare_ima_plusplus(args)
    except Exception as exc:  # pragma: no cover - entrypoint safety
        logging.exception("IMA++ preparation failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
