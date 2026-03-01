from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


MASK_SOURCE_RANK = {
    "consensus_staple": 0,
    "consensus_mv": 1,
    "challenge_gt": 2,
    "single_annotator_only": 3,
    "assisted": 4,
    "unknown": 5,
}

SPLIT_RANK = {
    "test": 0,
    "val": 1,
    "train": 2,
    "unknown": 3,
}

_DCT_MATRIX_CACHE: Dict[int, np.ndarray] = {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _dct_matrix_ortho(n: int) -> np.ndarray:
    cached = _DCT_MATRIX_CACHE.get(int(n))
    if cached is not None:
        return cached
    n = int(n)
    if n <= 0:
        raise ValueError("DCT size must be positive")
    x = np.arange(n, dtype=np.float64)
    k = np.arange(n, dtype=np.float64).reshape(-1, 1)
    mat = np.cos((np.pi / (2.0 * n)) * (2.0 * x + 1.0) * k)
    mat[0, :] *= np.sqrt(1.0 / n)
    if n > 1:
        mat[1:, :] *= np.sqrt(2.0 / n)
    _DCT_MATRIX_CACHE[n] = mat
    return mat


def _dct2_ortho(arr: np.ndarray) -> np.ndarray:
    # Orthonormal DCT-II computed with matrix multiplication.
    # Using NumPy here avoids platform-specific SciPy FFT backend crashes.
    src = np.asarray(arr, dtype=np.float64)
    if src.ndim != 2:
        raise ValueError("DCT input must be a 2D array")
    n_rows, n_cols = src.shape
    c_rows = _dct_matrix_ortho(n_rows)
    c_cols = _dct_matrix_ortho(n_cols)
    return c_rows @ src @ c_cols.T


def phash64_hex(image_gray: np.ndarray) -> str:
    # Resize to 32x32 with nearest-neighbor style sampling for dependency-free hashing.
    arr = np.asarray(image_gray, dtype=np.float32)
    if arr.ndim == 3:
        arr = np.mean(arr, axis=-1)
    h, w = arr.shape[:2]
    ys = np.linspace(0, h - 1, 32).astype(np.int32)
    xs = np.linspace(0, w - 1, 32).astype(np.int32)
    small = arr[np.ix_(ys, xs)]

    dct_2d = _dct2_ortho(small)
    patch = dct_2d[:8, :8].copy()
    flat = patch.flatten()
    median = np.median(flat[1:])
    bits = flat > median
    bits[0] = 0
    packed = 0
    for idx, bit in enumerate(bits.tolist()):
        if bit:
            packed |= 1 << (63 - idx)
    return f"{packed:016x}"


def hamming_hex64(left: str, right: str) -> int:
    lv = int(left, 16)
    rv = int(right, 16)
    return int((lv ^ rv).bit_count())


def _hamming_to_many(base: np.uint64, others: np.ndarray) -> np.ndarray:
    if others.size == 0:
        return np.empty((0,), dtype=np.int32)
    xor_vals = np.bitwise_xor(np.asarray(others, dtype=np.uint64), np.uint64(base))
    byte_view = xor_vals.view(np.uint8).reshape(-1, 8)
    bit_counts = np.unpackbits(byte_view, axis=1).sum(axis=1)
    return bit_counts.astype(np.int32, copy=False)


def _canonical_sort_key(row: pd.Series) -> Tuple[object, ...]:
    mask_rank = MASK_SOURCE_RANK.get(str(row.get("mask_source", "unknown")), MASK_SOURCE_RANK["unknown"])
    split_rank = SPLIT_RANK.get(str(row.get("split", "unknown")), SPLIT_RANK["unknown"])
    height = int(row.get("image_height", 0) or 0)
    width = int(row.get("image_width", 0) or 0)
    area_rank = -1 * (height * width)
    image_id = str(row.get("image_id", ""))
    return (mask_rank, split_rank, area_rank, image_id)


def _canonical_member(members: pd.DataFrame) -> pd.Series:
    ordered = sorted(
        (row for _, row in members.iterrows()),
        key=_canonical_sort_key,
    )
    return ordered[0]


def apply_exact_dedup(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), pd.DataFrame(), pd.DataFrame()

    working = df.copy()
    group_ids = []
    canonical_lookup: Dict[str, str] = {}
    dedup_rows: List[Dict[str, object]] = []

    grouped = list(working.groupby("sha256", sort=True))
    for idx, (sha, group) in enumerate(grouped):
        dedup_group = f"exact_{idx:06d}"
        canonical = _canonical_member(group)
        canonical_image = str(canonical["image_name"])
        canonical_lookup[str(sha)] = canonical_image
        members = sorted(group["image_name"].astype(str).tolist())
        dedup_rows.append(
            {
                "sha256": str(sha),
                "dedup_group_id": dedup_group,
                "canonical_image_id": str(canonical.get("image_id", canonical_image)),
                "canonical_image_name": canonical_image,
                "all_members": "|".join(members),
                "n_members": len(members),
            }
        )
        group_ids.extend([dedup_group] * len(group))

    dedup_map = pd.DataFrame(dedup_rows)
    working = working.merge(
        dedup_map[["sha256", "dedup_group_id", "canonical_image_name"]],
        on="sha256",
        how="left",
    )
    working["is_canonical"] = working["image_name"] == working["canonical_image_name"]

    report = (
        working.groupby("dataset_source_primary", dropna=False)
        .agg(
            total_images=("image_name", "count"),
            canonical_images=("is_canonical", "sum"),
        )
        .reset_index()
    )
    report["collapsed_duplicates"] = report["total_images"] - report["canonical_images"]

    return working, dedup_map, report


def _union_find(items: Iterable[int]) -> Tuple[Dict[int, int], Callable[[int], int], Callable[[int, int], None]]:
    parent = {i: i for i in items}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    return parent, find, union


def apply_near_dedup(df: pd.DataFrame, *, threshold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), pd.DataFrame()

    working = df.copy().reset_index(drop=True)
    parent, find, union = _union_find(range(len(working)))

    hashes = working["phash64_hex"].fillna("0" * 16).astype(str).tolist()
    hash_values = np.asarray([int(h, 16) for h in hashes], dtype=np.uint64)
    distance_limit = int(threshold)
    for i in range(len(hash_values)):
        tail = hash_values[i + 1 :]
        if tail.size == 0:
            continue
        distances = _hamming_to_many(hash_values[i], tail)
        neighbors = np.where(distances <= distance_limit)[0]
        for offset in neighbors.tolist():
            union(i, i + 1 + int(offset))

    cluster_map: Dict[int, List[int]] = defaultdict(list)
    for idx in range(len(working)):
        cluster_map[find(idx)].append(idx)

    near_rows: List[Dict[str, object]] = []
    cluster_id_for_index: Dict[int, str] = {}
    for order, (_, members) in enumerate(sorted(cluster_map.items(), key=lambda x: min(x[1]))):
        cluster_id = f"near_{order:06d}"
        member_df = working.iloc[members]
        canonical = _canonical_member(member_df)
        canonical_name = str(canonical["image_name"])
        member_names = sorted(member_df["image_name"].astype(str).tolist())
        near_rows.append(
            {
                "cluster_id": cluster_id,
                "canonical_image_name": canonical_name,
                "canonical_image_id": str(canonical.get("image_id", canonical_name)),
                "members": "|".join(member_names),
                "n_members": len(member_names),
            }
        )
        for idx in members:
            cluster_id_for_index[idx] = cluster_id

    working["near_dedup_group_id"] = [cluster_id_for_index[i] for i in range(len(working))]
    near_map = pd.DataFrame(near_rows)
    return working, near_map
