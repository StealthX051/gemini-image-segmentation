from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from gemini_segmentation.fairness_enhanced.dedup import hamming_hex64, phash64_hex, sha256_file

MASK_SOURCE_PATTERN = re.compile(r"(mask|seg|label|ground\s*truth|groundtruth|annotation)", re.IGNORECASE)


@dataclass(frozen=True)
class LeakageAuditResult:
    retrieval_duplicate: bool
    retrieval_mask_source: bool
    audit_unavailable: bool
    duplicate_reasons: List[str]
    mask_source_reasons: List[str]


def _is_binary_like(image_bgr: np.ndarray) -> bool:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    unique_vals = np.unique(gray)
    if unique_vals.size <= 4:
        return True
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    peaks = np.sort(hist)[-2:]
    return bool(peaks.sum() / max(1.0, hist.sum()) > 0.85)


def _hash_image_file(path: Path) -> Optional[Dict[str, str]]:
    if not path.exists() or not path.is_file():
        return None
    try:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        return {
            "sha256": sha256_file(path),
            "phash64": phash64_hex(image),
        }
    except Exception:
        return None


def audit_retrieval(
    *,
    input_image_path: Path,
    grounding: Dict[str, Any],
    near_hamming_threshold: int = 8,
) -> LeakageAuditResult:
    chunks = grounding.get("grounding_chunks") or []
    if not isinstance(chunks, list):
        chunks = []

    if not chunks:
        return LeakageAuditResult(
            retrieval_duplicate=False,
            retrieval_mask_source=False,
            audit_unavailable=True,
            duplicate_reasons=[],
            mask_source_reasons=[],
        )

    input_hash = _hash_image_file(input_image_path)
    duplicate_reasons: List[str] = []
    mask_source_reasons: List[str] = []
    duplicate_audit_possible = False

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        web_meta = chunk.get("web") if isinstance(chunk.get("web"), dict) else {}
        retrieved_image_meta = (
            chunk.get("retrieved_image") if isinstance(chunk.get("retrieved_image"), dict) else {}
        )
        url = str(chunk.get("uri") or chunk.get("url") or web_meta.get("uri") or web_meta.get("url") or "")
        title = str(chunk.get("title") or web_meta.get("title") or "")
        snippet = str(
            chunk.get("snippet")
            or chunk.get("text")
            or web_meta.get("snippet")
            or web_meta.get("text")
            or ""
        )

        text_blob = " ".join([url, title, snippet]).strip()
        if text_blob and MASK_SOURCE_PATTERN.search(text_blob):
            mask_source_reasons.append(f"text_indicator:{url or title}")

        local_path = (
            chunk.get("local_path")
            or chunk.get("localPath")
            or retrieved_image_meta.get("local_path")
            or retrieved_image_meta.get("localPath")
        )
        if isinstance(local_path, str) and local_path:
            retrieved = _hash_image_file(Path(local_path))
            if input_hash and retrieved:
                duplicate_audit_possible = True
                if retrieved["sha256"] == input_hash["sha256"]:
                    duplicate_reasons.append(f"sha256_match:{local_path}")
                else:
                    distance = hamming_hex64(input_hash["phash64"], retrieved["phash64"])
                    if distance <= int(near_hamming_threshold):
                        duplicate_reasons.append(f"phash_hamming_{distance}:{local_path}")
            image = cv2.imread(local_path, cv2.IMREAD_COLOR)
            if image is not None and _is_binary_like(image):
                mask_source_reasons.append(f"binary_like_image:{local_path}")

    return LeakageAuditResult(
        retrieval_duplicate=bool(duplicate_reasons),
        retrieval_mask_source=bool(mask_source_reasons),
        audit_unavailable=not duplicate_audit_possible,
        duplicate_reasons=duplicate_reasons,
        mask_source_reasons=mask_source_reasons,
    )
