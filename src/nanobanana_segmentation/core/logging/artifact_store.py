from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Tuple


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(path)


def _atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def run_dir(self, *, image_id: str, mode: str, replicate_idx: int, run_id: str) -> Path:
        path = self.root / image_id / mode / f"replicate_{replicate_idx}" / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_bytes(
        self,
        *,
        run_dir: Path,
        stem: str,
        payload: bytes,
        suffix: str,
        subdir: str = "",
    ) -> Tuple[Path, str]:
        digest = _sha256_bytes(payload)
        target_dir = run_dir / subdir if subdir else run_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        safe_suffix = suffix if suffix.startswith(".") else f".{suffix}"
        out_path = target_dir / f"{stem}_{digest[:16]}{safe_suffix}"
        _atomic_write_bytes(out_path, payload)
        return out_path, digest

    def write_json(self, *, run_dir: Path, name: str, payload: Dict[str, Any], subdir: str = "") -> Path:
        target_dir = run_dir / subdir if subdir else run_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        out = target_dir / name
        _atomic_write_text(out, json.dumps(payload, indent=2, sort_keys=True, default=str))
        return out
