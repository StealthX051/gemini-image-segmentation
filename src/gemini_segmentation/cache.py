from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_request_cache_key(
    *,
    image_path: Path,
    provider: str,
    model_name: str,
    prompt_hash: str,
    prompt_family: str | None,
    temperature: float | None,
    thinking_budget: int | None,
    targets: tuple[str, ...] | None = None,
) -> str:
    payload = {
        "provider": provider,
        "model_name": model_name,
        "prompt_hash": prompt_hash,
        "prompt_family": prompt_family,
        "temperature": temperature,
        "thinking_budget": thinking_budget,
        "targets": list(targets) if targets else None,
        "image_sha256": hash_file(image_path),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(serialized)


class DiskRequestCache:
    """Simple JSON blob cache keyed by deterministic request hashes."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path_for(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def load(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._path_for(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return None

    def save(self, key: str, payload: Dict[str, Any]) -> None:
        path = self._path_for(key)
        tmp_path = path.with_suffix(".tmp")
        blob = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        with self._lock:
            tmp_path.write_text(blob)
            tmp_path.replace(path)
