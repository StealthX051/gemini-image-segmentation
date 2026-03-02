from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from .artifact_store import ArtifactStore


@dataclass
class RunRecord:
    run_id: str
    image_id: str
    image_name: str
    split: str
    mode: str
    replicate_idx: int
    model_config: Dict[str, Any]
    tool_config: Dict[str, Any]
    prompt: Dict[str, Any]
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    outputs: Dict[str, Any] = field(default_factory=dict)
    evaluation: Dict[str, Any] = field(default_factory=dict)
    leakage: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def persist_run_record(store: ArtifactStore, *, run_dir, record: RunRecord):
    return store.write_json(run_dir=run_dir, name="run_record.json", payload=record.to_dict())
