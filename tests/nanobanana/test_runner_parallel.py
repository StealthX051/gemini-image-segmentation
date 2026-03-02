from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from nanobanana_segmentation.core.types import AttemptResult, DatasetItem, EngineResult, QCMetrics
from nanobanana_segmentation.study import runner
from nanobanana_segmentation.study.config import load_study_config


class _FakeEngine:
    def segment_once(self, request):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:6, 2:6] = 255
        attempt = AttemptResult(
            attempt_index=1,
            prompt_id="chromakey_v1",
            prompt_text="prompt",
            attempt_mode="chromakey",
            extraction_method="chromakey_hsv",
            qc_metrics=QCMetrics(
                resolution_match=True,
                mask_nonempty=True,
                mask_area_frac=0.25,
                component_count=1,
                largest_component_frac=1.0,
                speckle_score=0.0,
                border_touch=False,
                green_coverage=0.75,
                green_uniformity_proxy=1.0,
            ),
            qc_pass=True,
            failure_reasons=[],
            score=1000.0,
            transport_retries=0,
            transport_retry_events=[],
            thought_signature_present=False,
            thought_signatures=[],
            thought_summaries=[],
            grounding={"grounding_chunks": []},
            raw_request_path=None,
            raw_response_path=None,
            surrogate_image_path=None,
            intermediate_mask_paths=[],
            overlay_path=None,
        )
        return EngineResult(
            run_id=request.run_id,
            image_id=request.image_id,
            image_name=request.image_name,
            selected_attempt_index=1,
            qc_pass=True,
            warnings=[],
            attempts=[attempt],
            mask=mask,
            surrogate=None,
            mask_hash="fake",
            surrogate_hash=None,
            run_record_path=None,
        )


class _Leakage:
    retrieval_duplicate = False
    retrieval_mask_source = False
    audit_unavailable = False
    duplicate_reasons = []
    mask_source_reasons = []


def test_runner_parallel_stage0(tmp_path: Path, monkeypatch) -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    ok_img, enc_img = cv2.imencode(".png", image)
    ok_mask, enc_mask = cv2.imencode(".png", mask)
    assert ok_img and ok_mask

    image_path = tmp_path / "images" / "sample.png"
    mask_path = tmp_path / "masks" / "sample.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(bytes(enc_img))
    mask_path.write_bytes(bytes(enc_mask))

    cfg_path = tmp_path / "study.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "study_id: test_parallel",
                "dataset_name: pneumothorax_cxr",
                f"dataset_root: {tmp_path.as_posix()}",
                "target: pneumothorax",
                "matrix:",
                "  tool_modes: [closed, text]",
                "  thinking_levels: [minimal]",
                "  replicates: 1",
                "stages:",
                "  stage0_sample_size: 1",
                "execution:",
                "  workers: 2",
                "paths:",
                f"  results_root: {(tmp_path / 'results_nanobanana').as_posix()}",
                f"  artifacts_root: {(tmp_path / 'artifacts_nanobanana').as_posix()}",
            ]
        ),
        encoding="utf-8",
    )
    cfg = load_study_config(cfg_path)

    monkeypatch.setattr(runner, "_build_engine", lambda _: _FakeEngine())
    monkeypatch.setattr(
        runner,
        "load_dataset_items",
        lambda **_: [DatasetItem(image_path=image_path, mask_path=mask_path, split="test")],
    )
    monkeypatch.setattr(runner, "audit_retrieval", lambda **_: _Leakage())

    run_dir = runner._run_stage(cfg, stage="stage0")
    results_df = pd.read_csv(run_dir / "results.csv")
    assert len(results_df) == 2
    assert sorted(results_df["tool_mode"].tolist()) == ["closed", "text"]

    run_summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert run_summary["n_tasks"] == 2
    assert run_summary["n_failures"] == 0
    assert run_summary["execution"]["workers"] == 2

