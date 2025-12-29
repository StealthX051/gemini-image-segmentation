import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from gemini_segmentation.io import overlay_mask_on_img
from gemini_segmentation.paper.best_cases import generate_best_case_montage


def _write_image(path: Path, color: str = "white") -> None:
    img = Image.new("RGB", (32, 32), color=color)
    img.save(path)


def _write_mask(path: Path, *, box: tuple[int, int, int, int]) -> None:
    mask = np.zeros((32, 32), dtype=np.uint8)
    y0, x0, y1, x1 = box
    mask[y0:y1, x0:x1] = 255
    Image.fromarray(mask).save(path)


def test_generate_best_case_montage_respects_persisted_selection(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    images_dir = dataset_root / "images"
    masks_dir = dataset_root / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir()

    img_a = images_dir / "img_a.png"
    img_b = images_dir / "img_b.png"
    _write_image(img_a, "white")
    _write_image(img_b, "white")

    gt_a = masks_dir / "img_a.png"
    gt_b = masks_dir / "img_b.png"
    _write_mask(gt_a, box=(8, 8, 24, 24))
    _write_mask(gt_b, box=(10, 10, 20, 20))

    results_root = tmp_path / "results"
    run_dir = results_root / "polyp" / "gemini-2.5-flash" / "desc_v1-abcd" / "20240101-000000"
    (run_dir / "masks").mkdir(parents=True)
    (run_dir / "overlays").mkdir()

    pred_a = run_dir / "masks" / "img_a.png"
    pred_b = run_dir / "masks" / "img_b.png"
    _write_mask(pred_a, box=(8, 8, 24, 24))
    _write_mask(pred_b, box=(10, 10, 18, 18))

    overlay_a = overlay_mask_on_img(Image.open(img_a), np.array(Image.open(pred_a)), "red")
    overlay_b = overlay_mask_on_img(Image.open(img_b), np.array(Image.open(pred_b)), "red")
    overlay_a.save(run_dir / "overlays" / "img_a.png")
    overlay_b.save(run_dir / "overlays" / "img_b.png")

    metrics = run_dir / "metrics.csv"
    metrics.write_text("image_name,iou,dice,success\nimg_a.png,0.55,0.6,True\nimg_b.png,0.85,0.9,True\n")

    run_config = {
        "dataset_name": "polyp",
        "dataset_root": str(dataset_root),
        "model_name": "gemini-2.5-flash",
        "prompt_family": "desc_v1",
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config))

    config_path = tmp_path / "config.yaml"
    config_payload = {
        "model": "gemini-2.5-flash",
        "prompt_strategy": "desc_v1",
        "results_root": str(results_root),
        "tasks": {"polyp": {"targets": ["colorectal polyp"]}},
    }
    config_path.write_text(yaml.safe_dump(config_payload))

    selection_path = tmp_path / "selection.yaml"
    artifacts_dir = tmp_path / "artifacts"
    pdf_path, png_path = generate_best_case_montage(
        config_path=config_path, selection_path=selection_path, artifacts_dir=artifacts_dir
    )

    assert pdf_path.exists()
    assert png_path.exists()

    saved_selection = yaml.safe_load(selection_path.read_text())
    assert saved_selection["polyp"]["image_name"] == "img_b.png"

    # Change metrics to favor img_a, but persisted selection should remain img_b
    metrics.write_text("image_name,iou,dice,success\nimg_a.png,0.95,0.6,True\nimg_b.png,0.10,0.9,True\n")
    second_pdf, second_png = generate_best_case_montage(
        config_path=config_path, selection_path=selection_path, artifacts_dir=artifacts_dir
    )

    assert second_pdf == pdf_path
    assert second_png == png_path
    persisted = yaml.safe_load(selection_path.read_text())
    assert persisted["polyp"]["image_name"] == "img_b.png"


def test_generate_best_case_montage_validates_selection(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_payload = {
        "model": "gemini-2.5-flash",
        "prompt_strategy": "desc_v1",
        "tasks": {"polyp": {"targets": ["colorectal polyp"]}},
    }
    config_path.write_text(yaml.safe_dump(config_payload))

    selection_path = tmp_path / "selection.yaml"
    selection_payload = {
        "other": {
            "image_name": "foo.png",
            "image_path": "foo.png",
            "gt_mask_path": "bar.png",
            "pred_mask_path": "baz.png",
            "run_dir": "runs",
        }
    }
    selection_path.write_text(yaml.safe_dump(selection_payload))

    with pytest.raises(ValueError):
        generate_best_case_montage(
            config_path=config_path,
            selection_path=selection_path,
            artifacts_dir=tmp_path / "artifacts",
        )
