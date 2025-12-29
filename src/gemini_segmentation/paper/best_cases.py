from __future__ import annotations

"""Render best-case qualitative examples for Figure 1.

This module loads per-image metrics and masks from completed segmentation runs,
selects deterministic "best-case" examples per dataset/target (by default the
max IoU), and renders a montage showing the source image, ground truth overlay,
and prediction overlay with IoU and bounding-box labels. Selections can be
persisted to a YAML file so future runs are reproducible.
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import matplotlib

# Use a non-interactive backend for headless environments.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont

from ..data import discover_dataset
from ..io import overlay_mask_on_img

LOGGER = logging.getLogger(__name__)

DEFAULT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = DEFAULT_ROOT / "configs" / "figure1_best_cases.yaml"
DEFAULT_RESULTS_ROOT = DEFAULT_ROOT / "results"
DEFAULT_ARTIFACTS_DIR = DEFAULT_ROOT / "artifacts" / "figures" / "figure1_best_cases"
DEFAULT_SELECTION_PATH = DEFAULT_ARTIFACTS_DIR / "selection.yaml"


@dataclass
class TaskSelection:
    """Config for a single dataset."""

    dataset: str
    targets: Tuple[str, ...]
    run_dir: Optional[Path] = None


@dataclass
class BestCaseConfig:
    """Top-level selection parameters."""

    model: str
    prompt_strategy: str
    tasks: Mapping[str, TaskSelection]
    results_root: Path = DEFAULT_RESULTS_ROOT


@dataclass
class SelectedImage:
    dataset: str
    image_name: str
    iou: float
    run_dir: Path
    image_path: Path
    gt_mask_path: Path
    pred_mask_path: Path
    overlay_path: Optional[Path]


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return yaml.safe_load(path.read_text()) or {}


def _parse_task(name: str, payload: Mapping[str, object]) -> TaskSelection:
    targets = tuple(str(t) for t in payload.get("targets", []) if t)
    if not targets:
        raise ValueError(f"Task '{name}' must define at least one target")
    run_dir = payload.get("run_dir")
    run_dir_path = Path(run_dir) if run_dir else None
    return TaskSelection(dataset=name, targets=targets, run_dir=run_dir_path)


def load_best_case_config(path: Path = DEFAULT_CONFIG_PATH) -> BestCaseConfig:
    """Load the best-case selection config from YAML."""

    raw = _load_yaml(path)
    model = raw.get("model")
    prompt_strategy = raw.get("prompt_strategy")
    results_root = Path(raw.get("results_root", DEFAULT_RESULTS_ROOT))
    tasks_payload = raw.get("tasks", {})

    if not model:
        raise ValueError("Config must define 'model'")
    if not prompt_strategy:
        raise ValueError("Config must define 'prompt_strategy'")
    if not tasks_payload:
        raise ValueError("Config must include at least one task")

    tasks = {name: _parse_task(name, payload) for name, payload in tasks_payload.items()}
    return BestCaseConfig(model=str(model), prompt_strategy=str(prompt_strategy), tasks=tasks, results_root=results_root)


def _discover_candidate_runs(dataset: str, model: str, prompt_strategy: str, base_results: Path) -> List[Path]:
    dataset_dir = base_results / dataset / model
    if not dataset_dir.exists():
        return []

    candidates: List[Path] = []
    for prompt_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in prompt_dir.iterdir() if p.is_dir()):
            run_config_path = run_dir / "run_config.json"
            if not run_config_path.exists():
                continue
            try:
                run_cfg = json.loads(run_config_path.read_text())
            except json.JSONDecodeError:
                continue
            if (
                str(run_cfg.get("dataset_name")) == dataset
                and str(run_cfg.get("model_name")) == model
                and str(run_cfg.get("prompt_family")) == prompt_strategy
            ):
                candidates.append(run_dir)
    return candidates


def _select_run_dir(task: TaskSelection, *, base_results: Path, model: str, prompt_strategy: str) -> Path:
    if task.run_dir:
        return task.run_dir

    matches = _discover_candidate_runs(task.dataset, model, prompt_strategy, base_results)
    if not matches:
        raise FileNotFoundError(
            f"No runs found for dataset={task.dataset}, model={model}, prompt_strategy={prompt_strategy} under {base_results}"
        )

    chosen = sorted(matches)[-1]
    LOGGER.info("Using run %s for dataset %s", chosen, task.dataset)
    return chosen


def _load_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Per-image metrics not found at {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Metrics file {path} is empty")
    return df


def _best_image_name(metrics: pd.DataFrame) -> Tuple[str, float]:
    ordered = metrics.sort_values(by=["iou", "image_name"], ascending=[False, True])
    top = ordered.iloc[0]
    return str(top.image_name), float(top.iou)


def _load_run_config(run_dir: Path) -> Mapping[str, object]:
    run_config_path = run_dir / "run_config.json"
    if not run_config_path.exists():
        raise FileNotFoundError(f"run_config.json missing at {run_config_path}")
    return json.loads(run_config_path.read_text())


def _selection_from_yaml(path: Path) -> Dict[str, SelectedImage]:
    payload = _load_yaml(path)
    selections: Dict[str, SelectedImage] = {}
    for dataset, data in payload.items():
        selections[dataset] = SelectedImage(
            dataset=dataset,
            image_name=str(data["image_name"]),
            iou=float(data.get("iou", 0.0)),
            run_dir=Path(data["run_dir"]),
            image_path=Path(data["image_path"]),
            gt_mask_path=Path(data["gt_mask_path"]),
            pred_mask_path=Path(data["pred_mask_path"]),
            overlay_path=Path(data["overlay_path"]) if data.get("overlay_path") else None,
        )
    return selections


def _validate_loaded_selection(selection: Mapping[str, SelectedImage], tasks: Mapping[str, TaskSelection]) -> None:
    missing = [dataset for dataset in tasks if dataset not in selection]
    if missing:
        raise ValueError(
            "Persisted selection does not cover configured datasets: " + ", ".join(sorted(missing))
        )

    extras = [dataset for dataset in selection if dataset not in tasks]
    if extras:
        raise ValueError(
            "Persisted selection includes datasets not present in config: " + ", ".join(sorted(extras))
        )

    for dataset, sel in selection.items():
        for path_name in ["image_path", "gt_mask_path", "pred_mask_path"]:
            candidate = getattr(sel, path_name)
            if not candidate.exists():
                raise FileNotFoundError(
                    f"Persisted selection for {dataset} references missing file: {candidate}"
                )
        if sel.run_dir and not sel.run_dir.exists():
            raise FileNotFoundError(
                f"Persisted selection for {dataset} references missing run directory: {sel.run_dir}"
            )


def _persist_selection(path: Path, selections: Mapping[str, SelectedImage]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: MutableMapping[str, dict] = {}
    for dataset, sel in selections.items():
        payload[dataset] = {
            "image_name": sel.image_name,
            "iou": sel.iou,
            "run_dir": str(sel.run_dir),
            "image_path": str(sel.image_path),
            "gt_mask_path": str(sel.gt_mask_path),
            "pred_mask_path": str(sel.pred_mask_path),
            "overlay_path": str(sel.overlay_path) if sel.overlay_path else None,
        }
    path.write_text(yaml.safe_dump(payload, sort_keys=True))
    LOGGER.info("Persisted selection to %s", path)


def _select_best_images(config: BestCaseConfig, selection_path: Optional[Path]) -> Dict[str, SelectedImage]:
    if selection_path and selection_path.exists():
        LOGGER.info("Loading existing selection from %s", selection_path)
        selections = _selection_from_yaml(selection_path)
        _validate_loaded_selection(selections, config.tasks)
        return selections

    selections: Dict[str, SelectedImage] = {}
    for dataset, task in config.tasks.items():
        run_dir = _select_run_dir(task, base_results=config.results_root, model=config.model, prompt_strategy=config.prompt_strategy)
        run_cfg = _load_run_config(run_dir)
        dataset_root = Path(run_cfg.get("dataset_root", "")).expanduser().resolve()
        dataset_paths = discover_dataset(dataset_root, dataset)

        metrics = _load_metrics(run_dir / "metrics.csv")
        image_name, iou = _best_image_name(metrics)

        image_path = dataset_paths.images_dir / image_name
        gt_mask_path = dataset_paths.masks_dir / image_name
        pred_mask_path = run_dir / "masks" / image_name
        overlay_path = run_dir / "overlays" / image_name
        overlay_resolved = overlay_path if overlay_path.exists() else None

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found for selection: {image_path}")
        if not gt_mask_path.exists():
            raise FileNotFoundError(f"Ground truth mask missing for {image_name} at {gt_mask_path}")
        if not pred_mask_path.exists():
            raise FileNotFoundError(f"Prediction mask missing for {image_name} at {pred_mask_path}")

        selections[dataset] = SelectedImage(
            dataset=dataset,
            image_name=image_name,
            iou=iou,
            run_dir=run_dir,
            image_path=image_path,
            gt_mask_path=gt_mask_path,
            pred_mask_path=pred_mask_path,
            overlay_path=overlay_resolved,
        )

    if selection_path:
        _persist_selection(selection_path, selections)

    return selections


def _bounding_box(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0 or xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _annotate_image(img: Image.Image, text: str, *, bbox: Optional[Tuple[int, int, int, int]] = None) -> Image.Image:
    annotated = img.copy().convert("RGBA")
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("arial.ttf", size=18)
    except OSError:
        font = ImageFont.load_default()
    padding = 6
    text_width = font.getlength(text)
    draw.rectangle(
        [(padding, padding), (padding + 4 + text_width, padding + 24)], fill=(0, 0, 0, 180)
    )
    draw.text((padding + 2, padding + 4), text, fill=(255, 255, 255, 255), font=font)
    if bbox:
        draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline=(255, 255, 0, 255), width=3)
    return annotated.convert("RGB")


def _build_panels(selections: Mapping[str, SelectedImage], tasks: Mapping[str, TaskSelection]) -> List[Tuple[str, Image.Image, Image.Image, Image.Image]]:
    rows: List[Tuple[str, Image.Image, Image.Image, Image.Image]] = []
    for dataset, selection in selections.items():
        task = tasks[dataset]
        base_img = Image.open(selection.image_path).convert("RGB")
        gt_mask = np.array(Image.open(selection.gt_mask_path))
        pred_mask = np.array(Image.open(selection.pred_mask_path))

        gt_overlay = overlay_mask_on_img(base_img, gt_mask, "lime")
        bbox = _bounding_box(pred_mask)
        if selection.overlay_path and selection.overlay_path.exists():
            pred_panel = Image.open(selection.overlay_path).convert("RGB")
        else:
            pred_panel = overlay_mask_on_img(base_img, pred_mask, "red")
        label = ", ".join(task.targets)
        pred_text = f"IoU={selection.iou:.3f} | {label}"
        pred_panel = _annotate_image(pred_panel, pred_text, bbox=bbox)
        rows.append((dataset, base_img, gt_overlay, pred_panel))
    return rows


def _render_montage(rows: Iterable[Tuple[str, Image.Image, Image.Image, Image.Image]], output_dir: Path) -> Tuple[Path, Path]:
    rows_list = list(rows)
    n_rows = len(rows_list)
    if n_rows == 0:
        raise ValueError("No selections available to render")

    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 4 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)  # type: ignore[assignment]

    column_titles = ["Input", "Ground truth", "Prediction"]
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title)

    for row_idx, (dataset, src, gt, pred) in enumerate(rows_list):
        panels = [src, gt, pred]
        for col_idx, panel in enumerate(panels):
            axes[row_idx, col_idx].imshow(panel)
            axes[row_idx, col_idx].axis("off")
            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel(dataset, fontsize=12)

    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "figure1_best_cases.pdf"
    png_path = output_dir / "figure1_best_cases.png"
    fig.savefig(pdf_path, dpi=300)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    LOGGER.info("Best-case montage saved under %s", output_dir)
    return pdf_path, png_path


def generate_best_case_montage(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    selection_path: Path = DEFAULT_SELECTION_PATH,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
) -> Tuple[Path, Path]:
    """Select best-case examples and render Figure 1 montage."""

    config = load_best_case_config(config_path)
    selections = _select_best_images(config, selection_path)
    rows = _build_panels(selections, config.tasks)
    return _render_montage(rows, artifacts_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Figure 1 best-case montage")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to the selection config YAML"
    )
    parser.add_argument(
        "--selection", type=Path, default=DEFAULT_SELECTION_PATH, help="Optional path to persist or read selections"
    )
    parser.add_argument(
        "--artifacts", type=Path, default=DEFAULT_ARTIFACTS_DIR, help="Directory for rendered PDF/PNG outputs"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    generate_best_case_montage(
        config_path=args.config,
        selection_path=args.selection,
        artifacts_dir=args.artifacts,
    )


if __name__ == "__main__":
    main()
