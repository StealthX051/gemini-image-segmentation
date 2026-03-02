from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from gemini_segmentation.data import (
    DEFAULT_MANIFEST_TEMPLATE,
    paired_masks,
    sample_images,
)

from nanobanana_segmentation.core.types import DatasetItem


def _resolve_data_dirs(dataset_root: Path) -> tuple[Path, Path]:
    images_candidates = [dataset_root / "images", dataset_root / "image"]
    masks_candidates = [dataset_root / "masks", dataset_root / "mask"]

    images_dir = next((p for p in images_candidates if p.is_dir()), None)
    masks_dir = next((p for p in masks_candidates if p.is_dir()), None)

    if images_dir is None or masks_dir is None:
        raise FileNotFoundError(
            "Dataset root must contain images/masks or image/mask directories. "
            f"Checked images candidates={images_candidates}, masks candidates={masks_candidates}"
        )
    return images_dir, masks_dir


def _load_manifest_images(
    *,
    images_dir: Path,
    manifest_path: Path,
    regenerate_if_missing: bool,
) -> List[Path]:
    if manifest_path.exists():
        lines = [line.strip() for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [images_dir / line for line in lines]

    if regenerate_if_missing:
        images = sorted(p for p in images_dir.iterdir() if p.is_file())
        manifest_path.write_text("\n".join(p.name for p in images), encoding="utf-8")
        return images

    raise FileNotFoundError(f"Manifest not found: {manifest_path}")


def load_dataset_items(
    *,
    dataset_name: str,
    dataset_root: Path,
    manifest: Optional[str],
    sample_size: Optional[int],
) -> List[DatasetItem]:
    images_dir, masks_dir = _resolve_data_dirs(dataset_root)
    manifest_name = manifest or DEFAULT_MANIFEST_TEMPLATE.format(dataset=dataset_name)
    manifest_path = Path(manifest_name)
    if not manifest_path.is_absolute():
        manifest_path = dataset_root / manifest_path

    images = _load_manifest_images(
        images_dir=images_dir,
        manifest_path=manifest_path,
        regenerate_if_missing=manifest is None,
    )
    images = sample_images(images, sample_size)
    pairs = paired_masks(images, masks_dir)
    return [DatasetItem(image_path=img, mask_path=mask, split="unknown") for img, mask in pairs]
