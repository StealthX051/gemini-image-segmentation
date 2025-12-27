from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

from PIL import Image


@dataclass(frozen=True)
class DatasetPaths:
    images_dir: Path
    masks_dir: Path
    manifest_path: Path


DEFAULT_MANIFEST_TEMPLATE = "master_imagelist_{dataset}.txt"


def discover_dataset(dataset_root: Path, dataset_name: str, manifest_filename: str | None = None) -> DatasetPaths:
    images_dir = dataset_root / "images"
    masks_dir = dataset_root / "masks"
    if not images_dir.is_dir() or not masks_dir.is_dir():
        raise FileNotFoundError(
            f"Expected 'images/' and 'masks/' under {dataset_root}, found {images_dir} and {masks_dir}"
        )
    manifest_name = manifest_filename or DEFAULT_MANIFEST_TEMPLATE.format(dataset=dataset_name)
    manifest_path = (dataset_root / manifest_name) if not Path(manifest_name).is_absolute() else Path(manifest_name)
    return DatasetPaths(images_dir=images_dir, masks_dir=masks_dir, manifest_path=manifest_path)


def read_manifest(
    paths: DatasetPaths,
    *,
    fallbacks: Iterable[Path] | None = None,
    regenerate_if_missing: bool = True,
) -> List[Path]:
    manifest_candidates: List[Path] = [paths.manifest_path]
    if fallbacks:
        manifest_candidates.extend(fallbacks)

    for manifest in manifest_candidates:
        if manifest.exists():
            logging.info("Loading manifest from %s", manifest)
            lines = [line.strip() for line in manifest.read_text().splitlines() if line.strip()]
            return [paths.images_dir / line for line in lines]

    if regenerate_if_missing:
        logging.info("Manifest missing at %s; regenerating from images directory", manifest_candidates[0])
        images = sorted(p for p in paths.images_dir.iterdir() if p.is_file())
        manifest_candidates[0].write_text("\n".join(p.name for p in images))
        return images

    raise FileNotFoundError(
        f"None of the manifest candidates exist: {', '.join(str(m) for m in manifest_candidates)}"
    )


def sample_images(images: List[Path], sample_size: int | None) -> List[Path]:
    if sample_size is None or sample_size >= len(images):
        return images
    return images[:sample_size]


def paired_masks(images: Iterable[Path], masks_dir: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for img in images:
        mask_path = masks_dir / img.name
        if not mask_path.exists():
            logging.warning("Mask missing for %s; skipping", img.name)
            continue
        pairs.append((img, mask_path))
    return pairs


def load_image(path: Path) -> Image.Image:
    return Image.open(path)
