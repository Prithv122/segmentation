"""Synthetic chips with known masks: every metric on them has a hand-computable value.

Eight 32x32 chips across four scenes: two train scenes (A, B), one val scene (C), one test
scene (D). Each chip holds one circular "water" blob (high green, low NIR, so NDWI > 0) on
"land" (low green, high NIR, so NDWI < 0). Chip ``*_1`` also has an ignored 4x4 corner.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from segment.bands import BAND_INDEX, BANDS, IGNORE
from segment.data import ChipRecord, save_chip, write_manifest

SIZE = 32
SCENES = {"A": "train", "B": "train", "C": "val", "D": "test"}
TILES = {"A": "T01AAA", "B": "T02BBB", "C": "T03CCC", "D": "T04DDD"}
#: (centre_row, centre_col, radius) per chip index within a scene.
BLOBS = [(10, 12, 6), (20, 18, 8)]

WATER_REFLECTANCE = {"blue": 900, "green": 800, "red": 500, "nir": 200, "swir1": 100}
LAND_REFLECTANCE = {"blue": 500, "green": 700, "red": 800, "nir": 3000, "swir1": 2000}


def blob_mask(cy: int, cx: int, r: int, size: int = SIZE) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    return ((yy - cy) ** 2 + (xx - cx) ** 2 <= r * r).astype(np.uint8)


def chip_arrays(cy: int, cx: int, r: int, ignore_corner: bool) -> tuple[np.ndarray, np.ndarray]:
    water = blob_mask(cy, cx, r)
    image = np.empty((len(BANDS), SIZE, SIZE), dtype=np.uint16)
    for band in BANDS:
        image[BAND_INDEX[band]] = np.where(
            water == 1, WATER_REFLECTANCE[band], LAND_REFLECTANCE[band]
        )
    mask = water.copy()
    if ignore_corner:
        mask[:4, :4] = IGNORE
    return image, mask


def make_fixture(out_dir: Path) -> Path:
    """Write ``out_dir/chips/*.npz`` and ``out_dir/manifest.csv``; return the manifest path."""
    out_dir = Path(out_dir)
    records = []
    for scene, split in SCENES.items():
        for i, (cy, cx, r) in enumerate(BLOBS):
            chip_id = f"{scene}_{i}"
            image, mask = chip_arrays(cy, cx, r, ignore_corner=(i == 1))
            rel = f"chips/{chip_id}.npz"
            sha = save_chip(out_dir / rel, image, mask)
            valid = mask != IGNORE
            records.append(
                ChipRecord(
                    chip_id=chip_id,
                    scene=scene,
                    s2_tile=TILES[scene],
                    split=split,
                    row=0,
                    col=i * SIZE,
                    valid_frac=float(valid.mean()),
                    water_frac=float((mask == 1).sum() / valid.sum()),
                    image_path_rel=rel,
                    licence="synthetic",
                    source_url="tests/fixtures/make_fixture.py",
                    sha256=sha,
                )
            )
    manifest = out_dir / "manifest.csv"
    write_manifest(manifest, records)
    return manifest
