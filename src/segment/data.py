"""Chip manifest (the source of truth for split membership) and chip loading."""

from __future__ import annotations

import csv
import hashlib
import io
from collections import defaultdict
from dataclasses import astuple, dataclass, fields
from pathlib import Path

import numpy as np

from segment.bands import BAND_INDEX, BANDS, IGNORE, MODEL_BANDS, REFLECTANCE_SCALE

#: Manifest column order. One row per chip.
#: ``scene`` is the leakage unit (one S1S2-Water sample = one 100 km Sentinel-2 tile and date);
#: ``s2_tile`` is the MGRS tile id, checked separately so two scenes on one tile can't straddle
#: splits.
MANIFEST_COLUMNS: tuple[str, ...] = (
    "chip_id",
    "scene",
    "s2_tile",
    "split",
    "row",
    "col",
    "valid_frac",
    "water_frac",
    "image_path_rel",
    "licence",
    "source_url",
    "sha256",
)
SPLITS: tuple[str, ...] = ("train", "val", "test")


@dataclass(frozen=True)
class ChipRecord:
    chip_id: str
    scene: str
    s2_tile: str
    split: str
    row: int
    col: int
    valid_frac: float
    water_frac: float
    image_path_rel: str
    licence: str
    source_url: str
    sha256: str


assert tuple(f.name for f in fields(ChipRecord)) == MANIFEST_COLUMNS


def read_manifest(path: Path) -> list[ChipRecord]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
            raise ValueError(f"{path}: columns {reader.fieldnames} != {MANIFEST_COLUMNS}")
        return [
            ChipRecord(
                chip_id=r["chip_id"],
                scene=r["scene"],
                s2_tile=r["s2_tile"],
                split=r["split"],
                row=int(r["row"]),
                col=int(r["col"]),
                valid_frac=float(r["valid_frac"]),
                water_frac=float(r["water_frac"]),
                image_path_rel=r["image_path_rel"],
                licence=r["licence"],
                source_url=r["source_url"],
                sha256=r["sha256"],
            )
            for r in reader
        ]


def _fmt(value: object) -> str:
    return f"{value:.6f}" if isinstance(value, float) else str(value)


def write_manifest(path: Path, records: list[ChipRecord]) -> None:
    """Sorted by chip_id, LF line endings, so a rebuild is byte-identical."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(MANIFEST_COLUMNS)
    for rec in sorted(records, key=lambda r: r.chip_id):
        writer.writerow([_fmt(v) for v in astuple(rec)])
    Path(path).write_bytes(buf.getvalue().encode("utf-8"))


def check_splits(records: list[ChipRecord]) -> list[str]:
    """Leakage audit. Returns human-readable violations; empty list means clean.

    Checks: every split is a known name; chip_ids unique; no scene in two splits;
    no s2_tile in two splits; every split non-empty.
    """
    problems: list[str] = []
    seen: set[str] = set()
    for rec in records:
        if rec.split not in SPLITS:
            problems.append(f"chip {rec.chip_id}: unknown split {rec.split!r}")
        if rec.chip_id in seen:
            problems.append(f"chip {rec.chip_id}: duplicate chip_id")
        seen.add(rec.chip_id)
    for key in ("scene", "s2_tile"):
        splits_of: dict[str, set[str]] = defaultdict(set)
        for rec in records:
            splits_of[getattr(rec, key)].add(rec.split)
        for value, splits in sorted(splits_of.items()):
            if len(splits) > 1:
                problems.append(f"{key} {value} appears in splits {sorted(splits)}")
    present = {rec.split for rec in records}
    problems.extend(f"split {s!r} is empty" for s in SPLITS if s not in present)
    return problems


def load_chip(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(image uint16 (5,H,W), mask uint8 (H,W)) from a chip .npz."""
    with np.load(path) as z:
        image, mask = z["image"], z["mask"]
    if image.dtype != np.uint16 or image.ndim != 3 or image.shape[0] != len(BANDS):
        raise ValueError(f"{path}: image {image.dtype} {image.shape}, want uint16 (5,H,W)")
    if mask.dtype != np.uint8 or mask.shape != image.shape[1:]:
        raise ValueError(f"{path}: mask {mask.dtype} {mask.shape}, want uint8 {image.shape[1:]}")
    return image, mask


def content_sha256(image: np.ndarray, mask: np.ndarray) -> str:
    """Hash of the chip's arrays (shapes, dtypes and C-order bytes), not of the compressed file.

    Compressed bytes depend on the zlib build, so a chip rebuilt on another machine is checked
    against the manifest by content.
    """
    h = hashlib.sha256()
    for arr in (image, mask):
        h.update(f"{arr.dtype.str}{arr.shape}".encode())
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def save_chip(path: Path, image: np.ndarray, mask: np.ndarray) -> str:
    """Write a compressed chip .npz and return its :func:`content_sha256`."""
    if image.dtype != np.uint16 or mask.dtype != np.uint8:
        raise ValueError("image must be uint16 and mask uint8")
    if not np.isin(mask, (0, 1, IGNORE)).all():
        raise ValueError("mask values must be 0, 1 or IGNORE")
    buf = io.BytesIO()
    np.savez_compressed(buf, image=image, mask=mask)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.getvalue())
    return content_sha256(image, mask)


def to_model_input(image: np.ndarray) -> np.ndarray:
    """float32 (4,H,W): MODEL_BANDS selected, divided by REFLECTANCE_SCALE, clipped to [0, 1]."""
    idx = [BAND_INDEX[b] for b in MODEL_BANDS]
    x = image[idx].astype(np.float32) / REFLECTANCE_SCALE
    return np.clip(x, 0.0, 1.0)
