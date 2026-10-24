"""Score a predictor over one split and write the per-chip metrics CSV."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

#: Per-chip CSV columns.
PER_CHIP_COLUMNS: tuple[str, ...] = (
    "method",
    "split",
    "chip_id",
    "scene",
    "tp",
    "fp",
    "fn",
    "tn",
    "iou",
    "dice",
)


def score_split(
    method: str,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    manifest: Path,
    chips_dir: Path,
    split: str,
    out_csv: Path,
    masks_out: Path | None = None,
) -> list[dict]:
    """Run ``predict_fn(image) -> 0/1 mask`` on every chip of ``split``; write rows to ``out_csv``.

    When ``masks_out`` is given, also save the predicted masks there as one compressed ``.npz``
    keyed by chip_id (for overlays).
    """
    raise NotImplementedError


def unet_predict_fn(checkpoint: Path, device: str = "cpu", threshold: float = 0.5):
    """Load ``best.pt`` and return a ``predict_fn`` (image uint16 (5,H,W) -> uint8 (H,W))."""
    raise NotImplementedError
