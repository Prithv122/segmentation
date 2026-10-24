"""Overlay PNGs: RGB chip with true-positive / false-positive / false-negative pixels coloured."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def rgb_preview(image: np.ndarray, gain: float = 3.5) -> np.ndarray:
    """uint8 (H,W,3) natural-colour preview from a chip image (L1C reflectance looks dark)."""
    raise NotImplementedError


def overlay(image: np.ndarray, target: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """uint8 (H,W,3): TP cyan, FP orange, FN magenta, IGNORE greyed, blended over the preview."""
    raise NotImplementedError


def save_panel(
    path: Path, image: np.ndarray, target: np.ndarray, preds: dict[str, np.ndarray]
) -> None:
    """One PNG row: RGB | target | one overlay per method, each titled with its chip IoU."""
    raise NotImplementedError
