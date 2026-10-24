"""Overlay PNGs: RGB chip with true-positive / false-positive / false-negative pixels coloured."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from segment.bands import BAND_INDEX, IGNORE, REFLECTANCE_SCALE
from segment.metrics import confusion, iou

TP_COLOUR = (0, 255, 255)
FP_COLOUR = (255, 140, 0)
FN_COLOUR = (255, 0, 255)
IGNORE_GREY = 90
ALPHA = 0.5
GAP = 2
TITLE_HEIGHT = 12
BACKGROUND = (255, 255, 255)


def rgb_preview(image: np.ndarray, gain: float = 3.5) -> np.ndarray:
    """uint8 (H,W,3) natural-colour preview from a chip image (L1C reflectance looks dark)."""
    bands = [BAND_INDEX["red"], BAND_INDEX["green"], BAND_INDEX["blue"]]
    rgb = image[bands].astype(np.float32).transpose(1, 2, 0) / REFLECTANCE_SCALE * gain
    return np.rint(np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def overlay(image: np.ndarray, target: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """uint8 (H,W,3): TP cyan, FP orange, FN magenta, IGNORE greyed, blended over the preview."""
    base = rgb_preview(image).astype(np.float32)
    valid = target != IGNORE
    p = pred.astype(bool) & valid
    t = (target == 1) & valid
    out = base.copy()
    for mask, colour in ((p & t, TP_COLOUR), (p & ~t, FP_COLOUR), (~p & t, FN_COLOUR)):
        out[mask] = (1 - ALPHA) * base[mask] + ALPHA * np.array(colour, dtype=np.float32)
    out[~valid] = 0.4 * base[~valid] + 0.6 * IGNORE_GREY
    return np.rint(out).astype(np.uint8)


def _target_tile(target: np.ndarray) -> np.ndarray:
    tile = np.zeros((*target.shape, 3), dtype=np.uint8)
    tile[target == 1] = 255
    tile[target == IGNORE] = 128
    return tile


def _titled(tile: np.ndarray, title: str) -> Image.Image:
    """Tile with a title strip above it, clipped to the tile width."""
    h, w = tile.shape[:2]
    img = Image.new("RGB", (w, TITLE_HEIGHT + h), BACKGROUND)
    ImageDraw.Draw(img).text((1, 0), title, fill=(0, 0, 0))
    img.paste(Image.fromarray(tile), (0, TITLE_HEIGHT))
    return img


def save_panel(
    path: Path, image: np.ndarray, target: np.ndarray, preds: dict[str, np.ndarray]
) -> None:
    """One PNG row: RGB | target | one overlay per method, each titled with its chip IoU."""
    tiles = [_titled(rgb_preview(image), "RGB"), _titled(_target_tile(target), "Target")]
    for name, pred in preds.items():
        score = iou(confusion(pred, target))
        label = "n/a" if score != score else f"{score:.3f}"
        tiles.append(_titled(overlay(image, target, pred), f"{name} IoU {label}"))
    h, w = target.shape
    canvas = Image.new(
        "RGB", (len(tiles) * w + (len(tiles) - 1) * GAP, TITLE_HEIGHT + h), BACKGROUND
    )
    for i, tile in enumerate(tiles):
        canvas.paste(tile, (i * (w + GAP), 0))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG")
