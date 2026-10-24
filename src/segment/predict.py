"""Score a predictor over one split and write the per-chip metrics CSV."""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np

import segment.data as data
from segment.metrics import confusion, dice, iou

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


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.6f}"
    return str(value)


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

    ``chips_dir`` is the root that each manifest ``image_path_rel`` is relative to (``None``
    means the manifest's folder). When ``masks_out`` is given, also save the predicted masks
    there as one compressed ``.npz`` keyed by chip_id (for overlays).
    """
    manifest = Path(manifest)
    root = Path(chips_dir) if chips_dir is not None else manifest.parent
    records = sorted(
        (r for r in data.read_manifest(manifest) if r.split == split), key=lambda r: r.chip_id
    )
    if not records:
        raise ValueError(f"{manifest}: no chips in split {split!r}")

    rows: list[dict] = []
    masks: dict[str, np.ndarray] = {}
    for rec in records:
        image, target = data.load_chip(root / rec.image_path_rel)
        pred = np.asarray(predict_fn(image))
        if pred.shape != target.shape:
            raise ValueError(f"{rec.chip_id}: prediction {pred.shape} vs mask {target.shape}")
        pred = (pred != 0).astype(np.uint8)
        counts = confusion(pred, target)
        rows.append(
            {
                "method": method,
                "split": split,
                "chip_id": rec.chip_id,
                "scene": rec.scene,
                "tp": counts.tp,
                "fp": counts.fp,
                "fn": counts.fn,
                "tn": counts.tn,
                "iou": iou(counts),
                "dice": dice(counts),
            }
        )
        masks[rec.chip_id] = pred

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(PER_CHIP_COLUMNS)
    for row in rows:
        writer.writerow([_fmt(row[c]) for c in PER_CHIP_COLUMNS])
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_bytes(buf.getvalue().encode("utf-8"))

    if masks_out is not None:
        masks_out = Path(masks_out)
        masks_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(masks_out, **masks)
    return rows


def unet_predict_fn(checkpoint: Path, device: str = "cpu", threshold: float = 0.5):
    """Load ``best.pt`` and return a ``predict_fn`` (image uint16 (5,H,W) -> uint8 (H,W))."""
    import torch

    from segment.model import build_unet, crop_to, pad_to_multiple

    ckpt = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    model = build_unet(ckpt["encoder"], None, ckpt["in_channels"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    @torch.no_grad()
    def predict_fn(image: np.ndarray) -> np.ndarray:
        x = data.to_model_input(image)
        h, w = x.shape[-2:]
        x = pad_to_multiple(x, mode="reflect")
        logits = model(torch.from_numpy(np.ascontiguousarray(x))[None].to(device))
        prob = torch.sigmoid(logits.float())[0, 0].cpu().numpy()
        return crop_to(prob > threshold, h, w).astype(np.uint8)

    return predict_fn
