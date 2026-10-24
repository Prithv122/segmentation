"""Pixel metrics for binary water masks. Pixels marked IGNORE never count."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from segment.bands import IGNORE


@dataclass(frozen=True)
class Counts:
    """Confusion counts over the non-ignored pixels of one or more chips."""

    tp: int
    fp: int
    fn: int
    tn: int

    def __add__(self, other: Counts) -> Counts:
        return Counts(
            self.tp + other.tp, self.fp + other.fp, self.fn + other.fn, self.tn + other.tn
        )

    @staticmethod
    def zero() -> Counts:
        return Counts(0, 0, 0, 0)


def confusion(pred: np.ndarray, target: np.ndarray, ignore: int = IGNORE) -> Counts:
    """Counts for a 0/1 prediction against a {0, 1, ignore} target of the same shape."""
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs target {target.shape}")
    valid = target != ignore
    p = pred.astype(bool) & valid
    t = (target == 1) & valid
    tp = int(np.count_nonzero(p & t))
    fp = int(np.count_nonzero(p & ~t))
    fn = int(np.count_nonzero(~p & t))
    tn = int(np.count_nonzero(valid)) - tp - fp - fn
    return Counts(tp, fp, fn, tn)


def _ratio(num: int, den: int) -> float:
    return float("nan") if den == 0 else num / den


def iou(c: Counts) -> float:
    """tp / (tp + fp + fn). NaN when the denominator is 0 (no water predicted or present)."""
    return _ratio(c.tp, c.tp + c.fp + c.fn)


def dice(c: Counts) -> float:
    """2tp / (2tp + fp + fn). NaN when the denominator is 0."""
    return _ratio(2 * c.tp, 2 * c.tp + c.fp + c.fn)


def precision(c: Counts) -> float:
    return _ratio(c.tp, c.tp + c.fp)


def recall(c: Counts) -> float:
    return _ratio(c.tp, c.tp + c.fn)


def _as_array(counts: Sequence[Counts]) -> np.ndarray:
    return np.array([[c.tp, c.fp, c.fn, c.tn] for c in counts], dtype=np.int64).reshape(-1, 4)


def pooled(counts: Sequence[Counts]) -> Counts:
    """Sum of per-chip counts (the pixel-weighted aggregate)."""
    tp, fp, fn, tn = (int(v) for v in _as_array(counts).sum(axis=0))
    return Counts(tp, fp, fn, tn)


def bootstrap_ci(
    counts: Sequence[Counts],
    metric: Callable[[Counts], float] = iou,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile CI of a pooled (pixel-weighted) metric, resampling chips with replacement."""
    arr = _as_array(counts)
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    sums = arr[idx].sum(axis=1)
    stats = np.array([metric(Counts(*(int(v) for v in row))) for row in sums])
    stats = stats[~np.isnan(stats)]
    if len(stats) == 0:
        return (float("nan"), float("nan"))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))
