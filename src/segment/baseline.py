"""Classical baselines: an index thresholded by Otsu (fitted on train chips only) and a constant."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from segment.bands import IGNORE
from segment.indices import mndwi, ndwi

_INDEX_FNS: dict[str, Callable[[np.ndarray], np.ndarray]] = {"ndwi": ndwi, "mndwi": mndwi}


def _index_fn(name: str) -> Callable[[np.ndarray], np.ndarray]:
    try:
        return _INDEX_FNS[name]
    except KeyError:
        raise ValueError(f"unknown index {name!r}, want one of {sorted(_INDEX_FNS)}") from None


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's threshold of a 1-D sample: the cut maximising between-class variance.

    The sample is histogrammed into ``bins`` equal-width bins over [min, max]. The returned
    threshold is the *upper edge* of the last bin assigned to the low class, so it pairs with
    ``value > threshold`` meaning high class. When several cuts tie for the maximum (an empty
    gap between two modes), the middle of that plateau is used. A constant sample returns that
    constant. Non-finite values are ignored.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("otsu_threshold needs at least one finite value")
    lo, hi = float(v.min()), float(v.max())
    if lo == hi:
        return lo
    hist, edges = np.histogram(v, bins=bins, range=(lo, hi))
    hist = hist.astype(np.float64)
    centres = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    s0 = np.cumsum(hist * centres)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = s0 / w0
        mu1 = (s0[-1] - s0) / w1
        between = w0 * w1 * (mu0 - mu1) ** 2
    between = np.nan_to_num(between, nan=0.0)
    ties = np.flatnonzero(between >= between.max() * (1 - 1e-9))
    k = int(ties[(len(ties) - 1) // 2])
    return float(edges[k + 1])


@dataclass(frozen=True)
class IndexThreshold:
    """Predict water where ``index(image) > threshold``."""

    index: str  # "ndwi" or "mndwi"
    threshold: float

    def predict(self, image: np.ndarray) -> np.ndarray:
        """uint8 (H, W) 0/1 mask."""
        return (_index_fn(self.index)(image) > self.threshold).astype(np.uint8)


def fit_index_threshold(
    chips: Iterable[tuple[np.ndarray, np.ndarray]],
    index: str = "ndwi",
    max_pixels_per_chip: int = 20000,
    seed: int = 0,
) -> IndexThreshold:
    """Fit Otsu on index values of valid (non-IGNORE) pixels, sampled from (image, mask) chips."""
    fn = _index_fn(index)
    rng = np.random.default_rng(seed)
    samples = []
    for image, mask in chips:
        values = fn(image)[mask != IGNORE]
        if values.size > max_pixels_per_chip:
            values = values[rng.choice(values.size, size=max_pixels_per_chip, replace=False)]
        samples.append(values)
    if not samples or sum(s.size for s in samples) == 0:
        raise ValueError("no valid pixels to fit a threshold on")
    return IndexThreshold(index=index, threshold=otsu_threshold(np.concatenate(samples)))


@dataclass(frozen=True)
class Constant:
    """Always predict one class: the train set's majority class (``fit_constant``)."""

    value: int

    def predict(self, image: np.ndarray) -> np.ndarray:
        return np.full(image.shape[1:], self.value, dtype=np.uint8)


def fit_constant(masks: Iterable[np.ndarray]) -> Constant:
    """Majority class over valid pixels; ties (and no valid pixels) give 0."""
    water = land = 0
    for mask in masks:
        water += int(np.count_nonzero(mask == 1))
        land += int(np.count_nonzero(mask == 0))
    return Constant(1 if water > land else 0)
