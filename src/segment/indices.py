"""Spectral water indices on a chip's ``image`` array (see :mod:`segment.bands`)."""

from __future__ import annotations

import numpy as np

from segment.bands import BAND_INDEX


def _normalised_difference(image: np.ndarray, a: str, b: str) -> np.ndarray:
    """(a - b) / (a + b) in float, never in the chip's uint16; 0 where the sum is 0."""
    x = image[BAND_INDEX[a]].astype(np.float32)
    y = image[BAND_INDEX[b]].astype(np.float32)
    den = x + y
    out = np.zeros(den.shape, dtype=np.float32)
    np.divide(x - y, den, out=out, where=den != 0)
    return out


def ndwi(image: np.ndarray) -> np.ndarray:
    """McFeeters NDWI = (green - nir) / (green + nir), float32 (H, W); 0 where the sum is 0."""
    return _normalised_difference(image, "green", "nir")


def mndwi(image: np.ndarray) -> np.ndarray:
    """Xu MNDWI = (green - swir1) / (green + swir1), float32 (H, W); 0 where the sum is 0."""
    return _normalised_difference(image, "green", "swir1")
