"""Spectral water indices on a chip's ``image`` array (see :mod:`segment.bands`)."""

import numpy as np


def ndwi(image: np.ndarray) -> np.ndarray:
    """McFeeters NDWI = (green - nir) / (green + nir), float32 (H, W); 0 where the sum is 0."""
    raise NotImplementedError


def mndwi(image: np.ndarray) -> np.ndarray:
    """Xu MNDWI = (green - swir1) / (green + swir1), float32 (H, W); 0 where the sum is 0."""
    raise NotImplementedError
