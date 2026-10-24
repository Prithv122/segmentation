"""Classical baselines: an index thresholded by Otsu (fitted on train chips only) and a constant."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's threshold of a 1-D sample: the cut maximising between-class variance."""
    raise NotImplementedError


@dataclass(frozen=True)
class IndexThreshold:
    """Predict water where ``index(image) > threshold``."""

    index: str  # "ndwi" or "mndwi"
    threshold: float

    def predict(self, image: np.ndarray) -> np.ndarray:
        """uint8 (H, W) 0/1 mask."""
        raise NotImplementedError


def fit_index_threshold(
    chips: Iterable[tuple[np.ndarray, np.ndarray]],
    index: str = "ndwi",
    max_pixels_per_chip: int = 20000,
    seed: int = 0,
) -> IndexThreshold:
    """Fit Otsu on index values of valid (non-IGNORE) pixels, sampled from (image, mask) chips."""
    raise NotImplementedError


@dataclass(frozen=True)
class Constant:
    """Always predict one class: the train set's majority class (``fit_constant``)."""

    value: int

    def predict(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def fit_constant(masks: Iterable[np.ndarray]) -> Constant:
    raise NotImplementedError
