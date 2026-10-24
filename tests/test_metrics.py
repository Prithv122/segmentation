from __future__ import annotations

import math

import numpy as np
import pytest

from segment.bands import IGNORE
from segment.metrics import (
    Counts,
    bootstrap_ci,
    confusion,
    dice,
    iou,
    pooled,
    precision,
    recall,
)

# 3x3 example, worked by hand (row-major; X marks an ignored pixel):
#   pred   target   outcome
#   1 1 0  1 0 0    tp fp tn
#   1 0 0  1 1 X    tp fn --
#   0 1 1  0 X 1    tn -- tp      (pred at the ignored pixels does not count)
# tp=3, fp=1, fn=1, tn=2 over 7 valid pixels.
PRED = np.array([[1, 1, 0], [1, 0, 0], [0, 1, 1]], dtype=np.uint8)
TARGET = np.array([[1, 0, 0], [1, 1, IGNORE], [0, IGNORE, 1]], dtype=np.uint8)


def test_confusion_hand_computed():
    assert confusion(PRED, TARGET) == Counts(tp=3, fp=1, fn=1, tn=2)


def test_ignored_pixels_are_excluded():
    flipped = PRED.copy()
    flipped[1, 2] = 1
    flipped[2, 1] = 0
    assert confusion(flipped, TARGET) == confusion(PRED, TARGET)


def test_metrics_hand_computed():
    c = Counts(3, 1, 1, 2)
    assert iou(c) == pytest.approx(3 / 5)
    assert dice(c) == pytest.approx(6 / 8)
    assert precision(c) == pytest.approx(3 / 4)
    assert recall(c) == pytest.approx(3 / 4)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        confusion(np.zeros((3, 3), np.uint8), np.zeros((3, 4), np.uint8))


def test_all_ignored_is_zero_counts():
    target = np.full((2, 2), IGNORE, dtype=np.uint8)
    assert confusion(np.ones((2, 2), np.uint8), target) == Counts.zero()


def test_nan_when_undefined():
    empty = Counts.zero()
    assert math.isnan(iou(empty)) and math.isnan(dice(empty))
    assert math.isnan(precision(empty)) and math.isnan(recall(empty))
    assert math.isnan(iou(Counts(0, 0, 0, 10)))
    assert math.isnan(precision(Counts(0, 0, 4, 6)))
    assert recall(Counts(0, 0, 4, 6)) == 0.0


def test_counts_addition_and_zero():
    a, b = Counts(1, 2, 3, 4), Counts(10, 20, 30, 40)
    assert a + b == Counts(11, 22, 33, 44)
    assert a + Counts.zero() == a


def test_pooled():
    assert pooled([Counts(1, 2, 3, 4), Counts(10, 20, 30, 40)]) == Counts(11, 22, 33, 44)
    assert pooled([]) == Counts.zero()
    assert isinstance(pooled([Counts(1, 1, 1, 1)]).tp, int)


def test_pooled_is_pixel_weighted_not_mean_of_ious():
    big, small = Counts(90, 10, 0, 0), Counts(0, 0, 1, 0)
    assert iou(pooled([big, small])) == pytest.approx(90 / 101)
    assert (iou(big) + iou(small)) / 2 != pytest.approx(iou(pooled([big, small])))


def _sample() -> list[Counts]:
    rng = np.random.default_rng(1)
    return [
        Counts(
            int(rng.integers(50, 100)),
            int(rng.integers(0, 20)),
            int(rng.integers(0, 20)),
            int(rng.integers(100, 200)),
        )
        for _ in range(12)
    ]


def test_bootstrap_ci_deterministic_for_seed():
    counts = _sample()
    assert bootstrap_ci(counts, seed=3, n_boot=300) == bootstrap_ci(counts, seed=3, n_boot=300)
    assert bootstrap_ci(counts, seed=3, n_boot=300) != bootstrap_ci(counts, seed=4, n_boot=300)


def test_bootstrap_ci_contains_point_estimate():
    counts = _sample()
    lo, hi = bootstrap_ci(counts, n_boot=500)
    assert lo < hi
    assert lo <= iou(pooled(counts)) <= hi


def test_bootstrap_ci_other_metric():
    counts = _sample()
    lo, hi = bootstrap_ci(counts, metric=dice, n_boot=500)
    assert lo <= dice(pooled(counts)) <= hi


def test_bootstrap_ci_degenerate_sample_has_zero_width():
    counts = [Counts(5, 1, 1, 3)] * 6
    lo, hi = bootstrap_ci(counts, n_boot=200)
    assert lo == hi == pytest.approx(iou(Counts(5, 1, 1, 3)))


def test_bootstrap_ci_empty_and_undefined_are_nan():
    assert all(math.isnan(v) for v in bootstrap_ci([]))
    assert all(math.isnan(v) for v in bootstrap_ci([Counts(0, 0, 0, 9)], n_boot=50))
