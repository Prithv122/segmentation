from __future__ import annotations

import numpy as np
import pytest

from segment.bands import BAND_INDEX, BANDS
from segment.baseline import (
    Constant,
    IndexThreshold,
    fit_constant,
    fit_index_threshold,
    otsu_threshold,
)
from segment.data import load_chip, read_manifest
from segment.indices import ndwi
from segment.metrics import confusion, iou, pooled


def _load(manifest, split):
    root = manifest.parent
    return [load_chip(root / r.image_path_rel) for r in read_manifest(manifest) if r.split == split]


def test_otsu_bimodal_between_modes():
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.normal(-0.6, 0.05, 2000), rng.normal(0.5, 0.1, 1000)])
    assert -0.3 < otsu_threshold(values) < 0.2


def test_otsu_hand_checkable_example():
    # Values 0 and 10, 4 bins over [0, 10] (edges 0, 2.5, 5, 7.5, 10). Cuts after bins 0, 1
    # and 2 separate the two values equally well; the middle of that plateau is the cut after
    # bin 1, whose upper edge is 5.0.
    values = np.array([0.0] * 5 + [10.0] * 5)
    assert otsu_threshold(values, bins=4) == pytest.approx(5.0)
    assert otsu_threshold(values) == pytest.approx(5.0, abs=10 / 256)


def test_otsu_threshold_is_upper_bin_edge():
    values = np.array([0.0, 1.0, 9.0, 10.0])
    edges = np.linspace(0, 10, 11)
    t = otsu_threshold(values, bins=10)
    assert np.isclose(edges, t).any()
    assert ((values > t) == [False, False, True, True]).all()


def test_otsu_constant_and_edge_inputs():
    assert otsu_threshold(np.full(50, 0.25)) == 0.25
    assert otsu_threshold(np.array([3.0])) == 3.0
    assert otsu_threshold(np.array([1.0, np.nan, 1.0])) == 1.0
    with pytest.raises(ValueError):
        otsu_threshold(np.array([]))


def test_index_threshold_predict():
    image = np.zeros((len(BANDS), 2, 2), dtype=np.uint16)
    image[BAND_INDEX["green"]] = [[800, 700], [800, 700]]
    image[BAND_INDEX["nir"]] = [[200, 3000], [200, 3000]]
    pred = IndexThreshold("ndwi", 0.0).predict(image)
    assert pred.dtype == np.uint8
    assert pred.tolist() == [[1, 0], [1, 0]]


def test_predict_is_strictly_greater():
    image = np.zeros((len(BANDS), 1, 2), dtype=np.uint16)
    image[BAND_INDEX["green"]] = 3
    image[BAND_INDEX["nir"]] = 1
    t = float(ndwi(image)[0, 0])
    assert IndexThreshold("ndwi", t).predict(image).sum() == 0
    assert IndexThreshold("ndwi", t - 1e-3).predict(image).sum() == 2


def test_unknown_index_raises(fixture_manifest):
    chips = _load(fixture_manifest, "train")
    with pytest.raises(ValueError, match="unknown index"):
        fit_index_threshold(chips, index="bogus")
    with pytest.raises(ValueError, match="unknown index"):
        IndexThreshold("bogus", 0.0).predict(chips[0][0])


def test_fit_on_fixture_and_perfect_test_iou(fixture_manifest):
    model = fit_index_threshold(_load(fixture_manifest, "train"))
    assert model.index == "ndwi"
    assert -0.62 < model.threshold < 0.6
    counts = [confusion(model.predict(img), m) for img, m in _load(fixture_manifest, "test")]
    assert iou(pooled(counts)) == 1.0


def test_fit_mndwi_on_fixture(fixture_manifest):
    model = fit_index_threshold(_load(fixture_manifest, "train"), index="mndwi")
    counts = [confusion(model.predict(img), m) for img, m in _load(fixture_manifest, "test")]
    assert iou(pooled(counts)) == 1.0


def test_fit_excludes_ignored_pixels(fixture_manifest):
    img, mask = _load(fixture_manifest, "train")[1]
    assert (mask == 255).any()
    poisoned = img.copy()
    poisoned[:, :4, :4] = 0
    poisoned[BAND_INDEX["green"], :4, :4] = 10000
    assert fit_index_threshold([(poisoned, mask)]) == fit_index_threshold([(img, mask)])


def test_fit_deterministic_for_seed_and_subsampling(fixture_manifest):
    chips = _load(fixture_manifest, "train")
    a = fit_index_threshold(chips, max_pixels_per_chip=100, seed=7)
    b = fit_index_threshold(chips, max_pixels_per_chip=100, seed=7)
    assert a == b
    assert -0.62 < a.threshold < 0.6


def test_fit_no_valid_pixels_raises():
    image = np.zeros((len(BANDS), 2, 2), dtype=np.uint16)
    mask = np.full((2, 2), 255, dtype=np.uint8)
    with pytest.raises(ValueError):
        fit_index_threshold([(image, mask)])
    with pytest.raises(ValueError):
        fit_index_threshold([])


def test_fit_constant_on_fixture_is_land(fixture_manifest):
    masks = [m for _, m in _load(fixture_manifest, "train")]
    assert fit_constant(masks) == Constant(0)


def test_fit_constant_majority_and_ties():
    water = np.ones((2, 2), dtype=np.uint8)
    land = np.zeros((2, 2), dtype=np.uint8)
    ignored = np.full((2, 2), 255, dtype=np.uint8)
    assert fit_constant([water, water, land]) == Constant(1)
    assert fit_constant([water, land]) == Constant(0)
    assert fit_constant([water, ignored, ignored]) == Constant(1)
    assert fit_constant([]) == Constant(0)


def test_constant_predict_shape_and_dtype():
    image = np.zeros((len(BANDS), 3, 4), dtype=np.uint16)
    for value in (0, 1):
        pred = Constant(value).predict(image)
        assert pred.shape == (3, 4) and pred.dtype == np.uint8
        assert (pred == value).all()
