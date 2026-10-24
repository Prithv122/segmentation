from __future__ import annotations

import numpy as np
import pytest

from segment.bands import BAND_INDEX, BANDS
from segment.indices import mndwi, ndwi


def _image(**bands: int) -> np.ndarray:
    """1x1 chip with the given band values (others 0)."""
    image = np.zeros((len(BANDS), 1, 1), dtype=np.uint16)
    for name, value in bands.items():
        image[BAND_INDEX[name]] = value
    return image


def test_ndwi_fixture_values():
    water = ndwi(_image(green=800, nir=200))
    land = ndwi(_image(green=700, nir=3000))
    assert water.dtype == np.float32 and water.shape == (1, 1)
    assert water[0, 0] == pytest.approx(0.6, abs=1e-6)
    assert land[0, 0] == pytest.approx(-2300 / 3700, abs=1e-6)
    assert land[0, 0] == pytest.approx(-0.62162, abs=1e-5)


def test_mndwi_uses_swir():
    out = mndwi(_image(green=800, swir1=100, nir=5000))
    assert out[0, 0] == pytest.approx(700 / 900, abs=1e-6)


def test_zero_sum_pixels_give_zero():
    image = np.zeros((len(BANDS), 2, 2), dtype=np.uint16)
    image[BAND_INDEX["green"], 0, 0] = 800
    image[BAND_INDEX["nir"], 0, 0] = 200
    for fn in (ndwi, mndwi):
        out = fn(image)
        assert not np.isnan(out).any()
        assert out[1, 1] == 0.0 and out[0, 1] == 0.0


def test_no_uint16_overflow():
    out = ndwi(_image(green=60000, nir=60000))
    assert out[0, 0] == 0.0
    out = ndwi(_image(green=60000, nir=5000))
    assert out[0, 0] == pytest.approx(55000 / 65000, abs=1e-6)


def test_no_uint16_underflow():
    out = ndwi(_image(green=100, nir=300))
    assert out[0, 0] == pytest.approx(-0.5, abs=1e-6)


def test_output_range_on_random_chip():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 10000, size=(len(BANDS), 8, 8), dtype=np.uint16)
    for fn in (ndwi, mndwi):
        out = fn(image)
        assert out.shape == (8, 8) and out.dtype == np.float32
        assert out.min() >= -1.0 and out.max() <= 1.0
