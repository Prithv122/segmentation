import numpy as np
import pytest
import torch

from segment.bands import IGNORE
from segment.model import build_unet, crop_to, pad_to_multiple


def test_unet_output_shape():
    model = build_unet("resnet18", None, 4).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 4, 64, 96))
    assert out.shape == (2, 1, 64, 96)


@pytest.mark.parametrize("shape", [(4, 32, 32), (4, 33, 50), (4, 1, 31)])
def test_pad_crop_round_trip_image(shape):
    rng = np.random.default_rng(0)
    x = rng.random(shape, dtype=np.float32)
    padded = pad_to_multiple(x, mode="reflect")
    assert padded.shape[-2] % 32 == 0 and padded.shape[-1] % 32 == 0
    assert padded.shape[-2] - shape[-2] < 32 and padded.shape[-1] - shape[-1] < 32
    assert np.array_equal(crop_to(padded, *shape[-2:]), x)


def test_pad_mask_uses_ignore():
    mask = np.ones((40, 40), dtype=np.uint8)
    padded = pad_to_multiple(mask, mode="constant", value=IGNORE)
    assert padded.shape == (64, 64)
    assert (padded[:40, :40] == 1).all()
    assert (padded[40:] == IGNORE).all() and (padded[:, 40:] == IGNORE).all()
    assert np.array_equal(crop_to(padded, 40, 40), mask)
