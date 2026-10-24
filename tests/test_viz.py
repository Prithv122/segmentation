import numpy as np
from PIL import Image

from segment.bands import IGNORE
from segment.data import load_chip, read_manifest
from segment.viz import overlay, rgb_preview, save_panel


def flat_image(value=2000, size=4):
    return np.full((5, size, size), value, dtype=np.uint16)


def test_rgb_preview_shape_dtype_and_scaling():
    image = flat_image(1000)
    image[2] = 1500  # red
    image[1] = 30000  # green saturates
    prev = rgb_preview(image)
    assert prev.shape == (4, 4, 3) and prev.dtype == np.uint8
    assert prev[0, 0, 0] == round(1500 / 10000 * 3.5 * 255)
    assert prev[0, 0, 1] == 255
    assert prev[0, 0, 2] == round(1000 / 10000 * 3.5 * 255)


def classified_overlay():
    image = flat_image(2000)
    target = np.zeros((4, 4), dtype=np.uint8)
    pred = np.zeros((4, 4), dtype=np.uint8)
    target[0, 0], pred[0, 0] = 1, 1  # TP
    target[0, 1], pred[0, 1] = 0, 1  # FP
    target[0, 2], pred[0, 2] = 1, 0  # FN
    target[0, 3], pred[0, 3] = IGNORE, 1  # ignored, even though predicted water
    return image, target, pred, overlay(image, target, pred)


def test_overlay_colours():
    image, _, _, out = classified_overlay()
    assert out.shape == (4, 4, 3) and out.dtype == np.uint8
    tp, fp, fn = out[0, 0].astype(int), out[0, 1].astype(int), out[0, 2].astype(int)
    assert tp[0] < tp[1] and tp[0] < tp[2]  # cyan: red is the weakest channel
    assert fp[0] > fp[1] > fp[2]  # orange
    assert fn[1] < fn[0] and fn[1] < fn[2]  # magenta: green is the weakest channel
    # true negatives are untouched
    assert np.array_equal(out[1:], rgb_preview(image)[1:])


def test_overlay_ignore_is_greyed():
    image, _, _, out = classified_overlay()
    prev = rgb_preview(image)
    assert not np.array_equal(out[0, 3], prev[0, 3])
    assert out[0, 3, 0] == out[0, 3, 1] == out[0, 3, 2]


def fixture_chip(fixture_manifest):
    rec = next(r for r in read_manifest(fixture_manifest) if r.chip_id == "D_1")
    return load_chip(fixture_manifest.parent / rec.image_path_rel)


def test_save_panel_width_grows_with_methods(fixture_manifest, tmp_path):
    image, mask = fixture_chip(fixture_manifest)
    perfect = (mask == 1).astype(np.uint8)
    empty = np.zeros_like(perfect)
    h, w = mask.shape

    one = tmp_path / "out" / "one.png"
    save_panel(one, image, mask, {"unet": perfect})
    two = tmp_path / "two.png"
    save_panel(two, image, mask, {"unet": perfect, "ndwi_otsu": empty})

    with Image.open(one) as a, Image.open(two) as b:
        assert a.format == "PNG" and b.format == "PNG"
        assert a.size[0] == 3 * w + 2 * 2
        assert b.size[0] == 4 * w + 3 * 2
        assert b.size[0] > a.size[0]
        assert a.size[1] == b.size[1] > h
        title_strip = np.asarray(a.convert("RGB"))[: a.size[1] - h]
        assert (title_strip < 255).any()  # titles are drawn
