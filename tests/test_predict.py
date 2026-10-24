import csv

import numpy as np
import pytest

import segment.data as data
from segment.predict import PER_CHIP_COLUMNS, _fmt, score_split


def _true_mask_fn(manifest):
    root = manifest.parent
    by_image: dict[bytes, np.ndarray] = {}
    for rec in data.read_manifest(manifest):
        image, mask = data.load_chip(root / rec.image_path_rel)
        by_image[image.tobytes()] = (mask == 1).astype(np.uint8)
    return lambda image: by_image[image.tobytes()]


def test_perfect_predictor_scores_iou_one(fixture_manifest, tmp_path):
    out = tmp_path / "per_chip.csv"
    rows = score_split(
        "oracle",
        _true_mask_fn(fixture_manifest),
        fixture_manifest,
        fixture_manifest.parent,
        "val",
        out,
    )
    assert [r["chip_id"] for r in rows] == ["C_0", "C_1"]
    assert all(r["iou"] == 1.0 and r["fp"] == 0 and r["fn"] == 0 for r in rows)
    raw = out.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    parsed = list(csv.reader(raw.decode().splitlines()))
    assert tuple(parsed[0]) == PER_CHIP_COLUMNS
    first = dict(zip(parsed[0], parsed[1], strict=True))
    assert first["method"] == "oracle" and first["split"] == "val"
    assert first["iou"] == "1.000000" and first["dice"] == "1.000000"


def test_all_zero_predictor_has_no_true_positives(fixture_manifest, tmp_path):
    rows = score_split(
        "zeros",
        lambda img: np.zeros(img.shape[1:], np.uint8),
        fixture_manifest,
        fixture_manifest.parent,
        "train",
        tmp_path / "zeros.csv",
    )
    assert [r["chip_id"] for r in rows] == ["A_0", "A_1", "B_0", "B_1"]
    assert all(r["tp"] == 0 and r["fp"] == 0 and r["fn"] > 0 and r["iou"] == 0.0 for r in rows)
    a1 = next(r for r in rows if r["chip_id"] == "A_1")
    assert a1["tp"] + a1["fp"] + a1["fn"] + a1["tn"] == 32 * 32 - 16


def test_masks_out_keyed_by_chip_id(fixture_manifest, tmp_path):
    masks = tmp_path / "sub" / "masks.npz"
    rows = score_split(
        "oracle",
        _true_mask_fn(fixture_manifest),
        fixture_manifest,
        fixture_manifest.parent,
        "test",
        tmp_path / "t.csv",
        masks,
    )
    assert len(rows) == 2
    with np.load(masks) as z:
        assert sorted(z.files) == ["D_0", "D_1"]
        assert z["D_0"].shape == (32, 32) and z["D_0"].dtype == np.uint8


def test_nan_is_written_as_nan():
    assert _fmt(float("nan")) == "nan"
    assert _fmt(0.5) == "0.500000"
    assert _fmt(3) == "3"


def test_unknown_split_and_shape_mismatch(fixture_manifest, tmp_path):
    root = fixture_manifest.parent
    with pytest.raises(ValueError, match="no chips"):
        score_split("m", lambda i: i, fixture_manifest, root, "nope", tmp_path / "a.csv")
    with pytest.raises(ValueError, match="prediction"):
        score_split(
            "m",
            lambda i: np.zeros((3, 3), np.uint8),
            fixture_manifest,
            root,
            "val",
            tmp_path / "a.csv",
        )
