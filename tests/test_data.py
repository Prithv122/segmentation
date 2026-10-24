from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from segment.bands import BAND_INDEX, BANDS, IGNORE
from segment.data import (
    MANIFEST_COLUMNS,
    check_splits,
    load_chip,
    read_manifest,
    save_chip,
    to_model_input,
    write_manifest,
)


def _arrays(h: int = 4, w: int = 5):
    return np.zeros((len(BANDS), h, w), dtype=np.uint16), np.zeros((h, w), dtype=np.uint8)


def test_manifest_round_trip_is_byte_identical(fixture_manifest, tmp_path):
    original = fixture_manifest.read_bytes()
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    write_manifest(first, read_manifest(fixture_manifest))
    write_manifest(second, read_manifest(first))
    assert first.read_bytes() == second.read_bytes() == original
    assert b"\r" not in original


def test_manifest_sorted_by_chip_id(fixture_manifest, tmp_path):
    records = read_manifest(fixture_manifest)
    out = tmp_path / "m.csv"
    write_manifest(out, list(reversed(records)))
    assert out.read_bytes() == fixture_manifest.read_bytes()
    assert [r.chip_id for r in read_manifest(out)] == sorted(r.chip_id for r in records)


def test_read_manifest_types(fixture_manifest):
    records = read_manifest(fixture_manifest)
    assert len(records) == 8
    assert isinstance(records[0].row, int) and isinstance(records[0].valid_frac, float)


def test_read_manifest_rejects_wrong_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("chip_id,scene\nx,y\n", encoding="utf-8")
    with pytest.raises(ValueError, match="columns"):
        read_manifest(bad)
    reordered = tmp_path / "reordered.csv"
    reordered.write_text(",".join(reversed(MANIFEST_COLUMNS)) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="columns"):
        read_manifest(reordered)


def test_check_splits_clean_on_fixture(fixture_manifest):
    assert check_splits(read_manifest(fixture_manifest)) == []


@pytest.fixture
def records(fixture_manifest):
    return read_manifest(fixture_manifest)


def _with(records, chip_id, **changes):
    return [replace(r, **changes) if r.chip_id == chip_id else r for r in records]


def test_check_splits_unknown_split(records):
    problems = check_splits(_with(records, "A_0", split="holdout"))
    assert any("unknown split" in p and "A_0" in p for p in problems)


def test_check_splits_duplicate_id(records):
    problems = check_splits([*records, records[0]])
    assert any("duplicate chip_id" in p for p in problems)


def test_check_splits_scene_in_two_splits(records):
    problems = check_splits(_with(records, "A_0", split="test"))
    assert any(p.startswith("scene A") for p in problems)


def test_check_splits_tile_in_two_splits(records):
    problems = check_splits(_with(records, "C_0", s2_tile="T04DDD"))
    assert any(p.startswith("s2_tile T04DDD") for p in problems)
    assert not any(p.startswith("scene") for p in problems)


def test_check_splits_empty_split(records):
    problems = check_splits([r for r in records if r.split != "val"])
    assert problems == ["split 'val' is empty"]


def test_save_chip_rejects_bad_dtypes(tmp_path):
    image, mask = _arrays()
    with pytest.raises(ValueError, match="uint16"):
        save_chip(tmp_path / "c.npz", image.astype(np.float32), mask)
    with pytest.raises(ValueError, match="uint16"):
        save_chip(tmp_path / "c.npz", image, mask.astype(np.int64))
    assert not (tmp_path / "c.npz").exists()


def test_save_chip_rejects_bad_mask_values(tmp_path):
    image, mask = _arrays()
    mask[0, 0] = 2
    with pytest.raises(ValueError, match="mask values"):
        save_chip(tmp_path / "c.npz", image, mask)


def test_save_chip_sha256_matches_file_and_round_trips(tmp_path):
    image, mask = _arrays()
    image[BAND_INDEX["nir"], 1, 2] = 4321
    mask[0, 0], mask[1, 1] = 1, IGNORE
    path = tmp_path / "sub" / "c.npz"
    sha = save_chip(path, image, mask)
    assert sha == hashlib.sha256(path.read_bytes()).hexdigest()
    got_image, got_mask = load_chip(path)
    assert np.array_equal(got_image, image) and np.array_equal(got_mask, mask)


def test_save_chip_is_deterministic(tmp_path):
    image, mask = _arrays()
    assert save_chip(tmp_path / "a.npz", image, mask) == save_chip(tmp_path / "b.npz", image, mask)


def _write_raw(path, image, mask):
    np.savez_compressed(path, image=image, mask=mask)
    return path


def test_load_chip_validates(tmp_path):
    image, mask = _arrays()
    with pytest.raises(ValueError, match="image"):
        load_chip(_write_raw(tmp_path / "a.npz", image[:4], mask))
    with pytest.raises(ValueError, match="image"):
        load_chip(_write_raw(tmp_path / "b.npz", image.astype(np.float32), mask))
    with pytest.raises(ValueError, match="image"):
        load_chip(_write_raw(tmp_path / "c.npz", image[0], mask))
    with pytest.raises(ValueError, match="mask"):
        load_chip(_write_raw(tmp_path / "d.npz", image, mask[:, :-1]))
    with pytest.raises(ValueError, match="mask"):
        load_chip(_write_raw(tmp_path / "e.npz", image, mask.astype(np.int32)))


def test_load_fixture_chip(fixture_manifest):
    rec = read_manifest(fixture_manifest)[0]
    image, mask = load_chip(fixture_manifest.parent / rec.image_path_rel)
    assert image.shape == (5, 32, 32) and mask.shape == (32, 32)


def test_to_model_input_selects_scales_and_clips():
    image = np.zeros((len(BANDS), 2, 2), dtype=np.uint16)
    for name, value in zip(BANDS, (1000, 2000, 3000, 4000, 9000), strict=True):
        image[BAND_INDEX[name]] = value
    image[BAND_INDEX["nir"], 0, 0] = 20000
    x = to_model_input(image)
    assert x.dtype == np.float32 and x.shape == (4, 2, 2)
    assert x[:, 1, 1] == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert x[3, 0, 0] == 1.0
    assert not np.isclose(x, 0.9).any()
    assert x.min() >= 0.0 and x.max() <= 1.0
