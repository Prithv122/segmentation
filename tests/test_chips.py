from __future__ import annotations

import copy
import csv
import re
import threading
import zipfile
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from segment import chips
from segment.bands import IGNORE
from segment.chips import (
    LICENCE,
    MIN_VALID_FRAC,
    MIRROR_REVISION,
    MIRROR_URL,
    WATER_CHIP_FRAC,
    Candidate,
    Scene,
    _stream_member,
    build_all,
    build_scene,
    candidate_grid,
    chip_id,
    cut_scene,
    read_mask,
    read_scene,
    read_scenes,
    scene_from_meta,
    select_chips,
    verify_chips,
    write_scenes,
)
from segment.data import check_splits, content_sha256, load_chip, read_manifest, save_chip

SCENES_CSV = Path(__file__).resolve().parents[1] / "data" / "scenes.csv"


def make_scene(number: int = 71, split: str = "train", tile: str = "T35UNT") -> Scene:
    return Scene(
        scene=chips.scene_name(number),
        number=number,
        part=1,
        split=split,
        s2_tile=tile,
        date="20190911",
        landcover="Cropland rainfed",
        bbox="0.00 0.00 1.00 1.00",
    )


def make_arrays(size: int = 16, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Random 5-band image and a 0/1 mask with no IGNORE pixels."""
    rng = np.random.default_rng(seed)
    image = rng.integers(1, 10000, size=(5, size, size), dtype=np.uint16)
    mask = rng.integers(0, 2, size=(size, size), dtype=np.uint8)
    return image, mask


# ---------------------------------------------------------------- candidate_grid


def test_candidate_grid_hand_computed():
    # 20x20 mask, chip 8: a 2x2 grid; the 4-pixel strip on the bottom and right is dropped.
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[:2, :8] = 1  # block (0,0): 16 water pixels ...
    mask[6:8, :8] = IGNORE  # ... and 16 ignored, so valid 48/64 = 0.75, water 16/48 = 1/3
    mask[:8, 8:16] = IGNORE  # block (0,1): entirely ignored
    mask[8:16, 8:16] = 1  # block (1,1): entirely water
    mask[16:, :] = 1  # partial edge strips must not be seen
    mask[:, 16:] = 1
    cands = candidate_grid(mask, chip=8)
    assert [(c.row, c.col) for c in cands] == [(0, 0), (0, 8), (8, 0), (8, 8)]
    assert cands[0].valid_frac == pytest.approx(0.75)
    assert cands[0].water_frac == pytest.approx(1 / 3)
    assert cands[1].valid_frac == 0.0 and cands[1].water_frac == 0.0
    assert cands[2].valid_frac == 1.0 and cands[2].water_frac == 0.0
    assert cands[3].valid_frac == 1.0 and cands[3].water_frac == 1.0


def test_candidate_grid_smaller_than_one_chip_is_empty():
    assert candidate_grid(np.zeros((7, 20), dtype=np.uint8), chip=8) == []


# ---------------------------------------------------------------- select_chips


def make_candidates(n: int, watery: set[int] | None = None) -> list[Candidate]:
    watery = watery or set()
    return [
        Candidate(row=i * 8, col=0, valid_frac=1.0, water_frac=0.5 if i in watery else 0.0)
        for i in range(n)
    ]


def test_select_excludes_low_valid_chips():
    cands = [
        Candidate(0, 0, MIN_VALID_FRAC - 0.01, 0.0),
        Candidate(8, 0, MIN_VALID_FRAC, 0.0),
        Candidate(16, 0, 1.0, 0.0),
    ]
    got = select_chips(cands, "val", k=10, seed=0)
    assert [c.row for c in got] == [8, 16]


def test_select_exclusion_applies_before_sampling():
    cands = make_candidates(30)
    bad = {3, 4, 5}
    cands = [Candidate(c.row, c.col, 0.1, 0.0) if i in bad else c for i, c in enumerate(cands)]
    for seed in range(10):
        got = select_chips(cands, "test", k=10, seed=seed)
        assert len(got) == 10
        assert all(c.valid_frac >= MIN_VALID_FRAC for c in got)


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_select_returns_all_when_fewer_than_k(split):
    cands = make_candidates(5)
    assert select_chips(cands, split, k=5, seed=1) == cands
    assert select_chips(cands, split, k=50, seed=1) == cands


def test_select_val_test_deterministic_and_seed_dependent():
    cands = make_candidates(100)
    a = select_chips(cands, "val", k=10, seed=3)
    assert a == select_chips(cands, "val", k=10, seed=3)
    assert a != select_chips(cands, "val", k=10, seed=4)
    assert len(a) == 10


def test_select_val_test_ignore_water_share():
    # uniform sampling: with 5 watery chips out of 100 the sample is not steered towards them
    cands = make_candidates(100, watery=set(range(5)))
    n_water = [
        sum(c.water_frac >= WATER_CHIP_FRAC for c in select_chips(cands, "val", 20, seed=s))
        for s in range(20)
    ]
    assert np.mean(n_water) < 3  # expectation is 1; the train rule would give 5


def test_select_output_in_grid_order_without_duplicates():
    cands = make_candidates(100, watery=set(range(0, 100, 3)))
    for split in ("train", "val", "test"):
        got = select_chips(cands, split, k=20, seed=2)
        rows = [c.row for c in got]
        assert rows == sorted(rows)
        assert len(set(rows)) == len(rows) == 20


def test_select_train_scarce_water_takes_all_watery():
    watery = {7, 40, 90}
    cands = make_candidates(100, watery=watery)
    for seed in range(10):
        got = select_chips(cands, "train", k=20, seed=seed)
        assert len(got) == 20
        assert watery <= {c.row // 8 for c in got}


def test_select_train_no_water_still_fills_k():
    got = select_chips(make_candidates(100), "train", k=20, seed=0)
    assert len(got) == 20


def test_select_train_plentiful_water_guarantees_half():
    watery = set(range(40))
    cands = make_candidates(100, watery=watery)
    counts = []
    for seed in range(20):
        got = select_chips(cands, "train", k=20, seed=seed)
        assert len(got) == 20
        counts.append(sum(c.water_frac >= WATER_CHIP_FRAC for c in got))
    assert min(counts) >= 10  # k // 2 guaranteed; the other half is uniform over the rest
    assert np.mean(counts) > 11  # uniform sampling alone would give 8


def test_select_train_water_threshold_is_inclusive():
    cands = [Candidate(i * 8, 0, 1.0, WATER_CHIP_FRAC if i == 0 else 0.0) for i in range(50)]
    for seed in range(10):
        assert 0 in {c.row // 8 for c in select_chips(cands, "train", k=10, seed=seed)}


# ---------------------------------------------------------------- cut_scene


CHIP_ID_RE = re.compile(r"^s\d{3}_r\d{5}_c\d{5}$")


def test_cut_scene_writes_chips_and_records(tmp_path):
    scene = make_scene(71, "train", "T35UNT")
    image, mask = make_arrays(16)
    records = cut_scene(scene, image, mask, tmp_path, chip=8, k=3)
    assert len(records) == 3
    for rec in records:
        assert CHIP_ID_RE.match(rec.chip_id)
        assert rec.chip_id == f"s071_r{rec.row:05d}_c{rec.col:05d}"
        assert rec.image_path_rel == f"chips/s071/{rec.chip_id}.npz"
        assert rec.scene == "s071" and rec.split == "train" and rec.s2_tile == "T35UNT"
        assert rec.licence == LICENCE and rec.source_url == chips.SOURCE_URL
        img, msk = load_chip(tmp_path / rec.image_path_rel)
        assert img.shape == (5, 8, 8) and msk.shape == (8, 8)
        assert np.array_equal(img, image[:, rec.row : rec.row + 8, rec.col : rec.col + 8])
        assert np.array_equal(msk, mask[rec.row : rec.row + 8, rec.col : rec.col + 8])
        assert rec.sha256 == content_sha256(img, msk)
        assert rec.valid_frac == 1.0
        assert rec.water_frac == pytest.approx(float((msk == 1).mean()))
    assert [r.chip_id for r in records] == sorted(r.chip_id for r in records)


def test_cut_scene_default_k_returns_all_when_few_candidates(tmp_path):
    image, mask = make_arrays(16)
    records = cut_scene(make_scene(5, "val"), image, mask, tmp_path, chip=8)
    assert len(records) == 4
    assert {r.split for r in records} == {"val"}


def test_cut_scene_is_deterministic(tmp_path):
    image, mask = make_arrays(32)
    a = cut_scene(make_scene(9), image, mask, tmp_path / "a", chip=8, k=5)
    b = cut_scene(make_scene(9), image, mask, tmp_path / "b", chip=8, k=5)
    assert a == b


def _nodata_scene() -> tuple[np.ndarray, np.ndarray]:
    """16x16 scene; chip (0, 0) has a 4x8 nodata block, 2 mask-IGNORE pixels and 16 valid water."""
    image, _ = make_arrays(16)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[0:2, 0:8] = 1  # water inside the nodata block: becomes IGNORE
    mask[4:6, 0:8] = 1  # 16 water pixels that stay valid
    mask[7, 0:2] = IGNORE  # ignored by the source valid mask
    mask[8:, 8:] = 1
    image[:, 0:4, 0:8] = 0
    image[0, 6, 6] = 0  # one zero band only: still valid
    return image, mask


def test_cut_scene_all_zero_pixels_become_ignore_and_fractions_follow(tmp_path):
    image, mask = _nodata_scene()
    records = cut_scene(make_scene(), image, mask, tmp_path, chip=8, k=4)
    first = next(r for r in records if (r.row, r.col) == (0, 0))
    _, saved = load_chip(tmp_path / first.image_path_rel)
    assert (saved[0:4, 0:8] == IGNORE).all()  # nodata, even where the source mask said water
    assert (saved[7, 0:2] == IGNORE).all()
    assert (saved == IGNORE).sum() == 32 + 2
    assert saved[6, 6] == 0  # a single zero band does not make a pixel nodata
    # 64 - 32 - 2 = 30 valid pixels, 16 of them water
    assert first.valid_frac == pytest.approx(30 / 64)
    assert first.water_frac == pytest.approx(16 / 30)
    for rec in records:
        _, m = load_chip(tmp_path / rec.image_path_rel)
        n_valid = int((m != IGNORE).sum())
        assert rec.valid_frac == pytest.approx(n_valid / m.size)
        assert rec.water_frac == pytest.approx((m == 1).sum() / n_valid)
    other = next(r for r in records if (r.row, r.col) == (8, 8))
    assert other.valid_frac == 1.0 and other.water_frac == 1.0


def test_cut_scene_selects_on_mask_only_and_survives_fully_empty_chip(tmp_path):
    image, mask = make_arrays(16)
    image[:, 8:, 8:] = 0  # nodata everywhere in chip (8, 8), but the mask calls it valid
    records = cut_scene(make_scene(), image, mask, tmp_path, chip=8, k=4)
    assert len(records) == 4  # nodata does not affect eligibility
    empty = next(r for r in records if (r.row, r.col) == (8, 8))
    assert empty.valid_frac == 0.0 and empty.water_frac == 0.0
    _, saved = load_chip(tmp_path / empty.image_path_rel)
    assert (saved == IGNORE).all()


@pytest.mark.parametrize("split", ["train", "val"])
def test_cut_scene_whole_array_and_window_reader_agree(tmp_path, split):
    image, mask = _nodata_scene()
    scene = make_scene(split=split)
    whole = cut_scene(scene, image, mask, tmp_path / "whole", chip=8, k=3)
    calls: list[tuple[int, int, int]] = []
    windows = cut_scene(scene, window_reader(image, calls), mask, tmp_path / "win", chip=8, k=3)
    assert whole == windows
    assert sorted((r, c) for r, c, _ in calls) == sorted((r.row, r.col) for r in whole)
    for rec in whole:
        a_img, a_msk = load_chip(tmp_path / "whole" / rec.image_path_rel)
        b_img, b_msk = load_chip(tmp_path / "win" / rec.image_path_rel)
        assert a_img.dtype == b_img.dtype and np.array_equal(a_img, b_img)
        assert np.array_equal(a_msk, b_msk)
    assert verify_chips(windows, tmp_path / "win") == []


def test_cut_scene_does_not_modify_inputs(tmp_path):
    image, mask = _nodata_scene()
    image0, mask0 = image.copy(), mask.copy()
    cut_scene(make_scene(), image, mask, tmp_path, chip=8, k=4)
    assert np.array_equal(image, image0) and np.array_equal(mask, mask0)


def test_chip_id_format():
    assert chip_id(make_scene(3), Candidate(512, 1024, 1.0, 0.0)) == "s003_r00512_c01024"


# ---------------------------------------------------------------- build_scene / build_all


def window_reader(image: np.ndarray, calls: list | None = None):
    """A ``read(row, col, size)`` over a whole array, as the mirror opener provides."""

    def read(row: int, col: int, size: int) -> np.ndarray:
        if calls is not None:
            calls.append((row, col, size))
        return image[:, row : row + size, col : col + size]

    return read


class Fakes:
    """Injected opener that touches no network and records how often it was entered."""

    def __init__(self, fail_open: bool = False, use_reader: bool = False):
        self.opened: list[str] = []
        self.fail_open = fail_open
        self.use_reader = use_reader
        self.reads: list[tuple[int, int, int]] = []

    @contextmanager
    def opener(self, scene: Scene, tmp_dir: Path):
        self.opened.append(scene.scene)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        (tmp_dir / "msk.tif").write_bytes(b"x")
        if self.fail_open:
            raise RuntimeError("boom")
        image, mask = make_arrays(16)
        yield (window_reader(image, self.reads) if self.use_reader else image), mask


def test_build_scene_writes_records_and_cleans_tmp(tmp_path):
    fakes = Fakes()
    scene = make_scene(12)
    done = build_scene(scene, tmp_path, opener=fakes.opener, chip=8, k=2)
    assert done == tmp_path / "records" / "s012.csv"
    records = read_manifest(done)
    assert len(records) == 2 and all(r.scene == "s012" for r in records)
    assert not (tmp_path / "tmp" / "s012").exists()
    assert all((tmp_path / r.image_path_rel).exists() for r in records)


def test_build_scene_with_window_reader_reads_only_chosen_windows(tmp_path):
    fakes = Fakes(use_reader=True)
    done = build_scene(make_scene(12), tmp_path, opener=fakes.opener, chip=8, k=2)
    records = read_manifest(done)
    assert sorted((r.row, r.col) for r in records) == sorted((r, c) for r, c, _ in fakes.reads)
    assert {size for _, _, size in fakes.reads} == {8}
    assert verify_chips(records, tmp_path) == []


def test_build_scene_removes_tmp_when_opener_raises(tmp_path):
    fakes = Fakes(fail_open=True)
    with pytest.raises(RuntimeError, match="boom"):
        build_scene(make_scene(12), tmp_path, opener=fakes.opener, chip=8, k=2)
    assert fakes.opened == ["s012"]
    assert not (tmp_path / "tmp" / "s012").exists()
    assert not (tmp_path / "records" / "s012.csv").exists()


def test_build_scene_removes_tmp_when_cutting_raises(tmp_path):
    fakes = Fakes()
    with pytest.raises(KeyError):  # unknown split and no explicit k
        build_scene(make_scene(12, split="bogus"), tmp_path, opener=fakes.opener, chip=8)
    assert not (tmp_path / "tmp" / "s012").exists()
    assert not (tmp_path / "records" / "s012.csv").exists()


def test_build_scene_is_resumable(tmp_path):
    fakes = Fakes()
    scene = make_scene(12)
    first = build_scene(scene, tmp_path, opener=fakes.opener, chip=8, k=2)
    before = first.read_bytes()
    second = build_scene(scene, tmp_path, opener=fakes.opener, chip=8, k=2)
    assert second == first and first.read_bytes() == before
    assert fakes.opened == ["s012"]


def test_build_all_writes_sorted_combined_manifest(tmp_path):
    scenes = [
        make_scene(3, "test", "T03CCC"),
        make_scene(1, "train", "T01AAA"),
        make_scene(2, "val", "T02BBB"),
    ]
    fakes = Fakes(use_reader=True)
    paths = build_all(scenes, tmp_path, workers=2, opener=fakes.opener, chip=8, k=2)
    assert len(paths) == 3
    assert sorted(fakes.opened) == ["s001", "s002", "s003"]
    records = read_manifest(tmp_path / "manifest.csv")
    assert len(records) == 6
    ids = [r.chip_id for r in records]
    assert ids == sorted(ids)
    assert check_splits(records) == []
    assert verify_chips(records, tmp_path) == []


def test_build_all_reraises_failures(tmp_path):
    fakes = Fakes(fail_open=True)
    with pytest.raises(RuntimeError, match="boom"):
        build_all([make_scene(1)], tmp_path, workers=1, opener=fakes.opener)
    assert not (tmp_path / "manifest.csv").exists()


def test_mirror_url_is_pinned_to_a_revision():
    assert re.fullmatch(r"[0-9a-f]{40}", MIRROR_REVISION)
    url = MIRROR_URL.format(rel="71/sentinel12_s2_71_img.tif")
    assert url.endswith(f"/resolve/{MIRROR_REVISION}/71/sentinel12_s2_71_img.tif")


# ---------------------------------------------------------------- read_scene


def _write_tif(path: Path, data: np.ndarray) -> None:
    import rasterio

    data = data if data.ndim == 3 else data[None]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=data.dtype,
    ) as dst:
        dst.write(data)


def _write_scene_tifs(tmp_path: Path) -> dict[str, Path]:
    h = w = 4
    img = np.empty((6, h, w), dtype=np.uint16)
    for b in range(6):
        img[b] = 100 * (b + 1) + 1  # band b+1 holds 100*(b+1)+1 everywhere
    img[:, 0, 0] = 0  # all-zero pixel: read_scene keeps it, cut_scene turns it into IGNORE
    img[:5, 0, 2] = 0
    msk = np.zeros((h, w), dtype=np.uint8)
    msk[1, 0] = 1  # valid water
    msk[0, 1] = 1  # water, but marked invalid below
    msk[1, 1] = 2  # not the water code -> land
    valid = np.ones((h, w), dtype=np.uint8)
    valid[0, 1] = 0
    valid[3, 3] = 0  # invalid land
    paths = {"img": tmp_path / "i.tif", "msk": tmp_path / "m.tif", "valid": tmp_path / "v.tif"}
    _write_tif(paths["img"], img)
    _write_tif(paths["msk"], msk)
    _write_tif(paths["valid"], valid)
    return paths


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
def test_read_mask_water_land_and_invalid(tmp_path):
    pytest.importorskip("rasterio")
    paths = _write_scene_tifs(tmp_path)
    mask = read_mask(paths["msk"], paths["valid"])
    assert mask.shape == (4, 4) and mask.dtype == np.uint8
    assert mask[1, 0] == 1
    assert mask[0, 1] == IGNORE and mask[3, 3] == IGNORE  # valid == 0 wins over water
    assert mask[1, 1] == 0 and mask[2, 2] == 0  # code 2 is not water
    assert (mask == IGNORE).sum() == 2


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
def test_read_scene_keeps_bands_one_to_five_and_leaves_nodata_to_cut_scene(tmp_path):
    pytest.importorskip("rasterio")
    paths = _write_scene_tifs(tmp_path)
    image, mask = read_scene(paths)
    assert image.shape == (5, 4, 4) and image.dtype == np.uint16
    for b in range(5):
        assert image[b, 2, 2] == 100 * (b + 1) + 1  # bands 1..5 in source order
    assert 601 not in np.unique(image)  # band 6 dropped
    assert (image[:, 0, 0] == 0).all()  # nodata pixel is kept as read ...
    assert mask[0, 0] == 0 and mask[0, 2] == 0  # ... and is not IGNORE here
    assert np.array_equal(mask, read_mask(paths["msk"], paths["valid"]))


# ---------------------------------------------------------------- scenes


REAL_SRCID = "S2B_MSIL1C_20190911T092029_N0208_R093_T35UNT_20190911T120630"


def test_scene_from_meta():
    meta = {
        "id": "sentinel12_71_meta",
        "bbox": [2855000.123, 7500000.0, 2960000.456, 7600000.789],
        "properties": {
            "split": "test",
            "s2_srcids": [REAL_SRCID, "other"],
            "date_s2": "20190911",
            "landcover": "Cropland rainfed",
        },
    }
    scene = scene_from_meta(meta, part=3)
    assert scene == Scene(
        scene="s071",
        number=71,
        part=3,
        split="test",
        s2_tile="T35UNT",
        date="20190911",
        landcover="Cropland rainfed",
        bbox="2855000.12 7500000.00 2960000.46 7600000.79",
    )


def test_write_read_scenes_round_trip(tmp_path):
    scenes = [
        make_scene(20, "val", "T02BBB"),
        Scene(
            "s001", 1, 2, "train", "T01AAA", "20180601", 'Tree cover, "closed" (>15%)', "1 2 3 4"
        ),
    ]
    path = tmp_path / "scenes.csv"
    write_scenes(path, scenes)
    assert read_scenes(path) == sorted(scenes, key=lambda s: s.number)
    first = path.read_bytes()
    write_scenes(path, read_scenes(path))
    assert path.read_bytes() == first and b"\r" not in first


def test_committed_scenes_csv():
    scenes = read_scenes(SCENES_CSV)
    assert len(scenes) == 65
    assert Counter(s.split for s in scenes) == {"train": 33, "val": 16, "test": 16}
    assert len({s.number for s in scenes}) == 65
    splits_of: dict[str, set[str]] = {}
    for s in scenes:
        splits_of.setdefault(s.s2_tile, set()).add(s.split)
    assert all(len(v) == 1 for v in splits_of.values())
    with SCENES_CSV.open(newline="", encoding="utf-8") as fh:
        assert tuple(next(csv.reader(fh))) == chips.SCENE_COLUMNS


# ---------------------------------------------------------------- _stream_member


class RangeHandler(BaseHTTPRequestHandler):
    root: Path
    #: ``mode`` "once" or "always" drops the connection after ``drop_after`` body bytes of a
    #: data request (a range longer than the 30-byte local header); ``drops`` counts them.
    flaky: dict

    def do_GET(self):
        data = (self.root / self.path.lstrip("/")).read_bytes()
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
        try:
            if match is None:
                self.send_response(200)
                body = data
            else:
                start, end = int(match[1]), min(int(match[2]), len(data) - 1)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
                body = data[start : end + 1]
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            flaky = self.flaky
            drop = (
                len(body) > 100
                and len(body) > flaky["drop_after"]
                and (flaky["mode"] == "always" or (flaky["mode"] == "once" and not flaky["drops"]))
            )
            if drop:
                flaky["drops"] += 1
                self.wfile.write(body[: flaky["drop_after"]])
                self.wfile.flush()
                self.close_connection = True
                return
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):
        pass


@pytest.fixture
def flaky() -> dict:
    return {"mode": None, "drop_after": 0, "drops": 0}


@pytest.fixture
def zip_server(tmp_path, monkeypatch, flaky) -> Iterator[tuple[str, Path]]:
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    handler = type("Handler", (RangeHandler,), {"root": tmp_path, "flaky": flaky})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/parts.zip", tmp_path / "parts.zip"
    finally:
        server.shutdown()
        server.server_close()


PAYLOADS = {
    "a/deflated.bin": (zipfile.ZIP_DEFLATED, b"water" * 40000 + bytes(range(256)) * 50),
    "b/stored.bin": (zipfile.ZIP_STORED, np.random.default_rng(0).bytes(50000)),
    "c/deflated_noisy.bin": (zipfile.ZIP_DEFLATED, np.random.default_rng(1).bytes(30000)),
}


def _build_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as z:
        for name, (method, data) in PAYLOADS.items():
            z.writestr(name, data, compress_type=method)


@pytest.mark.parametrize("name", list(PAYLOADS))
def test_stream_member_round_trip(zip_server, tmp_path, name):
    url, zip_path = zip_server
    _build_zip(zip_path)
    with zipfile.ZipFile(zip_path) as z:
        info = z.getinfo(name)
    assert info.compress_type == PAYLOADS[name][0]
    out = _stream_member(url, info, tmp_path / "out" / "x.bin", chunk=4096)
    assert out.read_bytes() == PAYLOADS[name][1]


@pytest.mark.parametrize("name", ["b/stored.bin", "c/deflated_noisy.bin"])
def test_stream_member_resumes_after_dropped_connection(zip_server, tmp_path, flaky, name):
    url, zip_path = zip_server
    _build_zip(zip_path)
    with zipfile.ZipFile(zip_path) as z:
        info = z.getinfo(name)
    assert info.compress_size > 5000
    flaky.update(mode="once", drop_after=3000)
    out = _stream_member(url, info, tmp_path / "out" / "x.bin", chunk=1000, retries=2, timeout=5)
    assert flaky["drops"] == 1
    assert out.read_bytes() == PAYLOADS[name][1]


def test_stream_member_gives_up_when_connection_drops_without_progress(zip_server, tmp_path, flaky):
    url, zip_path = zip_server
    _build_zip(zip_path)
    with zipfile.ZipFile(zip_path) as z:
        info = z.getinfo("b/stored.bin")
    flaky.update(mode="always", drop_after=0)
    with pytest.raises(OSError, match="gave up"):
        _stream_member(url, info, tmp_path / "out" / "x.bin", chunk=500, retries=2, timeout=2)
    assert flaky["drops"] == 3  # the first try plus two retries, none of them making progress


def test_stream_member_keeps_resuming_while_each_try_makes_progress(zip_server, tmp_path, flaky):
    url, zip_path = zip_server
    _build_zip(zip_path)
    with zipfile.ZipFile(zip_path) as z:
        info = z.getinfo("b/stored.bin")
        expected = z.read("b/stored.bin")
    flaky.update(mode="always", drop_after=1000)
    out = _stream_member(url, info, tmp_path / "out" / "x.bin", chunk=500, retries=2, timeout=2)
    assert out.read_bytes() == expected
    assert flaky["drops"] > 2  # more drops than retries, yet it finished


def test_stream_member_detects_crc_mismatch(zip_server, tmp_path):
    url, zip_path = zip_server
    _build_zip(zip_path)
    with zipfile.ZipFile(zip_path) as z:
        info = copy.copy(z.getinfo("a/deflated.bin"))
    info.CRC ^= 1
    with pytest.raises(OSError, match="CRC"):
        _stream_member(url, info, tmp_path / "out" / "x.bin")


def test_stream_member_rejects_unsupported_compression(zip_server, tmp_path):
    url, zip_path = zip_server
    _build_zip(zip_path)
    with zipfile.ZipFile(zip_path) as z:
        info = copy.copy(z.getinfo("b/stored.bin"))
    info.compress_type = zipfile.ZIP_BZIP2
    with pytest.raises(OSError, match="unsupported compression"):
        _stream_member(url, info, tmp_path / "out" / "x.bin")


# ---------------------------------------------------------------- verify_chips


def test_verify_chips_clean_missing_and_altered(tmp_path):
    image, mask = make_arrays(16)
    records = cut_scene(make_scene(4), image, mask, tmp_path, chip=8, k=3)
    assert verify_chips(records, tmp_path) == []

    (tmp_path / records[0].image_path_rel).unlink()
    altered = tmp_path / records[1].image_path_rel
    img, msk = load_chip(altered)
    img = img.copy()
    img[0, 0, 0] ^= 1
    save_chip(altered, img, msk)

    problems = verify_chips(records, tmp_path)
    assert problems == [
        f"{records[0].chip_id}: missing",
        f"{records[1].chip_id}: sha256 mismatch",
    ]
