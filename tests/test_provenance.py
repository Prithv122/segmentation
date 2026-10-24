from __future__ import annotations

import csv
import json
import threading
import zlib
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import numpy as np
import pytest

from segment import chips, provenance
from segment.chips import Scene, member_names, write_scenes
from segment.cli import main
from segment.provenance import (
    REPORT_COLUMNS,
    check,
    crc32_url,
    mirror_sizes,
    summary_line,
    write_report,
    zenodo_members,
)


def make_scene(number: int, split: str, part: int = 1) -> Scene:
    return Scene(
        scene=chips.scene_name(number),
        number=number,
        part=part,
        split=split,
        s2_tile=f"T{number:02d}AAA",
        date="20190911",
        landcover="Cropland rainfed",
        bbox="0.00 0.00 1.00 1.00",
    )


SCENES = [make_scene(2, "val", part=2), make_scene(1, "train", part=1)]


class World:
    """Fake Zenodo + mirror: every member has known bytes, so sizes and CRCs are computable."""

    def __init__(self, scenes: list[Scene]):
        rng = np.random.default_rng(0)
        self.content: dict[str, bytes] = {}
        for scene in scenes:
            for name in member_names(scene).values():
                self.content[name] = rng.bytes(int(rng.integers(100, 1000)))
        self.zen = {n: (len(b), zlib.crc32(b)) for n, b in self.content.items()}
        self.sizes = {n: len(b) for n, b in self.content.items()}
        self.by_url = {chips.MIRROR_URL.format(rel=n): b for n, b in self.content.items()}
        self.fetched: list[str] = []
        self.crc_xor = 0
        self.crc_xor_urls: set[str] = set()

    def fetch_crc(self, url: str) -> tuple[int, int]:
        self.fetched.append(url)
        data = self.by_url[url]
        flip = self.crc_xor if url in self.crc_xor_urls else 0
        return zlib.crc32(data) ^ flip, len(data)


@pytest.fixture
def world() -> World:
    return World(SCENES)


def run_check(world: World, full_image: set[str]) -> list[dict]:
    return check(SCENES, full_image, fetch_crc=world.fetch_crc, zen=world.zen, sizes=world.sizes)


def url_of(scene_number: int, layer: str) -> str:
    return chips.MIRROR_URL.format(rel=member_names(make_scene(scene_number, "train"))[layer])


# ---------------------------------------------------------------- check


def test_check_all_match(world):
    rows = run_check(world, {"s001"})
    assert [(r["scene"], r["layer"]) for r in rows] == [
        ("s001", "img"),
        ("s001", "msk"),
        ("s001", "valid"),
        ("s002", "img"),
        ("s002", "msk"),
        ("s002", "valid"),
    ]
    assert all(r["size_ok"] is True for r in rows)
    assert {(r["scene"], r["layer"]) for r in rows if r["crc_ok"] is True} == {
        ("s001", "img"),
        ("s001", "msk"),
        ("s001", "valid"),
        ("s002", "msk"),
        ("s002", "valid"),
    }
    assert summary_line(rows) == "6/6 sizes match; 4/4 mask CRCs match; 1/1 image CRCs match"
    assert provenance.mismatches(rows) == []


def test_check_row_fields(world):
    rows = {(r["scene"], r["layer"]): r for r in run_check(world, {"s001"})}
    row = rows[("s001", "msk")]
    name = member_names(SCENES[1])["msk"]
    assert row["split"] == "train"
    assert row["zenodo_size"] == row["mirror_size"] == len(world.content[name])
    assert row["zenodo_crc"] == row["mirror_crc"] == f"{zlib.crc32(world.content[name]):08x}"
    assert len(row["zenodo_crc"]) == 8
    assert set(row) == set(REPORT_COLUMNS)


def test_check_images_only_crc_checked_when_requested(world):
    rows = {(r["scene"], r["layer"]): r for r in run_check(world, set())}
    assert rows[("s001", "img")]["mirror_crc"] == "" and rows[("s001", "img")]["crc_ok"] == ""
    assert all("_img." not in u for u in world.fetched)
    assert len(world.fetched) == 4  # msk and valid of both scenes
    assert summary_line(list(rows.values())) == (
        "6/6 sizes match; 4/4 mask CRCs match; 0/0 image CRCs match"
    )


def test_check_fetches_each_wanted_url_once(world):
    run_check(world, {"s001", "s002"})
    assert sorted(world.fetched) == sorted(world.by_url)


def test_check_flags_size_mismatch(world):
    name = member_names(SCENES[1])["msk"]
    world.sizes[name] += 1
    rows = {(r["scene"], r["layer"]): r for r in run_check(world, set())}
    bad = rows[("s001", "msk")]
    assert bad["size_ok"] is False and bad["mirror_size"] == bad["zenodo_size"] + 1
    assert bad["crc_ok"] is True  # the content itself still matches
    assert provenance.mismatches(list(rows.values())) == [bad]
    assert summary_line(list(rows.values())).startswith("5/6 sizes match")


@pytest.mark.parametrize(
    ("layer", "expected"),
    [
        ("msk", "3/4 mask CRCs match; 1/1 image CRCs match"),
        ("valid", "3/4 mask CRCs match; 1/1 image CRCs match"),
        ("img", "4/4 mask CRCs match; 0/1 image CRCs match"),
    ],
)
def test_check_flags_crc_mismatch(world, layer, expected):
    world.crc_xor = 1
    world.crc_xor_urls = {url_of(1, layer)}
    rows = {(r["scene"], r["layer"]): r for r in run_check(world, {"s001"})}
    bad = rows[("s001", layer)]
    assert bad["crc_ok"] is False and bad["mirror_crc"] != bad["zenodo_crc"]
    assert bad["size_ok"] is True
    assert provenance.mismatches(list(rows.values())) == [bad]
    assert expected in summary_line(list(rows.values()))


def test_check_flags_byte_count_mismatch(world):
    def short(url: str) -> tuple[int, int]:
        data = world.by_url[url]
        return zlib.crc32(data), len(data) - 1

    rows = check(SCENES, set(), fetch_crc=short, zen=world.zen, sizes=world.sizes)
    assert all(r["crc_ok"] is False for r in rows if r["layer"] != "img")


def test_check_missing_mirror_file_is_flagged_without_fetching(world):
    name = member_names(SCENES[0])["valid"]  # scene s002
    del world.sizes[name]
    rows = {(r["scene"], r["layer"]): r for r in run_check(world, set())}
    bad = rows[("s002", "valid")]
    assert bad["mirror_size"] == "" and bad["size_ok"] is False
    assert bad["mirror_crc"] == "" and bad["crc_ok"] is False
    assert chips.MIRROR_URL.format(rel=name) not in world.fetched
    assert provenance.mismatches(list(rows.values())) == [bad]


def test_check_rejects_unknown_full_image_scene(world):
    with pytest.raises(ValueError, match="s999"):
        run_check(world, {"s999"})


# ---------------------------------------------------------------- report


def test_write_report_format_and_determinism(world, tmp_path):
    rows = run_check(world, {"s001"})
    first = tmp_path / "out" / "provenance.csv"
    line = write_report(rows, first)
    assert line == summary_line(rows)
    second = tmp_path / "again.csv"
    write_report(list(reversed(rows)), second)
    assert first.read_bytes() == second.read_bytes()
    raw = first.read_bytes()
    assert b"\r" not in raw
    with first.open(newline="", encoding="utf-8") as fh:
        parsed = list(csv.DictReader(fh))
    assert tuple(parsed[0]) == REPORT_COLUMNS
    assert [(r["scene"], r["layer"]) for r in parsed] == sorted(
        (r["scene"], r["layer"]) for r in rows
    )
    img2 = next(r for r in parsed if (r["scene"], r["layer"]) == ("s002", "img"))
    assert img2["mirror_crc"] == "" and img2["crc_ok"] == "" and img2["size_ok"] == "True"


# ---------------------------------------------------------------- network functions, served locally


class Routes(BaseHTTPRequestHandler):
    routes: dict[str, tuple[bytes, dict[str, str]]]

    def do_GET(self):
        body, headers = self.routes.get(self.path, (b"", {}))
        try:
            self.send_response(200 if self.path in self.routes else 404)
            self.send_header("Content-Length", str(len(body)))
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):
        pass


@pytest.fixture
def server(monkeypatch) -> Iterator[SimpleNamespace]:
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    routes: dict[str, tuple[bytes, dict[str, str]]] = {}
    handler = type("Handler", (Routes,), {"routes": routes})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(
        target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield SimpleNamespace(base=f"http://127.0.0.1:{httpd.server_address[1]}", routes=routes)
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_crc32_url_streams_in_chunks(server):
    data = np.random.default_rng(1).bytes(2_500_000)  # > 2 MiB: several 1 MiB reads
    server.routes["/big.tif"] = (data, {})
    assert crc32_url(server.base + "/big.tif") == (zlib.crc32(data), len(data))
    assert crc32_url(server.base + "/big.tif", chunk=1000) == (zlib.crc32(data), len(data))


def test_crc32_url_empty_body(server):
    server.routes["/empty"] = (b"", {})
    assert crc32_url(server.base + "/empty") == (0, 0)


def test_mirror_sizes_follows_link_header_and_skips_directories(server):
    page1 = [
        {"type": "directory", "path": "1"},
        {"type": "file", "path": "1/sentinel12_s2_1_img.tif", "size": 123},
    ]
    page2 = [{"type": "file", "path": "1/sentinel12_s2_1_msk.tif", "size": 45}]
    server.routes["/tree"] = (
        json.dumps(page1).encode(),
        {"Link": f'<{server.base}/tree2>; rel="next"'},
    )
    server.routes["/tree2"] = (
        json.dumps(page2).encode(),
        {"Link": f'<{server.base}/tree>; rel="prev"'},
    )
    assert mirror_sizes(server.base + "/tree") == {
        "1/sentinel12_s2_1_img.tif": 123,
        "1/sentinel12_s2_1_msk.tif": 45,
    }


def test_mirror_tree_url_is_pinned():
    url = provenance.MIRROR_TREE_URL.format(revision=chips.MIRROR_REVISION)
    assert f"/tree/{chips.MIRROR_REVISION}?recursive=true" in url


def test_zenodo_members_opens_one_remote_zip_per_part(monkeypatch):
    remotezip = pytest.importorskip("remotezip")
    opened: list[str] = []

    class FakeZip:
        def __init__(self, url: str):
            opened.append(url)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def getinfo(self, name: str):
            return SimpleNamespace(file_size=len(name), CRC=zlib.crc32(name.encode()))

    monkeypatch.setattr(remotezip, "RemoteZip", FakeZip)
    scenes = [make_scene(1, "train", 1), make_scene(2, "val", 2), make_scene(3, "test", 1)]
    got = zenodo_members(scenes)
    assert sorted(opened) == [chips.PART_URL.format(part=1), chips.PART_URL.format(part=2)]
    assert len(got) == 9
    for scene in scenes:
        for name in member_names(scene).values():
            assert got[name] == (len(name), zlib.crc32(name.encode()))


# ---------------------------------------------------------------- cli


@pytest.fixture
def cli_world(world, tmp_path, monkeypatch):
    scenes_csv = tmp_path / "scenes.csv"
    write_scenes(scenes_csv, SCENES)
    real_check = provenance.check
    seen = {}

    def fake_check(scenes, full_image):
        seen["full_image"] = full_image
        return real_check(
            scenes, full_image, fetch_crc=world.fetch_crc, zen=world.zen, sizes=world.sizes
        )

    monkeypatch.setattr(provenance, "check", fake_check)
    return SimpleNamespace(scenes=scenes_csv, out=tmp_path / "res" / "p.csv", seen=seen)


def test_cli_provenance_ok(cli_world, capsys):
    argv = ["provenance", "--scenes", str(cli_world.scenes), "--out", str(cli_world.out)]
    assert main([*argv, "--full-image", "s001,s002"]) == 0
    assert cli_world.seen["full_image"] == {"s001", "s002"}
    assert cli_world.out.exists()
    assert "6/6 sizes match; 4/4 mask CRCs match; 2/2 image CRCs match" in capsys.readouterr().out


def test_cli_provenance_exit_code_on_mismatch(cli_world, world, capsys):
    world.sizes[member_names(SCENES[1])["msk"]] += 1
    argv = ["provenance", "--scenes", str(cli_world.scenes), "--out", str(cli_world.out)]
    assert main(argv) == 1
    assert cli_world.seen["full_image"] == set()
    out = capsys.readouterr().out
    assert "MISMATCH s001 msk" in out and "5/6 sizes match" in out
    assert cli_world.out.exists()
