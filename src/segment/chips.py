"""S1S2-Water scenes -> fixed 512 px chips, deterministic, one scene at a time.

Each scene's Sentinel-2 image, water mask and valid-pixel mask are pulled out of the Zenodo
zip parts by HTTP range requests (no full-part download), cut into a non-overlapping grid,
sampled with a seed derived from the scene number, written as chips, and the full-size
GeoTIFFs are deleted. Nothing here needs a GPU.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import shutil
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import astuple, dataclass, fields
from pathlib import Path

import numpy as np

from segment.bands import IGNORE
from segment.data import ChipRecord, content_sha256, load_chip, save_chip, write_manifest

log = logging.getLogger(__name__)

ZENODO_RECORD = "11278238"
SOURCE_URL = f"https://zenodo.org/records/{ZENODO_RECORD}"
PART_URL = "https://zenodo.org/api/records/" + ZENODO_RECORD + "/files/part{part}.zip/content"
N_PARTS = 6
LICENCE = "CC-BY-4.0"

#: Source band order is blue, green, red, nir, swir1, swir2; chips keep the first five.
SOURCE_BANDS = (1, 2, 3, 4, 5)  # rasterio is 1-based

CHIP = 512
MIN_VALID_FRAC = 0.8
WATER_CHIP_FRAC = 0.01
CHIPS_PER_SCENE = {"train": 64, "val": 32, "test": 32}
SEED = 45


@dataclass(frozen=True)
class Scene:
    scene: str  # "s071" for sample 71
    number: int
    part: int
    split: str
    s2_tile: str
    date: str
    landcover: str
    bbox: str  # minx miny maxx maxy, EPSG:3857 as in the source metadata


SCENE_COLUMNS = tuple(f.name for f in fields(Scene))


def scene_name(number: int) -> str:
    return f"s{number:03d}"


def _tile_of(s2_srcid: str) -> str:
    return next(t for t in s2_srcid.split("_") if len(t) == 6 and t.startswith("T"))


def scene_from_meta(meta: dict, part: int) -> Scene:
    props = meta["properties"]
    number = int(meta["id"].split("_")[1])
    return Scene(
        scene=scene_name(number),
        number=number,
        part=part,
        split=props["split"],
        s2_tile=_tile_of(props["s2_srcids"][0]),
        date=props["date_s2"],
        landcover=props["landcover"],
        bbox=" ".join(f"{v:.2f}" for v in meta["bbox"]),
    )


def list_scenes_remote() -> list[Scene]:
    """Read every scene's STAC metadata from the six zip parts (a few KB each)."""
    from remotezip import RemoteZip

    scenes = []
    for part in range(1, N_PARTS + 1):
        with RemoteZip(PART_URL.format(part=part)) as z:
            for info in z.infolist():
                if info.filename.endswith("_meta.json"):
                    scenes.append(scene_from_meta(json.loads(z.read(info.filename)), part))
    return sorted(scenes, key=lambda s: s.number)


def write_scenes(path: Path, scenes: list[Scene]) -> None:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(SCENE_COLUMNS)
    writer.writerows(astuple(s) for s in sorted(scenes, key=lambda s: s.number))
    Path(path).write_bytes(buf.getvalue().encode("utf-8"))


def read_scenes(path: Path) -> list[Scene]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return [
            Scene(**{**r, "number": int(r["number"]), "part": int(r["part"])})
            for r in csv.DictReader(fh)
        ]


def member_names(scene: Scene) -> dict[str, str]:
    n = scene.number
    return {
        "img": f"{n}/sentinel12_s2_{n}_img.tif",
        "msk": f"{n}/sentinel12_s2_{n}_msk.tif",
        "valid": f"{n}/sentinel12_s2_{n}_valid.tif",
    }


def _stream_member(url: str, info, dest: Path, chunk: int = 1 << 20) -> Path:
    """Download one zip member with a single ranged GET and inflate it on the fly.

    ``remotezip.extract`` issues many small range requests and ran at under 0.5 MB/s against
    Zenodo; one streamed request runs at the connection's full rate.
    """
    import struct
    import urllib.request
    import zlib

    start = info.header_offset
    end = start + 30 + len(info.filename.encode()) + 1024 + info.compress_size
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(req, timeout=120) as resp, dest.open("wb") as out:
        header = resp.read(30)
        if header[:4] != b"PK\x03\x04":
            raise OSError(f"{info.filename}: not a local file header")
        name_len, extra_len = struct.unpack("<HH", header[26:30])
        resp.read(name_len + extra_len)
        if info.compress_type == 8:
            inflater = zlib.decompressobj(-15)
        elif info.compress_type != 0:
            raise OSError(f"{info.filename}: unsupported compression {info.compress_type}")
        remaining = info.compress_size
        crc = 0
        while remaining:
            block = resp.read(min(chunk, remaining))
            if not block:
                raise OSError(f"{info.filename}: connection closed early")
            remaining -= len(block)
            data = inflater.decompress(block) if info.compress_type == 8 else block
            crc = zlib.crc32(data, crc)
            out.write(data)
        if info.compress_type == 8:
            tail = inflater.flush()
            crc = zlib.crc32(tail, crc)
            out.write(tail)
    if crc != info.CRC:
        raise OSError(f"{info.filename}: CRC mismatch")
    return dest


def fetch_scene(scene: Scene, tmp_dir: Path) -> dict[str, Path]:
    """Extract the scene's three Sentinel-2 GeoTIFFs from its zip part into ``tmp_dir``."""
    from remotezip import RemoteZip

    url = PART_URL.format(part=scene.part)
    with RemoteZip(url) as z:
        infos = {key: z.getinfo(name) for key, name in member_names(scene).items()}
    return {
        key: _stream_member(url, info, tmp_dir / Path(info.filename).name)
        for key, info in infos.items()
    }


def read_scene(paths: dict[str, Path]) -> tuple[np.ndarray, np.ndarray]:
    """(image uint16 (5,H,W), mask uint8 (H,W)); invalid or all-zero pixels become IGNORE."""
    import rasterio

    with rasterio.open(paths["img"]) as src:
        image = src.read(list(SOURCE_BANDS)).astype(np.uint16, copy=False)
    with rasterio.open(paths["msk"]) as src:
        water = src.read(1)
    with rasterio.open(paths["valid"]) as src:
        valid = src.read(1)
    mask = (water == 1).astype(np.uint8)
    mask[(valid == 0) | (image == 0).all(axis=0)] = IGNORE
    return image, mask


@dataclass(frozen=True)
class Candidate:
    row: int
    col: int
    valid_frac: float
    water_frac: float  # of valid pixels


def candidate_grid(mask: np.ndarray, chip: int = CHIP) -> list[Candidate]:
    """Every full, non-overlapping chip of the grid anchored at (0, 0), row-major."""
    nr, nc = mask.shape[0] // chip, mask.shape[1] // chip
    blocks = mask[: nr * chip, : nc * chip].reshape(nr, chip, nc, chip).swapaxes(1, 2)
    valid = (blocks != IGNORE).sum(axis=(2, 3))
    water = (blocks == 1).sum(axis=(2, 3))
    out = []
    for i in range(nr):
        for j in range(nc):
            v = int(valid[i, j])
            out.append(
                Candidate(i * chip, j * chip, v / chip**2, float(water[i, j]) / v if v else 0.0)
            )
    return out


def select_chips(candidates: list[Candidate], split: str, k: int, seed: int) -> list[Candidate]:
    """Pick ``k`` eligible chips (valid share >= MIN_VALID_FRAC).

    val/test: uniform without replacement, so the sample represents the scene.
    train: up to half from chips with >= WATER_CHIP_FRAC water, the rest uniform from what is
    left, so the model sees water in scenes where it is rare. Returned in grid order.
    """
    rng = np.random.default_rng(seed)
    eligible = [c for c in candidates if c.valid_frac >= MIN_VALID_FRAC]
    if len(eligible) <= k:
        return eligible
    if split == "train":
        watery = [i for i, c in enumerate(eligible) if c.water_frac >= WATER_CHIP_FRAC]
        n_w = min(k // 2, len(watery))
        first = set(rng.choice(watery, size=n_w, replace=False).tolist()) if n_w else set()
        rest = [i for i in range(len(eligible)) if i not in first]
        chosen = first | set(rng.choice(rest, size=k - n_w, replace=False).tolist())
    else:
        chosen = set(rng.choice(len(eligible), size=k, replace=False).tolist())
    return [eligible[i] for i in sorted(chosen)]


def chip_id(scene: Scene, cand: Candidate) -> str:
    return f"{scene.scene}_r{cand.row:05d}_c{cand.col:05d}"


def cut_scene(
    scene: Scene,
    image: np.ndarray,
    mask: np.ndarray,
    out_root: Path,
    chip: int = CHIP,
    k: int | None = None,
    seed: int = SEED,
) -> list[ChipRecord]:
    """Select and write this scene's chips under ``out_root/chips/<scene>/``."""
    k = CHIPS_PER_SCENE[scene.split] if k is None else k
    picked = select_chips(candidate_grid(mask, chip), scene.split, k, seed=seed + scene.number)
    records = []
    for cand in picked:
        cid = chip_id(scene, cand)
        rel = f"chips/{scene.scene}/{cid}.npz"
        img = np.ascontiguousarray(image[:, cand.row : cand.row + chip, cand.col : cand.col + chip])
        msk = np.ascontiguousarray(mask[cand.row : cand.row + chip, cand.col : cand.col + chip])
        sha = save_chip(out_root / rel, img, msk)
        records.append(
            ChipRecord(
                chip_id=cid,
                scene=scene.scene,
                s2_tile=scene.s2_tile,
                split=scene.split,
                row=cand.row,
                col=cand.col,
                valid_frac=cand.valid_frac,
                water_frac=cand.water_frac,
                image_path_rel=rel,
                licence=LICENCE,
                source_url=SOURCE_URL,
                sha256=sha,
            )
        )
    return records


def build_scene(
    scene: Scene,
    out_root: Path,
    fetch: Callable[[Scene, Path], dict[str, Path]] = fetch_scene,
    read: Callable[[dict[str, Path]], tuple[np.ndarray, np.ndarray]] = read_scene,
    **cut_kwargs,
) -> Path:
    """Fetch, cut and clean up one scene; resumable via ``out_root/records/<scene>.csv``."""
    done = out_root / "records" / f"{scene.scene}.csv"
    if done.exists():
        return done
    tmp = out_root / "tmp" / scene.scene
    try:
        image, mask = read(fetch(scene, tmp))
        records = cut_scene(scene, image, mask, out_root, **cut_kwargs)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    done.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(done, records)
    log.info("%s %s: %d chips", scene.scene, scene.split, len(records))
    return done


def build_all(
    scenes: Iterable[Scene], out_root: Path, workers: int = 4, **build_kwargs
) -> list[Path]:
    """Build every scene with a thread pool (downloads dominate); failures are re-raised."""
    from segment.data import read_manifest

    scenes = list(scenes)
    paths = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(build_scene, s, out_root, **build_kwargs): s for s in scenes}
        for fut in as_completed(futures):
            paths.append(fut.result())
    records = [r for p in sorted(paths) for r in read_manifest(p)]
    write_manifest(out_root / "manifest.csv", records)
    return paths


def verify_chips(records: list[ChipRecord], root: Path) -> list[str]:
    """Chips missing or whose content hash differs from the manifest."""
    problems = []
    for rec in records:
        path = root / rec.image_path_rel
        if not path.exists():
            problems.append(f"{rec.chip_id}: missing")
        elif content_sha256(*load_chip(path)) != rec.sha256:
            problems.append(f"{rec.chip_id}: sha256 mismatch")
    return problems
