"""S1S2-Water scenes -> fixed 512 px chips, deterministic, one scene at a time.

Chip selection depends on a scene's water and valid-pixel masks only. Image chips are read as
512 px windows from the pinned mirror's Cloud-Optimised GeoTIFFs (default) or cut from the whole
scene pulled out of the Zenodo zip parts by ranged requests (authoritative, about 10x slower).
Both sources give the same chips; ``segment verify`` checks them by content hash. Nothing here
needs a GPU.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import shutil
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import AbstractContextManager, contextmanager
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

#: Unzipped copy of the same v1.0.1 files (CC BY 4.0 permits redistribution), pinned to a revision.
#: Its Cloud-Optimised GeoTIFFs have 512 px tiles aligned with the chip grid, so one chip is one
#: ranged read. `segment provenance` checks it against the Zenodo zip.
MIRROR_REVISION = "e0551de726bf2fdd12ba4b5c24b470f6bc42d824"
MIRROR_URL = (
    "https://huggingface.co/datasets/idomogalla/s1s2_water/resolve/" + MIRROR_REVISION + "/{rel}"
)
GDAL_HTTP_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "2",
}

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


def _open_range(url: str, start: int, end: int, timeout: float):
    import urllib.request

    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    return urllib.request.urlopen(req, timeout=timeout)


def _stream_member(
    url: str,
    info,
    dest: Path,
    chunk: int = 1 << 20,
    retries: int = 8,
    timeout: float = 120,
) -> Path:
    """Download one zip member with a streamed ranged GET and inflate it on the fly.

    ``remotezip.extract`` issues many small range requests and ran at under 0.5 MB/s against
    Zenodo; one streamed request runs at the connection's full rate. A stalled or dropped
    connection resumes from the last compressed byte received; it gives up after ``retries``
    failures in a row with no progress.
    """
    import struct
    import zlib

    if info.compress_type not in (0, 8):
        raise OSError(f"{info.filename}: unsupported compression {info.compress_type}")
    start = info.header_offset
    with _open_range(url, start, start + 29, timeout) as resp:
        header = resp.read(30)
    if header[:4] != b"PK\x03\x04":
        raise OSError(f"{info.filename}: not a local file header")
    name_len, extra_len = struct.unpack("<HH", header[26:30])
    data_start = start + 30 + name_len + extra_len
    data_end = data_start + info.compress_size - 1

    inflater = zlib.decompressobj(-15) if info.compress_type == 8 else None
    received, crc, failures, received_at_failure = 0, 0, 0, -1
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as out:
        while received < info.compress_size:
            try:
                with _open_range(url, data_start + received, data_end, timeout) as resp:
                    while received < info.compress_size:
                        block = resp.read(min(chunk, info.compress_size - received))
                        if not block:
                            raise OSError("connection closed early")
                        received += len(block)
                        data = inflater.decompress(block) if inflater else block
                        crc = zlib.crc32(data, crc)
                        out.write(data)
            except (OSError, TimeoutError) as exc:
                # Only consecutive failures without progress count towards the limit.
                failures = 1 if received > received_at_failure else failures + 1
                received_at_failure = received
                if failures > retries:
                    raise OSError(f"{info.filename}: gave up after {retries} retries") from exc
                log.warning("%s: %s at %d bytes, resuming", info.filename, exc, received)
        if inflater:
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


def read_mask(msk_path: Path, valid_path: Path) -> np.ndarray:
    """uint8 (H,W): water 1, land 0, IGNORE where the valid-pixel mask is 0."""
    import rasterio

    with rasterio.open(msk_path) as src:
        water = src.read(1)
    with rasterio.open(valid_path) as src:
        valid = src.read(1)
    mask = (water == 1).astype(np.uint8)
    mask[valid == 0] = IGNORE
    return mask


def read_scene(paths: dict[str, Path]) -> tuple[np.ndarray, np.ndarray]:
    """(image uint16 (5,H,W), mask uint8 (H,W)) from local GeoTIFFs.

    The all-zero (nodata) rule is applied per chip in :func:`cut_scene`, so chip selection
    depends on the masks only and both sources select the same chips.
    """
    import rasterio

    with rasterio.open(paths["img"]) as src:
        image = src.read(list(SOURCE_BANDS)).astype(np.uint16, copy=False)
    return image, read_mask(paths["msk"], paths["valid"])


ChipReader = Callable[[int, int, int], np.ndarray]


@contextmanager
def open_scene_zenodo(scene: Scene, tmp_dir: Path) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Whole scene from the Zenodo zip (authoritative, ~1.7 GB per scene, slow)."""
    yield read_scene(fetch_scene(scene, tmp_dir))


def _download(url: str, dest: Path) -> Path:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as resp, dest.open("wb") as out:
        shutil.copyfileobj(resp, out, 1 << 20)
    return dest


@contextmanager
def open_scene_mirror(scene: Scene, tmp_dir: Path) -> Iterator[tuple[ChipReader, np.ndarray]]:
    """Masks downloaded whole (about 2 MB); image chips read as windows over HTTP."""
    import rasterio
    from rasterio.windows import Window

    names = member_names(scene)
    msk = _download(MIRROR_URL.format(rel=names["msk"]), tmp_dir / "msk.tif")
    valid = _download(MIRROR_URL.format(rel=names["valid"]), tmp_dir / "valid.tif")
    mask = read_mask(msk, valid)
    with (
        rasterio.Env(**GDAL_HTTP_ENV),
        rasterio.open("/vsicurl/" + MIRROR_URL.format(rel=names["img"])) as src,
    ):
        if src.shape != mask.shape:
            raise OSError(f"{scene.scene}: image {src.shape} != mask {mask.shape}")

        def read(row: int, col: int, size: int) -> np.ndarray:
            window = Window(col, row, size, size)
            return src.read(list(SOURCE_BANDS), window=window).astype(np.uint16, copy=False)

        yield read, mask


SceneOpener = Callable[[Scene, Path], AbstractContextManager]


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
    image: np.ndarray | ChipReader,
    mask: np.ndarray,
    out_root: Path,
    chip: int = CHIP,
    k: int | None = None,
    seed: int = SEED,
) -> list[ChipRecord]:
    """Select and write this scene's chips under ``out_root/chips/<scene>/``.

    ``image`` is the whole scene or a ``read(row, col, size)`` window reader. Pixels where every
    band is 0 (nodata) become IGNORE; the record's fractions describe the saved mask.
    """
    k = CHIPS_PER_SCENE[scene.split] if k is None else k
    picked = select_chips(candidate_grid(mask, chip), scene.split, k, seed=seed + scene.number)
    records = []
    for cand in picked:
        cid = chip_id(scene, cand)
        rel = f"chips/{scene.scene}/{cid}.npz"
        if callable(image):
            img = np.ascontiguousarray(image(cand.row, cand.col, chip))
        else:
            img = np.ascontiguousarray(
                image[:, cand.row : cand.row + chip, cand.col : cand.col + chip]
            )
        msk = mask[cand.row : cand.row + chip, cand.col : cand.col + chip].copy()
        msk[(img == 0).all(axis=0)] = IGNORE
        sha = save_chip(out_root / rel, img, msk)
        n_valid = int((msk != IGNORE).sum())
        records.append(
            ChipRecord(
                chip_id=cid,
                scene=scene.scene,
                s2_tile=scene.s2_tile,
                split=scene.split,
                row=cand.row,
                col=cand.col,
                valid_frac=n_valid / msk.size,
                water_frac=int((msk == 1).sum()) / n_valid if n_valid else 0.0,
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
    opener: SceneOpener = open_scene_mirror,
    **cut_kwargs,
) -> Path:
    """Open, cut and clean up one scene; resumable via ``out_root/records/<scene>.csv``."""
    done = out_root / "records" / f"{scene.scene}.csv"
    if done.exists():
        return done
    tmp = out_root / "tmp" / scene.scene
    try:
        with opener(scene, tmp) as (image, mask):
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
