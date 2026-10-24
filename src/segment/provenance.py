"""Check the pinned mirror's Sentinel-2 files against the Zenodo zip, the authoritative source.

Sizes come from the mirror's file tree and the zip's central directory; CRC-32s come from
streaming the mirror's files and the zip's stored per-member CRC. Masks are always CRC-checked;
the (large) images only for the scenes named in ``full_image``.
"""

from __future__ import annotations

import csv
import io
import json
import re
import urllib.request
import zlib
from collections import defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from segment import chips
from segment.chips import Scene, member_names

LAYERS = ("img", "msk", "valid")
REPORT_COLUMNS = (
    "scene",
    "split",
    "layer",
    "zenodo_size",
    "mirror_size",
    "size_ok",
    "zenodo_crc",
    "mirror_crc",
    "crc_ok",
)
MIRROR_TREE_URL = (
    "https://huggingface.co/api/datasets/idomogalla/s1s2_water/tree/{revision}?recursive=true"
)
_NEXT_LINK = re.compile(r'<([^>]+)>\s*;\s*rel="next"')

#: (file_size, CRC-32) of one zip member, or the mirror's streamed (crc, n_bytes).
Member = tuple[int, int]


def zenodo_members(scenes: Iterable[Scene]) -> dict[str, Member]:
    """(file_size, CRC-32) of each scene's three Sentinel-2 members, one RemoteZip per part."""
    from remotezip import RemoteZip

    by_part: dict[int, list[Scene]] = defaultdict(list)
    for scene in scenes:
        by_part[scene.part].append(scene)
    out: dict[str, Member] = {}
    for part, group in sorted(by_part.items()):
        with RemoteZip(chips.PART_URL.format(part=part)) as z:
            for scene in group:
                for name in member_names(scene).values():
                    info = z.getinfo(name)
                    out[name] = (info.file_size, info.CRC)
    return out


def mirror_sizes(url: str | None = None) -> dict[str, int]:
    """File sizes of the mirror at the pinned revision, following the paginated tree listing."""
    next_url: str | None = url or MIRROR_TREE_URL.format(revision=chips.MIRROR_REVISION)
    sizes: dict[str, int] = {}
    while next_url:
        with urllib.request.urlopen(next_url, timeout=120) as resp:
            entries = json.loads(resp.read())
            link = resp.headers.get("Link", "")
        for entry in entries:
            if entry.get("type", "file") == "file" and "size" in entry:
                sizes[entry["path"]] = int(entry["size"])
        match = _NEXT_LINK.search(link)
        next_url = match.group(1) if match else None
    return sizes


def crc32_url(url: str, chunk: int = 1 << 20) -> tuple[int, int]:
    """(CRC-32, byte count) of a URL's body, streamed in 1 MiB chunks."""
    crc = n = 0
    with urllib.request.urlopen(url, timeout=120) as resp:
        while block := resp.read(chunk):
            crc = zlib.crc32(block, crc)
            n += len(block)
    return crc, n


def check(
    scenes: Iterable[Scene],
    full_image: set[str],
    fetch_crc: Callable[[str], tuple[int, int]] = crc32_url,
    zen: dict[str, Member] | None = None,
    sizes: dict[str, int] | None = None,
    workers: int = 8,
) -> list[dict]:
    """One row per (scene, layer): sizes always, CRCs for msk/valid and for ``full_image`` img.

    ``crc_ok`` is True only when the streamed CRC and byte count both equal Zenodo's; it is ""
    when the layer was not CRC-checked.
    """
    scenes = list(scenes)
    unknown = full_image - {s.scene for s in scenes}
    if unknown:
        raise ValueError(f"--full-image names not in the scene list: {sorted(unknown)}")
    zen = zenodo_members(scenes) if zen is None else zen
    sizes = mirror_sizes() if sizes is None else sizes

    rows: list[dict] = []
    jobs: list[tuple[dict, str]] = []
    for scene in sorted(scenes, key=lambda s: s.number):
        for layer, name in member_names(scene).items():
            z_size, z_crc = zen[name]
            m_size = sizes.get(name)
            row = {
                "scene": scene.scene,
                "split": scene.split,
                "layer": layer,
                "zenodo_size": z_size,
                "mirror_size": "" if m_size is None else m_size,
                "size_ok": m_size == z_size,
                "zenodo_crc": f"{z_crc:08x}",
                "mirror_crc": "",
                "crc_ok": "",
            }
            rows.append(row)
            if (layer != "img" or scene.scene in full_image) and m_size is not None:
                jobs.append((row, name))
            elif layer != "img" or scene.scene in full_image:
                row["crc_ok"] = False  # wanted but the mirror has no such file

    def run(job: tuple[dict, str]) -> tuple[int, int]:
        return fetch_crc(chips.MIRROR_URL.format(rel=job[1]))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run, jobs))
    for (row, name), (crc, n_bytes) in zip(jobs, results, strict=True):
        z_size, z_crc = zen[name]
        row["mirror_crc"] = f"{crc:08x}"
        row["crc_ok"] = crc == z_crc and n_bytes == z_size
    return sorted(rows, key=lambda r: (r["scene"], r["layer"]))


def _count(rows: list[dict], key: str, layers: tuple[str, ...]) -> tuple[int, int]:
    checked = [r for r in rows if r["layer"] in layers and r[key] != ""]
    return sum(1 for r in checked if r[key] is True), len(checked)


def summary_line(rows: list[dict]) -> str:
    sizes_ok, sizes_n = _count(rows, "size_ok", LAYERS)
    mask_ok, mask_n = _count(rows, "crc_ok", ("msk", "valid"))
    img_ok, img_n = _count(rows, "crc_ok", ("img",))
    return (
        f"{sizes_ok}/{sizes_n} sizes match; {mask_ok}/{mask_n} mask CRCs match; "
        f"{img_ok}/{img_n} image CRCs match"
    )


def mismatches(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["size_ok"] is not True or r["crc_ok"] is False]


def write_report(rows: list[dict], path: Path) -> str:
    """Write the rows as CSV (LF, sorted by scene and layer); return the one-line summary."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(REPORT_COLUMNS)
    for row in sorted(rows, key=lambda r: (r["scene"], r["layer"])):
        writer.writerow([row[c] for c in REPORT_COLUMNS])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.getvalue().encode("utf-8"))
    return summary_line(rows)
