"""Turn committed per-chip CSVs into the results summary. Reads ``results/`` only, never chips."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from segment.metrics import Counts, bootstrap_ci, dice, iou, precision, recall

SEED = 0
UNET = "unet"
BASELINE = "ndwi_otsu"
METHOD_ORDER = (UNET, BASELINE, "mndwi_otsu", "constant")
SPLIT_ORDER = ("train", "val", "test")

RULES = {
    "H1": (
        "gap = pooled test IoU(unet) - pooled test IoU(ndwi_otsu), paired chip-bootstrap 95% CI. "
        "supported if CI lower bound >= min_gap_iou; not supported if CI upper bound < "
        "min_gap_iou; inconclusive otherwise."
    ),
    "H2": (
        "drop = pooled IoU(unet, val) - pooled IoU(unet, test). poor generalisation if "
        "drop > max_drop_iou; within bound otherwise."
    ),
}


def _read_rows(results_dir: Path) -> list[dict]:
    files = sorted(Path(results_dir).glob("per_chip_*.csv"))
    if not files:
        raise FileNotFoundError(f"no per_chip_*.csv in {results_dir}")
    rows = []
    for path in files:
        with path.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(
                    {
                        "method": r["method"],
                        "split": r["split"],
                        "chip_id": r["chip_id"],
                        "scene": r["scene"],
                        "counts": Counts(int(r["tp"]), int(r["fp"]), int(r["fn"]), int(r["tn"])),
                    }
                )
    return rows


def _pooled(counts: list[Counts]) -> Counts:
    total = Counts.zero()
    for c in counts:
        total = total + c
    return total


def _group_stats(rows: list[dict], n_boot: int) -> dict:
    counts = [r["counts"] for r in rows]
    total = _pooled(counts)
    chip_iou = [iou(c) for c in counts]
    valid = [v for v in chip_iou if not math.isnan(v)]
    scenes: dict[str, list[Counts]] = defaultdict(list)
    for r in rows:
        scenes[r["scene"]].append(r["counts"])
    lo, hi = bootstrap_ci(counts, iou, n_boot=n_boot, seed=SEED)
    return {
        "n_chips": len(rows),
        "counts": {"tp": total.tp, "fp": total.fp, "fn": total.fn, "tn": total.tn},
        "iou": iou(total),
        "iou_ci": [lo, hi],
        "dice": dice(total),
        "precision": precision(total),
        "recall": recall(total),
        "macro_iou": sum(valid) / len(valid) if valid else float("nan"),
        "n_nan_chips": len(chip_iou) - len(valid),
        "per_scene": {
            s: {"n_chips": len(c), "iou": iou(_pooled(c))} for s, c in sorted(scenes.items())
        },
    }


def _as_counts(row: np.ndarray) -> Counts:
    return Counts(*(int(v) for v in row))


def _paired_gap(a: dict[str, Counts], b: dict[str, Counts], n_boot: int) -> dict | None:
    """Pooled IoU(a) - pooled IoU(b) over chips present in both; one index draw per replicate."""
    ids = sorted(set(a) & set(b))
    if not ids:
        return None
    ca = np.array([[a[i].tp, a[i].fp, a[i].fn, a[i].tn] for i in ids], dtype=np.int64)
    cb = np.array([[b[i].tp, b[i].fp, b[i].fn, b[i].tn] for i in ids], dtype=np.int64)
    point = iou(_as_counts(ca.sum(axis=0))) - iou(_as_counts(cb.sum(axis=0)))
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    sa, sb = ca[idx].sum(axis=1), cb[idx].sum(axis=1)
    diffs = np.array([iou(_as_counts(x)) - iou(_as_counts(y)) for x, y in zip(sa, sb, strict=True)])
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs):
        lo, hi = (float(q) for q in np.quantile(diffs, [0.025, 0.975]))
    else:
        lo = hi = float("nan")
    return {"n_chips": len(ids), "point": float(point), "ci": [lo, hi]}


def _drop(groups: dict, method: str) -> dict | None:
    val = groups.get(method, {}).get("val")
    test = groups.get(method, {}).get("test")
    if val is None or test is None:
        return None
    return {"val_iou": val["iou"], "test_iou": test["iou"], "drop": val["iou"] - test["iou"]}


def _verdicts(prereg: dict | None, comparisons: dict) -> dict:
    verdicts: dict[str, str | None] = {"H1": None, "H2": None}
    if prereg is None:
        return verdicts
    gap, drop = comparisons.get("gap_test"), comparisons.get("drop_unet")
    if gap is not None:
        lo, hi = gap["ci"]
        min_gap = prereg["min_gap_iou"]
        if lo >= min_gap:
            verdicts["H1"] = "supported"
        elif hi < min_gap:
            verdicts["H1"] = "not supported"
        else:
            verdicts["H1"] = "inconclusive"
    if drop is not None and not math.isnan(drop["drop"]):
        verdicts["H2"] = (
            "poor generalisation" if drop["drop"] > prereg["max_drop_iou"] else "within bound"
        )
    return verdicts


def _read_prereg(results_dir: Path) -> dict | None:
    path = Path(results_dir) / "prereg.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    missing = {"min_gap_iou", "max_drop_iou"} - set(data)
    if missing:
        raise ValueError(f"{path}: missing {sorted(missing)}")
    return {k: float(data[k]) for k in ("min_gap_iou", "max_drop_iou")}


def _clean(obj):
    """Recursively replace NaN/inf floats with None so the result is strict JSON."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def summarise(results_dir: Path, n_boot: int = 2000) -> dict:
    """Aggregate every ``per_chip_*.csv`` in ``results_dir``.

    For each (method, split): pooled IoU/Dice/precision/recall (pixel-weighted), macro IoU over
    chips (chips with no water in both target and prediction are excluded from the macro mean and
    counted), per-scene pooled IoU, and a chip-bootstrap 95% CI of pooled IoU.
    Also: the U-Net minus NDWI-Otsu IoU gap on test with a paired chip bootstrap CI, the U-Net's
    val-to-test IoU drop, and the pre-registered verdicts read from ``prereg.json``.
    """
    results_dir = Path(results_dir)
    rows = _read_rows(results_dir)
    by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_group[(r["method"], r["split"])].append(r)

    groups: dict[str, dict[str, dict]] = defaultdict(dict)
    for (method, split), grp in sorted(by_group.items()):
        groups[method][split] = _group_stats(grp, n_boot)

    test_counts = {
        m: {r["chip_id"]: r["counts"] for r in by_group[(m, "test")]}
        for m in (UNET, BASELINE)
        if (m, "test") in by_group
    }
    gap = None
    if UNET in test_counts and BASELINE in test_counts:
        gap = _paired_gap(test_counts[UNET], test_counts[BASELINE], n_boot)
    comparisons = {
        "gap_test": gap,
        "drop_unet": _drop(groups, UNET),
        "drop_ndwi_otsu": _drop(groups, BASELINE),
    }
    prereg = _read_prereg(results_dir)
    summary = {
        "n_boot": n_boot,
        "seed": SEED,
        "groups": dict(groups),
        "comparisons": comparisons,
        "prereg": prereg,
        "verdicts": _verdicts(prereg, comparisons),
        "rules": RULES,
    }
    return _clean(summary)


def _f(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _ordered(names: list[str], preferred: tuple[str, ...]) -> list[str]:
    return [n for n in preferred if n in names] + sorted(n for n in names if n not in preferred)


def _render_md(summary: dict) -> str:
    groups = summary["groups"]
    methods = _ordered(list(groups), METHOD_ORDER)
    lines = ["# Results summary", ""]
    lines += [
        "| method | split | chips | IoU [95% CI] | Dice | precision | recall | macro IoU |",
        "|---|---|---:|---|---:|---:|---:|---|",
    ]
    for m in methods:
        for s in _ordered(list(groups[m]), SPLIT_ORDER):
            g = groups[m][s]
            lo, hi = g["iou_ci"]
            macro = _f(g["macro_iou"])
            if g["n_nan_chips"]:
                macro += f" ({g['n_nan_chips']} NaN chips excluded)"
            lines.append(
                f"| {m} | {s} | {g['n_chips']} | {_f(g['iou'])} [{_f(lo)}, {_f(hi)}] "
                f"| {_f(g['dice'])} | {_f(g['precision'])} | {_f(g['recall'])} | {macro} |"
            )
    lines.append("")

    comp = summary["comparisons"]
    gap = comp["gap_test"]
    if gap is not None:
        lo, hi = gap["ci"]
        lines.append(
            f"- Gap on test (unet - ndwi_otsu, pooled IoU, {gap['n_chips']} paired chips): "
            f"{_f(gap['point'])} [{_f(lo)}, {_f(hi)}]"
        )
    for label, d in (("unet", comp["drop_unet"]), ("ndwi_otsu", comp["drop_ndwi_otsu"])):
        if d is not None:
            lines.append(
                f"- Val-to-test drop ({label}): {_f(d['drop'])} "
                f"(val {_f(d['val_iou'])} -> test {_f(d['test_iou'])})"
            )
    lines.append("")

    lines += ["## Pre-registered verdicts", ""]
    prereg = summary["prereg"]
    if prereg is None:
        lines.append("No prereg.json found: no verdicts.")
    else:
        lines.append(
            f"Thresholds: min_gap_iou = {prereg['min_gap_iou']:.3f}, "
            f"max_drop_iou = {prereg['max_drop_iou']:.3f}."
        )
    lines.append("")
    for h in ("H1", "H2"):
        verdict = summary["verdicts"][h]
        lines.append(f"- {h}: {verdict or 'n/a'}. Rule: {summary['rules'][h]}")
    lines.append("")

    test_methods = [m for m in methods if "test" in groups[m]]
    if test_methods:
        scenes = sorted({s for m in test_methods for s in groups[m]["test"]["per_scene"]})
        lines += ["## Per-scene IoU (test)", ""]
        lines.append("| scene | " + " | ".join(test_methods) + " |")
        lines.append("|---|" + "---:|" * len(test_methods))
        for sc in scenes:
            cells = []
            for m in test_methods:
                ps = groups[m]["test"]["per_scene"].get(sc)
                cells.append("n/a" if ps is None else f"{_f(ps['iou'])} (n={ps['n_chips']})")
            lines.append(f"| {sc} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def write_summary(summary: dict, results_dir: Path) -> tuple[Path, Path]:
    """Write ``summary.json`` and ``summary.md`` into ``results_dir``."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = results_dir / "summary.json", results_dir / "summary.md"
    text = json.dumps(_clean(summary), indent=2, sort_keys=True, allow_nan=False) + "\n"
    json_path.write_bytes(text.encode("utf-8"))
    md_path.write_bytes(_render_md(summary).encode("utf-8"))
    return json_path, md_path
