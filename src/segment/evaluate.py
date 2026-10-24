"""Turn committed per-chip CSVs into the results summary. Reads ``results/`` only, never chips."""

from __future__ import annotations

from pathlib import Path


def summarise(results_dir: Path) -> dict:
    """Aggregate every ``per_chip_*.csv`` in ``results_dir``.

    For each (method, split): pooled IoU/Dice/precision/recall (pixel-weighted), macro IoU over
    chips (chips with no water in both target and prediction are excluded from the macro mean and
    counted), per-scene pooled IoU, and a chip-bootstrap 95% CI of pooled IoU.
    Also: the U-Net minus NDWI-Otsu IoU gap on test with a paired chip bootstrap CI, the U-Net's
    val-to-test IoU drop, and the pre-registered verdicts read from ``prereg.json``.
    """
    raise NotImplementedError


def write_summary(summary: dict, results_dir: Path) -> tuple[Path, Path]:
    """Write ``summary.json`` and ``summary.md`` into ``results_dir``."""
    raise NotImplementedError
