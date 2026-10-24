"""``segment`` command line: scenes -> chips -> baselines / U-Net -> per-chip CSVs -> summary."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np

DATA = Path("data")
RESULTS = Path("results")
BASELINE_FIT = "baseline_fit.json"


def _manifest_records(manifest: Path):
    from segment.data import read_manifest

    return read_manifest(manifest)


def cmd_scenes(args: argparse.Namespace) -> int:
    from segment.chips import list_scenes_remote, write_scenes

    scenes = list_scenes_remote()
    write_scenes(args.out, scenes)
    print(f"{len(scenes)} scenes -> {args.out}: {dict(Counter(s.split for s in scenes))}")
    return 0


def cmd_chips(args: argparse.Namespace) -> int:
    from segment.chips import build_all, read_scenes
    from segment.data import check_splits, read_manifest

    scenes = read_scenes(args.scenes)
    if args.only:
        wanted = set(args.only.split(","))
        scenes = [s for s in scenes if s.scene in wanted]
    build_all(scenes, args.root, workers=args.workers)
    records = read_manifest(args.root / "manifest.csv")
    print(f"{len(records)} chips from {len(scenes)} scenes -> {args.root / 'manifest.csv'}")
    problems = check_splits(records) if not args.only else []
    for p in problems:
        print("LEAKAGE:", p)
    return 1 if problems else 0


def cmd_audit(args: argparse.Namespace) -> int:
    from segment.data import check_splits

    records = _manifest_records(args.manifest)
    problems = check_splits(records)
    by_split = Counter(r.split for r in records)
    scenes = {s: len({r.scene for r in records if r.split == s}) for s in by_split}
    print(f"chips per split {dict(by_split)}; scenes per split {scenes}")
    for p in problems:
        print("LEAKAGE:", p)
    print("clean" if not problems else f"{len(problems)} violation(s)")
    return 1 if problems else 0


def cmd_verify(args: argparse.Namespace) -> int:
    from segment.chips import verify_chips

    problems = verify_chips(_manifest_records(args.manifest), args.root)
    for p in problems:
        print(p)
    print("all chips match the manifest" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


def _chips(records, root: Path, split: str):
    from segment.data import load_chip

    for rec in records:
        if rec.split == split:
            yield load_chip(root / rec.image_path_rel)


def cmd_baseline(args: argparse.Namespace) -> int:
    from segment.baseline import fit_constant, fit_index_threshold
    from segment.predict import score_split

    records = _manifest_records(args.manifest)
    ndwi = fit_index_threshold(_chips(records, args.root, "train"), index="ndwi")
    mndwi = fit_index_threshold(_chips(records, args.root, "train"), index="mndwi")
    const = fit_constant(mask for _, mask in _chips(records, args.root, "train"))
    args.results.mkdir(parents=True, exist_ok=True)
    fit = {"ndwi_otsu": ndwi.threshold, "mndwi_otsu": mndwi.threshold, "constant": const.value}
    (args.results / BASELINE_FIT).write_text(json.dumps(fit, indent=2, sort_keys=True) + "\n")
    print("fitted on train:", fit)
    methods = {"ndwi_otsu": ndwi.predict, "mndwi_otsu": mndwi.predict, "constant": const.predict}
    for split in args.splits:
        for name, fn in methods.items():
            out = args.results / f"per_chip_{name}_{split}.csv"
            rows = score_split(name, fn, args.manifest, args.root, split, out)
            print(f"{name} {split}: {len(rows)} chips -> {out}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from segment.train import TrainConfig, train

    cfg = TrainConfig(
        chips_dir=args.root,
        manifest=args.manifest,
        out_dir=args.out,
        encoder=args.encoder,
        encoder_weights=None if args.encoder_weights == "none" else args.encoder_weights,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
    record = train(cfg)
    print(f"best epoch {record['best_epoch']} val IoU {record['best_val_iou']:.4f} -> {args.out}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from segment.predict import score_split, unet_predict_fn

    fn = unet_predict_fn(args.checkpoint, device=args.device)
    args.results.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        out = args.results / f"per_chip_unet_{split}.csv"
        masks = args.masks_dir / f"masks_unet_{split}.npz" if args.masks_dir else None
        rows = score_split("unet", fn, args.manifest, args.root, split, out, masks_out=masks)
        print(f"unet {split}: {len(rows)} chips -> {out}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from segment.evaluate import summarise, write_summary

    summary = summarise(args.results)
    json_path, md_path = write_summary(summary, args.results)
    print(md_path.read_text(encoding="utf-8"))
    print(f"-> {json_path}, {md_path}")
    return 0


def cmd_overlay(args: argparse.Namespace) -> int:
    from segment.baseline import IndexThreshold
    from segment.data import load_chip
    from segment.viz import save_panel

    fit = json.loads((args.results / BASELINE_FIT).read_text())
    ndwi = IndexThreshold("ndwi", fit["ndwi_otsu"])
    unet_masks = None
    if args.unet_masks and args.unet_masks.exists():
        unet_masks = np.load(args.unet_masks)
    by_id = {r.chip_id: r for r in _manifest_records(args.manifest)}
    args.out.mkdir(parents=True, exist_ok=True)
    for cid in args.chip_ids:
        image, mask = load_chip(args.root / by_id[cid].image_path_rel)
        preds = {"ndwi_otsu": ndwi.predict(image)}
        if unet_masks is not None and cid in unet_masks:
            preds["unet"] = unet_masks[cid]
        save_panel(args.out / f"{cid}.png", image, mask, preds)
        print(args.out / f"{cid}.png")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="segment", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    def manifest_args(sp):
        sp.add_argument("--manifest", type=Path, default=DATA / "manifest.csv")
        sp.add_argument("--root", type=Path, default=DATA, help="chip paths are relative to this")

    sp = sub.add_parser("scenes", help="read scene metadata from Zenodo -> data/scenes.csv")
    sp.add_argument("--out", type=Path, default=DATA / "scenes.csv")
    sp.set_defaults(func=cmd_scenes)

    sp = sub.add_parser("chips", help="download scenes one at a time and cut chips")
    sp.add_argument("--scenes", type=Path, default=DATA / "scenes.csv")
    sp.add_argument("--root", type=Path, default=DATA)
    sp.add_argument("--workers", type=int, default=4)
    sp.add_argument("--only", help="comma-separated scene names, e.g. s071,s073")
    sp.set_defaults(func=cmd_chips)

    sp = sub.add_parser("audit", help="leakage audit of the manifest")
    sp.add_argument("--manifest", type=Path, default=DATA / "manifest.csv")
    sp.set_defaults(func=cmd_audit)

    sp = sub.add_parser("verify", help="check every chip's content hash against the manifest")
    manifest_args(sp)
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("baseline", help="fit index thresholds on train, score val/test")
    manifest_args(sp)
    sp.add_argument("--results", type=Path, default=RESULTS)
    sp.add_argument("--splits", nargs="+", default=["val", "test"])
    sp.set_defaults(func=cmd_baseline)

    sp = sub.add_parser("train", help="train the U-Net (best checkpoint chosen on val)")
    manifest_args(sp)
    sp.add_argument("--out", type=Path, default=Path("runs/unet"))
    sp.add_argument("--encoder", default="resnet34")
    sp.add_argument("--encoder-weights", default="imagenet", help="'none' for random init")
    sp.add_argument("--epochs", type=int, default=40)
    sp.add_argument("--batch-size", type=int, default=16)
    sp.add_argument("--num-workers", type=int, default=2)
    sp.add_argument("--device", default="cuda")
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("predict", help="score a U-Net checkpoint on val/test")
    manifest_args(sp)
    sp.add_argument("--checkpoint", type=Path, required=True)
    sp.add_argument("--results", type=Path, default=RESULTS)
    sp.add_argument("--splits", nargs="+", default=["val", "test"])
    sp.add_argument("--masks-dir", type=Path, help="also save predicted masks here")
    sp.add_argument("--device", default="cpu")
    sp.set_defaults(func=cmd_predict)

    sp = sub.add_parser("eval", help="summary.md / summary.json from results/per_chip_*.csv")
    sp.add_argument("--results", type=Path, default=RESULTS)
    sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("overlay", help="RGB | target | per-method overlay panels")
    manifest_args(sp)
    sp.add_argument("chip_ids", nargs="+")
    sp.add_argument("--results", type=Path, default=RESULTS)
    sp.add_argument("--unet-masks", type=Path)
    sp.add_argument("--out", type=Path, default=RESULTS / "overlays")
    sp.set_defaults(func=cmd_overlay)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s"
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
