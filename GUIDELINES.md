# Surface-water segmentation — H3

**Tier:** 3 · **Category:** Computer Vision · **Wave:** Wave 4 — Advanced differentiators

The project's own engineering guide. Keep it short and current.

## What this is

Satellite surface-water segmentation: a U-Net vs an NDWI-Otsu threshold on Sentinel-2, scored once
on held-out scenes, with a leakage audit. Data: S1S2-Water (CC BY 4.0), 65 scenes split 33/16/16
by the dataset's authors. The design and decision rules are frozen in `PREREG.md`.

## Stack

Python 3.12 · NumPy · PyTorch + segmentation-models-pytorch (U-Net, ResNet-34) · rasterio +
range-request zip reads for chip extraction · Pillow for overlays. Training on a Kaggle T4; chips,
baselines and evaluation run on CPU.

## Acceptance criteria

- [ ] `data/manifest.csv` committed; `segment audit` clean (no scene or tile in two splits)
- [ ] Baselines (NDWI-Otsu primary, MNDWI-Otsu, constant) fitted on train only, scored on val/test
- [ ] U-Net trained once on Kaggle with the frozen config; checkpoint chosen on val; test scored once
- [ ] `results/summary.md` built from committed per-chip CSVs; H1/H2 verdicts by the frozen rules
- [ ] Overlays of best and worst test scenes per method; failure modes written up
- [ ] Runs from a clean clone using only the documented steps
- [ ] Tests, lint and CI green; README results come from real runs

## Conventions

- `uv` for dependencies; `ruff check` + `ruff format`; `pytest` (`-m slow` trains a tiny CPU model).
- No secrets in the repo. The only env var is the optional `SEGMENT_GIT_COMMIT`, which the Kaggle
  notebook sets so `run_record.json` names the commit it trained.
- No chip, image or mask from the dataset is committed. Chips are rebuilt from Zenodo and checked
  by content hash (`segment verify`).

## Project-specific notes

- `data/manifest.csv` is the source of truth for chip membership and split; never hand-edit it.
- The test split is scored once per method. Anything changed after seeing a test number is an
  amendment in `PREREG.md`, reported next to the original.
- Zenodo serves about 6 MB/s per connection; a scene's Sentinel-2 image is ~1.7 GB compressed.
  `segment chips` streams one member per request (the remotezip default is ~10× slower).
