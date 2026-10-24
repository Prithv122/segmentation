# Build Notes — surface-water segmentation

Working notes: what broke, what was tried, why X over Y.

---

## Log

### 2026-10-25 — dataset, licence gate, chip pipeline

- **Licence gate failed for the planned dataset.** Sen1Floods11's own STAC catalog
  (`gs://sen1floods11/v1.1/catalog/*/collection.json`) says `"license": "proprietary"` and its
  repository has no licence file. A HydroShare re-host labels a derivative CC BY, but its owner is
  not one of the dataset's authors, so it settles nothing. The design made licence confirmation a
  hard gate for a public repo, so the dataset changed before any data was scored (PREREG A0).
- **S1S2-Water instead.** CC BY 4.0 on the authors' Zenodo record 11278238 (checked through the
  Zenodo API, `metadata.license.id == "cc-by-4.0"`). It keeps the experiment intact: Sentinel-2
  with green and NIR (so NDWI works), hand-labelled water and valid-pixel masks, and an
  author-defined split by scene (33/16/16). All 65 scenes report `flood: false`: this is surface
  water, not flood mapping, and the README says so.
- **Leakage audit, scene level.** Every scene is a different MGRS tile (no tile in two splits);
  the chip manifest repeats the check per chip (`segment audit`).
- **Download speed shaped the pipeline.**
  - The data is six zip parts (20-33 GB each, deflate members), about 165 GB in total; one scene's
    Sentinel-2 image is about 1.7 GB compressed.
  - `remotezip.extract` ran at under 0.5 MB/s, because it makes many small range requests. A single
    streamed ranged GET with inline inflate and a CRC check fixed that (`_stream_member`).
  - Zenodo itself then throttled to about 0.45 MB/s per connection (6 MB/s in a first burst). At
    that rate the whole dataset is many hours to a day.
  - The unofficial Hugging Face mirror holds the same v1.0.1 files unzipped as Cloud-Optimised
    GeoTIFFs with 512 px tiles. Those tiles are aligned with the chip grid, so one chip is one ranged
    read. It ran at about 11.7 MB/s; one train scene (64 chips) took 149 s.
  - The mirror is pinned to a revision. Zenodo stays the authority: `segment provenance` checks the
    mirror's file sizes and the label masks' CRC32 against the Zenodo zip directory. One scene was
    rebuilt from the Zenodo zip and compared chip for chip.
- **Chip selection uses the masks only.** The "every band is 0 → ignore" rule moved from scene
  level to chip level, so the windowed path (which never loads a whole image) and the whole-scene
  path pick identical chips.
- **Chip hash = content hash.** The manifest's `sha256` hashes the arrays, not the `.npz` bytes,
  because compressed bytes depend on the zlib build and a Kaggle rebuild must verify against the
  local manifest.
- **Bootstrap unit = scene** (owner's pre-registration review): chips from one scene are
  correlated, so whole scenes are resampled and pooled IoU is recomputed per replicate. With
  16 test scenes the interval will be wide; that is the honest width.

### 2026-10-25 — chips built, provenance, baselines, fresh-eyes review

- **Chips:** 3,136 (2,112 / 512 / 512 over 33 / 16 / 16 scenes), every scene had its full
  quota; `segment audit` clean; `segment verify` all match. Built from the mirror with 8 workers
  in about 50 minutes, 4.6 GB.
- **Mirror == Zenodo, shown three ways.** (1) `segment provenance`: 195/195 file sizes, 130/130
  mask CRCs, 3/3 full image CRCs (one scene per split) match the Zenodo zip directory.
  (2) Scene s071 rebuilt from the Zenodo zip itself gives a `records/s071.csv` byte-identical to the
  mirror build (same chips, same content hashes). (3) The Zenodo rebuild hit a real dropped
  connection and resumed correctly, which is why `_stream_member` resumes and counts only
  failures that make no progress.
- **Baselines fitted on train only:** NDWI-Otsu threshold 0.0064 (near the textbook 0), MNDWI-Otsu
  0.133, constant = land. Val and test scored once, CSVs committed before any summary was read.
- **Fresh-eyes review** (clean context, code and PREREG only) found no bug that changes a number.
  Acted on before looking at any test summary:
  - `segment eval` now refuses a duplicated per-chip CSV (e.g. a "(1)" re-download) and refuses to
    compare U-Net and NDWI-Otsu if they scored different test chips.
  - `segment baseline` verifies every chip hash before fitting.
  - Kaggle install: rasterio no longer pinned to an exact version (Kaggle's Python may lack a
    wheel for it) and `--ignore-requires-python` for the editable install. The source parses and
    compiles on Python 3.11.
- **Two reporting cautions from the review.** `best_val_iou` in `run_record.json` is computed under
  fp16 autocast during training; the reported val IoU is the fp32 one from `segment predict`, so
  never quote the run-record number. The val IoU belongs to the checkpoint val chose, so it runs
  high and H2's drop is biased towards "poor generalisation"; PREREG calls H2 diagnostic, and the
  README must say so.

---

## Rejected approaches

| Approach | Why rejected |
|---|---|
| Sen1Floods11 | Licence listed as proprietary by its own catalog; failed the gate |
| SpaceNet 2 buildings (the written fallback) | Licence fine, but buildings have no classical index baseline, so the question would change |
| WorldFloods | CC BY-NC, and about 76-310 GB |
| Download the full Zenodo parts | 165 GB at ~0.5 MB/s per connection |
| `remotezip.extract` per member | Many small range requests, under 0.5 MB/s |
| Chip-level bootstrap | Treats correlated chips from one scene as independent; CI too narrow |

## Open questions

- [x] Does the mirror match Zenodo? Yes: sizes, mask CRCs, sampled image CRCs, and one scene rebuilt chip-for-chip
