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

### 2026-10-25 — baseline results and what they mean for H1; local CI

- **Baselines on test (pooled IoU, 95% scene CI):** NDWI-Otsu 0.960 [0.943, 0.972], MNDWI-Otsu
  0.966 [0.939, 0.978], constant 0. Val is harder than test for NDWI-Otsu (0.889; drop -0.071).
- **H1 is now arithmetically out of reach.** "Supported" needs the paired-gap CI lower bound
  >= 0.05, but the baseline already sits at 0.96, so the gap cannot exceed about 0.04. The margin
  was set before any number existed; it is not changed now (that would be fitting the rule to the
  data). The verdict will be "not supported" or "inconclusive", and the write-up says plainly
  that the pre-registered margin was unattainable on clear-sky surface water.
- **Where the threshold fails** (pooled IoU hides it: macro IoU is 0.607): s025 (0.394, boreal
  forest, little water), s047 (0.759, coastal wetland in Colombia: flooded vegetation and marsh
  are labelled water, and NDWI sees vegetation), s023 and s035 (~0.78-0.80). These scenes and the
  macro IoU are where a U-Net can add value; they are reported as exploratory, not as H1.
- **Local CI replay** (the workflow's steps, in order) on a fresh clone of `d0b106c` with a fresh
  install (`UV_NO_CACHE=1`): ruff clean, 165 passed, coverage 82%, slow CPU training test passed,
  `segment audit` clean, `segment eval` regenerates `results/` byte-identically. Differences from
  the hosted runner: Windows 11, `--basetemp` in scratch, `-p no:cacheprovider`. Hosted CI not
  run (no GitHub repo yet).

### 2026-10-25 — published, Kaggle run, results

- **Published** on the owner's yes: `Prithv122/segmentation`. Pre-push gate: no imagery, labels,
  arrays or weights tracked, no key patterns, one author.
- **Hosted CI failed on `05f7e16` with a real bug the Windows replay could not see.** On Linux,
  uv took torchvision from PyPI (a CUDA build) next to CPU torch, giving
  `operator torchvision::nms does not exist`. Fix `3e7dd8f` routes torchvision through the CPU
  index. It was replayed in a `python:3.12-bookworm` container before the push, and hosted CI run
  40144137619 then passed. Lesson: replay CI on the runner's OS, not just on the dev machine.
- **Kaggle kernel v1** (private, T4, `REPO_REF = 3e7dd8f` set only in the submitted copy): 3,136
  chips rebuilt from the mirror and every content hash matched the committed manifest. Training
  used the frozen config: best val epoch 4, early stop at epoch 12, about 25 s per epoch. Raw
  output is kept in `_extra-files/45-segmentation/kaggle_v1/` (outside the repo). The per-chip CSVs,
  run record, predicted masks and kernel log were committed as produced (`c4118a7`) before
  `segment eval` ran.
- **Results.** Test pooled IoU: U-Net 0.946 [0.920, 0.969], NDWI-Otsu 0.960 [0.943, 0.972]. Gap
  −0.014 [−0.039, +0.012] → **H1 not supported** (it could not have been). U-Net val→test drop
  −0.005 → **H2 within bound**. Macro IoU 0.600 vs 0.607.
- **Exploratory, on the weak-scene set fixed at `05f7e16`:** U-Net 0.882 vs NDWI-Otsu 0.783 on
  s025/s047/s023/s035, but 0.948 vs 0.967 on the other 12; better in 5 scenes, worse in 7.
  Overlays (local only, as they contain dataset imagery):
  - s035: the threshold calls mountain shadow water; the U-Net doesn't.
  - s025: the U-Net misses a river a few pixels wide.
  - s031: the U-Net misses bright turbid coastal shallows.
  - Overall the U-Net trades recall for precision (0.953 / 0.992 vs 0.980 / 0.979).
- **Not done on purpose:** no threshold tuning on the sigmoid, no second training run, no test
  rescoring. Any of these would be an amendment. The obvious next experiment is a hybrid
  (threshold for recall, network vetoes shadow), which needs its own pre-registration.

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

- [ ] Would a hybrid (NDWI-Otsu ∪ U-Net, minus U-Net-rejected shadow) beat both? Needs a new pre-registration and fresh held-out scenes.

- [x] Does the mirror match Zenodo? Yes: sizes, mask CRCs, sampled image CRCs, and one scene rebuilt chip-for-chip
