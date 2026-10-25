# Pre-registration — surface-water segmentation, U-Net vs NDWI-Otsu

Frozen before any real chip was scored by any method. Changes after this commit are logged as
numbered amendments at the bottom, with the date and the reason, and never rewrite the text above.

## Question

Does a U-Net trained on Sentinel-2 chips beat the classical method a remote-sensing practitioner
tries first (NDWI thresholded by Otsu) at mapping surface water in **scenes it never saw**, and by
how much does its IoU fall between the validation scenes and the held-out test scenes?

S1S2-Water holds ordinary surface-water scenes (lakes, rivers, reservoirs, coast), not flood
events. Nothing here supports a claim about flood mapping.

## Data

- **S1S2-Water v1.0.1** (Wieland et al., 2024, IEEE JSTARS, doi:10.1109/JSTARS.2023.3333969),
  Zenodo record 11278238, **CC BY 4.0**. 65 Sentinel-2 L1C scenes (100 km tiles), each with a
  hand-labelled water mask and a hand-labelled valid-pixel mask (cloud, cloud shadow, nodata).
- **Split: the dataset's own, by scene: 33 train, 16 val, 16 test.** No Sentinel-2 tile appears in
  two splits (checked in `data/scenes.csv` and by `check_splits` on the manifest).
- Nothing from the dataset is committed: only `data/scenes.csv` (scene metadata) and
  `data/manifest.csv` (chip positions and content hashes). Chips are rebuilt from Zenodo.

## Chips (`segment/chips.py`)

- Bands kept: blue, green, red, NIR, SWIR1 (reflectance × 10000, uint16).
- Mask: water 1, land 0; **ignored (255)** where the valid mask is 0 or every band is 0.
  Ignored pixels are excluded from training loss, threshold fitting and every metric.
- Grid: non-overlapping 512 × 512 chips anchored at pixel (0, 0); a chip is eligible when at least
  80% of its pixels are valid.
- Per scene, seed `45 + scene number`: **train** 64 chips, up to half drawn from chips with at
  least 1% water and the rest uniformly from the remainder (a training choice, so the model sees
  water in dry scenes); **val and test** 32 chips drawn uniformly, so the sample represents the
  scene. Scenes with fewer eligible chips contribute all of them.

## Methods

| Method | Role | Definition |
|---|---|---|
| `unet` | treatment | smp U-Net, ResNet-34 encoder, ImageNet weights, input blue/green/red/NIR ÷ 10000 clipped to [0, 1], one logit; water where sigmoid > 0.5 (threshold fixed, never tuned) |
| `ndwi_otsu` | **primary baseline** | NDWI = (green − NIR)/(green + NIR); threshold = Otsu (256 bins) over the NDWI of valid pixels of **train chips only**, ≤ 20,000 pixels sampled per chip with seed 0; water where NDWI > threshold |
| `mndwi_otsu` | context only | same procedure on MNDWI = (green − SWIR1)/(green + SWIR1) |
| `constant` | context only | the train set's majority class among valid pixels |

**U-Net training (frozen):** AdamW, lr 3e-4, weight decay 1e-4, cosine schedule, up to 40 epochs,
batch 16, random 256 px crops with flips and 90° rotations, loss BCE + soft Dice over non-ignored
pixels, mixed precision on GPU, seed 0. **Checkpoint = best pooled IoU on the val split;** early
stop after 8 epochs without improvement. One training run. The test split is never loaded during
training (asserted in code).

## Metrics

- **Primary: pooled IoU** of the water class on the test split (tp / (tp + fp + fn) summed over all
  test chips).
- Also reported: Dice, precision, recall, macro IoU over chips (chips with no water in target and
  prediction excluded and counted), per-scene IoU, the same on val.
- **Uncertainty: scene-clustered bootstrap.** Whole scenes are resampled with replacement and
  pooled IoU is recomputed per replicate; 2,000 replicates, seed 0, 95% percentile interval.
  The U-Net − NDWI-Otsu gap uses a **paired** scene bootstrap (one draw of scenes per replicate,
  shared by both methods).

## Hypotheses and decision rules

**H1 — U-Net beats NDWI-Otsu by a meaningful margin.** Effect = pooled IoU(unet, test) −
pooled IoU(ndwi_otsu, test). Minimum meaningful improvement **+0.05 IoU**.

- *supported*: paired-bootstrap 95% CI lower bound ≥ 0.05
- *not supported*: CI upper bound < 0.05
- *inconclusive*: otherwise

**H2 — generalisation (diagnostic, not a significance test).** Drop = pooled IoU(unet, val) −
pooled IoU(unet, test). A drop **> 0.15** is reported as poor generalisation.

The rules are in `segment/evaluate.py` and the thresholds in `results/prereg.json`.

**A null result is a result.** If NDWI-Otsu matches or beats the U-Net, it is reported as such, with
the per-scene breakdown and overlays showing where each method fails (cloud edges, shadow,
turbid or narrow water, bright urban surfaces).

## Test discipline and order of operations

1. Commit this file, `results/prereg.json`, `data/scenes.csv` and the code.
2. Build chips, commit `data/manifest.csv`; `check_splits` must return no violations.
3. Fit NDWI-Otsu, MNDWI-Otsu and constant on train; score them on val and test.
4. Train the U-Net (Kaggle GPU, notebook pinned to a commit); the notebook scores val and test
   once with the val-chosen checkpoint.
5. `segment eval` builds `results/summary.md` from the committed per-chip CSVs.

Baseline test scores (step 3) are seen before the U-Net is trained. The U-Net's configuration is
frozen above and is not changed because of them; any change would be an amendment and would be
reported next to the original result.

## Amendments

- **A0 (2026-10-25, before any data was scored).** Dataset changed from Sen1Floods11 to
  S1S2-Water. Sen1Floods11's own STAC catalog lists its licence as `proprietary` and its repository
  has no licence file, which failed the project's licence gate; S1S2-Water is CC BY 4.0 on the
  authors' Zenodo record. The framing changes from flood events to surface water and from a
  held-out event to held-out scenes; the model, inputs, baseline and decision rules do not.
- **A1 (decided 2026-10-25 before any chip was scored; written down after the results).** Chip
  *source*: image windows are read from a pinned unzipped mirror of the same v1.0.1 files
  (Hugging Face `idomogalla/s1s2_water` at revision `e0551de`) instead of the Zenodo zips, which
  sustained about 0.5 MB/s per connection. The chip rules above (grid, eligibility, sampling,
  seeds, masks) are unchanged, and "Chips are rebuilt from Zenodo" in the Data section should read
  "from Zenodo or the mirror". Equivalence was checked before scoring: 195/195 file sizes, 130/130
  mask CRCs and 3/3 full image CRCs match the Zenodo zip directory, and one scene rebuilt from the
  Zenodo zip gave byte-identical chip records (`results/provenance.csv`, commit `e34e2d9`). This
  amendment should have been written at the time of the switch; it changes no rule and no number.
