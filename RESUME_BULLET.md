# Resume Bullets — Surface-water segmentation

Form: **action → technical specifics → measured outcome.** Numbers or it doesn't go on the resume.

---

## Bullets

- Pre-registered a U-Net (ResNet-34, smp) vs NDWI + Otsu comparison for Sentinel-2 surface-water
  mapping, scored once on 16 held-out scenes with a paired scene-clustered bootstrap: the U-Net
  reached 0.946 pooled IoU against the threshold's 0.960 (gap −0.014, 95% CI [−0.039, +0.012]),
  so the network did not earn its cost over a label-free index; reported as a null result.
- Built a reproducible chip pipeline over a 165 GB public dataset (S1S2-Water, CC BY 4.0):
  3,136 chips read as 512 px Cloud-Optimised GeoTIFF windows from a pinned mirror, a manifest of
  content hashes verified on a second machine (Kaggle) and against the authoritative Zenodo zip
  (195/195 file sizes, 130/130 label CRCs, one scene rebuilt chip-for-chip), and a leakage audit
  by scene and MGRS tile.
- Traced the two methods' failure modes from per-scene results and overlays: on the four scenes
  where the threshold was weakest the U-Net scored 0.882 vs 0.783 IoU by rejecting terrain
  shadow, but it missed rivers a few pixels wide and turbid coastal shallows (recall 0.953 vs
  0.980); exploratory, scene set fixed before training.

**Shorter form, one line:** Pre-registered U-Net vs NDWI-Otsu water segmentation on 16 held-out
Sentinel-2 scenes: the threshold won on pooled IoU (0.960 vs 0.946, scene-bootstrap CI of the gap
[−0.039, +0.012]); the U-Net fixed shadow false alarms but missed thin rivers.

## Which roles this supports

- [x] Data Scientist / ML
- [x] AI Engineer (LLM/NLP/CV)
- [x] Data Engineer
- [ ] Data Analyst / Python Developer

Data Engineer is checked for the chip pipeline: ranged reads, a resumable streamed download with
CRC checks, a content-hashed manifest verified across machines, and provenance against the source.

## Keywords this project earns

_Only list what you actually used and could be questioned on._

semantic segmentation · U-Net · segmentation-models-pytorch · PyTorch · mixed precision ·
Sentinel-2 · NDWI / MNDWI · Otsu thresholding · IoU / Dice · rasterio · Cloud-Optimised GeoTIFF ·
HTTP range requests · train/test leakage audit (scene, MGRS tile) · pre-registration ·
scene-clustered paired bootstrap · Kaggle GPU · GitHub Actions

## Before using a number

- **0.946 / 0.960** are pooled test IoU over 512 chips in 16 scenes. Don't say the U-Net "reached
  95% accuracy": IoU is not accuracy, and the baseline was higher.
- **0.882 vs 0.783** is exploratory (4 scenes, no interval). Say "on the four hardest scenes", and
  say that the set was fixed before training.
- The val IoU in `run_record.json` (0.9408) is the fp16 number from training; quote 0.941 from
  `results/summary.md` (fp32), which matches it to three decimals.
