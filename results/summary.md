# Results summary

| method | split | chips | scenes | IoU [95% scene-bootstrap CI] | Dice | precision | recall | macro IoU |
|---|---|---:|---:|---|---:|---:|---:|---|
| ndwi_otsu | val | 512 | 16 | 0.889 [0.765, 0.953] | 0.941 | 0.896 | 0.991 | 0.476 (39 NaN chips excluded) |
| ndwi_otsu | test | 512 | 16 | 0.960 [0.943, 0.972] | 0.980 | 0.979 | 0.980 | 0.607 (48 NaN chips excluded) |
| mndwi_otsu | val | 512 | 16 | 0.857 [0.661, 0.957] | 0.923 | 0.863 | 0.992 | 0.550 (66 NaN chips excluded) |
| mndwi_otsu | test | 512 | 16 | 0.966 [0.939, 0.978] | 0.983 | 0.977 | 0.989 | 0.631 (66 NaN chips excluded) |
| constant | val | 512 | 16 | 0.000 [0.000, 0.000] | 0.000 | n/a | 0.000 | 0.000 (141 NaN chips excluded) |
| constant | test | 512 | 16 | 0.000 [0.000, 0.000] | 0.000 | n/a | 0.000 | 0.000 (90 NaN chips excluded) |

CIs are 95% percentile scene-bootstrap intervals (2000 replicates, seed 0).

- Val-to-test drop (ndwi_otsu): -0.071 (val 0.889 -> test 0.960)

## Pre-registered verdicts

Thresholds: min_gap_iou = 0.050, max_drop_iou = 0.150.

- H1: n/a. Rule: gap = pooled test IoU(unet) - pooled test IoU(ndwi_otsu), with a paired 95% scene-bootstrap CI (scenes resampled with replacement, 2000 replicates, seed 0). supported if CI lower bound >= min_gap_iou; not supported if CI upper bound < min_gap_iou; inconclusive otherwise.
- H2: n/a. Rule: drop = pooled IoU(unet, val) - pooled IoU(unet, test). poor generalisation if drop > max_drop_iou; within bound otherwise.

## Per-scene IoU (test)

| scene | ndwi_otsu | mndwi_otsu | constant |
|---|---:|---:|---:|
| s023 | 0.783 (n=32) | 0.894 (n=32) | 0.000 (n=32) |
| s025 | 0.394 (n=32) | 0.153 (n=32) | 0.000 (n=32) |
| s029 | 0.969 (n=32) | 0.968 (n=32) | 0.000 (n=32) |
| s031 | 0.958 (n=32) | 0.962 (n=32) | 0.000 (n=32) |
| s033 | 0.972 (n=32) | 0.960 (n=32) | 0.000 (n=32) |
| s035 | 0.801 (n=32) | 0.812 (n=32) | 0.000 (n=32) |
| s047 | 0.759 (n=32) | 0.811 (n=32) | 0.000 (n=32) |
| s053 | 0.972 (n=32) | 0.985 (n=32) | 0.000 (n=32) |
| s057 | 0.957 (n=32) | 0.977 (n=32) | 0.000 (n=32) |
| s075 | 0.929 (n=32) | 0.965 (n=32) | 0.000 (n=32) |
| s077 | 0.916 (n=32) | 0.591 (n=32) | 0.000 (n=32) |
| s078 | 0.879 (n=32) | 0.921 (n=32) | 0.000 (n=32) |
| s080 | 0.970 (n=32) | 0.965 (n=32) | 0.000 (n=32) |
| s082 | 0.995 (n=32) | 0.997 (n=32) | 0.000 (n=32) |
| s088 | 0.981 (n=32) | 0.980 (n=32) | 0.000 (n=32) |
| s089 | 0.975 (n=32) | 0.988 (n=32) | 0.000 (n=32) |
