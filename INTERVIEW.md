# Interview Prep — Surface-water segmentation

**Five questions, five answers.** An unanswered question means this project is not shipped.

---

### Q1. Walk me through the architecture in 90 seconds.

**A:** The data is S1S2-Water: 65 Sentinel-2 scenes with hand-labelled water and cloud masks,
split by the authors into 33 train, 16 val and 16 test scenes. It is about 165 GB of zips on
Zenodo, which served me about 0.5 MB/s, so I read 512-pixel windows from a pinned mirror of the
same files instead. A scene's chips are chosen from its masks alone: a fixed grid, at least 80%
valid pixels, a seed per scene. Each chip's array content is hashed into `data/manifest.csv`,
which is the source of truth. Kaggle rebuilt all 3,136 chips and matched every hash, and a
provenance check matched the mirror to the Zenodo zip.

Two arms read the same chips:

- **Baseline:** NDWI with an Otsu threshold fitted on training pixels only, then frozen.
- **Model:** a U-Net with a ResNet-34 encoder on RGB+NIR, trained once on a Kaggle T4 with the
  checkpoint picked on val.

Both write per-chip confusion counts. `segment eval` turns those into pooled and macro IoU, a
scene-clustered paired bootstrap, and the verdicts the pre-registration defines.

### Q2. Why bootstrap over scenes rather than chips or pixels?

**A:** Because chips from one scene aren't independent. They share the sun angle, the
atmosphere, the season, the water bodies, often the same lake cut into pieces. If I resample
512 chips as if they were independent, the interval reflects 512 samples when the real
information is closer to 16. Resampling whole scenes and recomputing pooled IoU on each
replicate gives the width that matches the question, "how would this do on a new place?". For
the U-Net-minus-threshold gap I use one scene draw per replicate for both methods (a paired
bootstrap), so scene difficulty cancels out. The interval is wide, [−0.039, +0.012], and
that width is the honest answer with 16 test scenes.

### Q3. What's the weakest part of this, and what would break first?

**A:** Three things.

- **The test set is small in the unit that matters.** There are 16 scenes, so per-scene findings,
  like the shadow pattern, are leads, not results.
- **The labels hide clouds from both methods.** The hand-made valid mask removes cloud and
  shadow, and nobody gives you that mask in production. A real system needs a cloud model in
  front, and its errors would dominate.
- **H1's margin was unattainable.** I set "the U-Net wins by 0.05 IoU" before knowing the
  threshold would score 0.96. I kept it frozen and said so. Next time I would run a pilot on the
  train split to size the margin before freezing it.

At scale, data access breaks first: you would read COGs from a bucket next to the compute.

### Q4. How do you know it works? What did you measure, and against what?

**A:** The baseline is the method a practitioner would try first: NDWI thresholded by Otsu. It
needs no labels, and it is fitted on train only and frozen. A constant "always land" baseline
checks that the metric isn't broken (IoU 0), and MNDWI is a second index for context.

The test split was scored once per method, with rules committed before any data existed. The
U-Net got 0.946 pooled IoU against the threshold's 0.960, a gap of −0.014 [−0.039, +0.012]. So
it does not beat the threshold, and H1 is not supported. Its val-to-test drop was about zero,
so it isn't overfitting to the validation scenes.

The pipeline itself is checked in several ways:

- hash verification of chips across two machines;
- a leakage audit by scene and by MGRS tile;
- one scene rebuilt from the original zip, chip for chip;
- 166 tests, including a tiny CPU training run in CI and a check that the summary regenerates
  byte-identically from the committed CSVs.

### Q5. Your network lost to a two-line threshold. Why should anyone hire you for ML?

**A:** Because I measured it instead of assuming it, and I found out where each one fails. Pooled
IoU is dominated by big, clear lakes, where an index is close to perfect, so there was no room
for the network to win there.

On the four scenes where the threshold was weakest, which I wrote down before training, the
U-Net scored 0.882 against 0.783. It does that by ignoring terrain shadow that NDWI calls
water. But it misses rivers a few pixels wide and bright, turbid coastal water, where reading
the spectrum per pixel is exactly right. That is post hoc and only four scenes, so I report it
as exploratory. It does point to a concrete next experiment: use the threshold for recall and
let the network veto shadow.

Knowing when a cheap baseline is good enough is a large part of ML engineering judgement.

---

## 30-second pitch

Water maps from Sentinel-2 can come from a free spectral index or from a trained network. I
pre-registered a comparison on 16 scenes held out by the dataset's authors, with a
scene-clustered bootstrap, and built a hash-verified chip pipeline over a 165 GB dataset to run
it. The threshold won: 0.960 IoU against the U-Net's 0.946, and the gap's interval straddles
zero. The network's real value showed up in specific places: it rejects mountain shadow, but it
misses thin rivers. That is what I'd build on next.
