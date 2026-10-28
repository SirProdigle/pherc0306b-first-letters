# v8Q: v8-in fine-tuned on PHerc1447 w058 and PHerc0343 (no PHerc0306B data)

The released v8-in base model (ResNet3D-50 + 2-D decoder, Hugging Face `YoussefMoNader/ink-8um-v8in`, `model.safetensors`, sha256 `3b94548d7f9ba9f99bc4b0ad8cafcec6d42585c06304e1bd11e47b5d0f88c4cd`), fine-tuned by our team.

- Weights: `v8Q_final.safetensors`, sha256 `4cb8e9217e5fba795cc0d7fc53afd998ad82c6c5d1250cdb9e63008f36a8f305`. They ship in the Docker image, sha256-checked at build time.
- Loading: the same class as v8-in, loaded with `load_state_dict(strict=True)` over the base.

## Training data

All of it is public text from other scrolls.

| source | labels | crop (px) | admissible 64 px tiles | ink share |
|---|---|---|---|---|
| PHerc1447 w058 (8.64 µm, 116 keV) | the Vesuvius team's ink labels (`YoussefMoNader/ink-8um-pherc1447-surfaces`) | y 2868:4368, x 671:4433 | 54,798 | 24.2 % |
| PHerc0343 (8.64 µm, 116 keV) | jeff-j-chen's stroke tracings | y 398:2348, x 875:5799 | 55,028 | 10.7 % |

The two sources are sampled 1:1 per batch. PHerc1447 w060 is held out for the no-harm test.

## Recipe

`train_v8.py`, the recipe of our earlier v8-in fine-tunes:

- all weights trained; BatchNorm running statistics frozen; bf16;
- AdamW at 1e-5 with a cosine decay to 1e-6 and a 5 % warm-up; weight decay 1e-6, not applied to norms or biases;
- effective batch 32 (micro-batch 1 × 32); 300 steps; the final checkpoint is used; seed 130697;
- loss: 0.5 × BCE (label smoothing 0.25) + 0.5 × Dice, masked to the labelled region;
- augmentations: flips, 90° rotations, depth jitter, coarse dropout and ±10 % contrast.

The loss fell from 0.714 at step 10 to about 0.67 at step 300 (`training_log.json`; `training_done.json` has the crops, the tile counts and the sha256 of the result and of its initialisation).

## Bars fixed in advance, and results

We wrote the bars down before any v8Q weight, map or score existed. Inference for every test: stride 21, Gaussian stitch divided by the weight sum, depth offset 0. Files: `judge_v8Q.json` and `bar_A_line.json`.

No-harm, on held-out PHerc1447 w060 (public labels; v8Q never saw w060; d0, no TTA). Bar: AP ≥ v8-in − 0.01. Pass.

| | AUC | AP | stroke width | skeleton recall | far hot |
|---|---|---|---|---|---|
| v8-in base | 0.804 | 0.697 | 0.379 mm | 0.674 | 0.129 |
| v8Q | 0.797 | 0.690 | 0.367 mm | 0.671 | 0.144 |

The AP difference is −0.0075.

Gain on the claim line, measured against our papyrologist's two tracings. Recipe: mirror4, on the touched surface, over the whole line.

- Bar: AP gain over v8-in ≥ +0.02 against both tracings, with a block-bootstrap 90 % CI above 0. Pass.
  - Tracing 1: AP 0.769 against 0.726 (+0.043, CI +0.015..+0.076).
  - Tracing 2: AP 0.805 against 0.767 (+0.039, CI +0.011..+0.074).
- Legibility gate against tracing 2, at the threshold that recovers 80 % of the traced skeleton. Pass.
  - Stroke width 0.42 mm against 0.38 mm; the bar is ≤ base + 0.05.
  - Skeleton recall 0.703 against 0.695.
  - Letters separated: 3 against 3.
  - Far marks: 33 against 41.

## Use in this package

v8Q appears in the letter grid and in three reading-aid pages, with Reader v2 or Reader V3, that the Docker `final` mode redraws and checks.

Code: `train_v8.py` (training), `v8lib.py` (inference and stitcher), `p4a_judge.py` (the bars). They are copied unchanged from our working tree, so their paths are those of our machine.
