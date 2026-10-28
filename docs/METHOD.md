# Method

How the claim image was made, from the CT to the PNG. Each step names the code that ran (in [`submission/code/`](../submission/code/)) and the record that holds its numbers (in [`submission/method/`](../submission/method/)). [`submission/method/METHOD.md`](../submission/method/METHOD.md) maps each step to the records in the package.

- [1. Segment](#1-segment)
- [2. Render](#2-render)
- [3. Ink models and the claim rule](#3-ink-models-and-the-claim-rule)
- [4. Training A-hn](#4-training-a-hn)
- [5. The image](#5-the-image)
- [6. Validation](#6-validation)
- [7. Human choices](#7-human-choices)
- [8. What did not help](#8-what-did-not-help)

## 1. Segment

The text lies on one sheet of PHerc. 0306B, scan volume `20250521133212` (ESRF, 8.64 µm, 116 keV). In our spiral fit of the band z 8400–9900 it is winding 74, at roughly x 3100–5750, y 2800–4200, z 9170–9610 in level-0 voxels.

### How it was found

We ran the villa spiral-fit chain over one band of each eligible scroll on rented GPUs. The 40 windings were snapped to the organisers' m7 surface prediction, flattened with Lasagna, rendered, and inked with Hecate and Reader v2. Every window was then scored for text-like rows by a row-periodicity score, a small glyph CNN, and later a single-line finder. On PHerc. 0306B, the glyph CNN ranked a window on winding 74 first of 72 candidates, and the line finder ranked the line first of 191 on its band. The scripts are in [`discovery/`](../discovery/).

### The submitted surface

The claim surface is a VC3D GrowPatch grown from the seed (3559, 3224, 9348) on the m7 normal grids, with a step of 20 voxels and 140 generations. Near the right end of the line the patch drifted off the sheet, so a touch-up moves those nodes back to where two independent spiral-fit copies of the winding and a sheet tracker on m7 agree. It uses CT geometry only. We adopted it on two checks:
- the recto-fibre score rose from 0.33 to 0.55;
- at 95.5 % of moved nodes, two independent surfaces confirm the new position.

The surface was then re-flattened. Inside the claim box the median stretch is 0.995 along the line and 1.000 across it (`mesh/distortion_in_box.json`). [`submission/code/surface/build_surface.sh`](../submission/code/surface/build_surface.sh) rebuilds the mesh byte for byte. The mesh is in tifxyz format in [`submission/mesh/`](../submission/mesh/), with one grid cell per 20 voxels (0.1728 mm).

## 2. Render

`vc_render_tifxyz` samples 41 layers along the surface normal, 1 voxel apart, from 20 voxels below the surface to 20 above, so layer 20 is the surface. The raster is 4900 × 7780 px at 8.64 µm per pixel. Claim-frame coordinates in mm are u = −36 + x / 115.741 along the line and v = −21 + y / 115.741 across it.

## 3. Ink models and the claim rule

| model | what it is | how it runs |
|---|---|---|
| TRACE v3B | public community model (GitHub `Jashann/vesuvius-scrolling`, release v1.0) | villa inference, overlap 0.5, eight flips and rotations averaged, mean of layers 19 to 21 (`code/ink_v2.py`) |
| A-hn | our fine-tune of v8-in, three seeds averaged (section 4) | layers 8 to 31, 64 px tiles at stride 21, ±3σ Gaussian stitch divided by the weight sum, one pass (`code/ink_v8.py --nsig 3`) |

The claim map is the per-pixel product of the two models' ink probabilities (`code/fuse_v7.py`), so a pixel is dark only where both models find ink. The organisers used the same product rule for PHerc. 1667. On known text the product marks 6.1 % of blank papyrus as ink, against 8.5 % for TRACE v3B alone and 12.3 % for A-hn alone (section 6). TRACE v3B is also one of the four teachers whose maps gave A-hn its machine labels.

## 4. Training A-hn

A-hn is v8-in (HF `YoussefMoNader/ink-8um-v8in`) fine-tuned three times, with seeds 0, 1 and 2. The claim uses the mean of the three maps. The training code is in [`submission/code/training/`](../submission/code/training/), as it ran. The records are in [`submission/method/model/`](../submission/method/model/).

### Data

Half of every batch is human labels and half is machine labels:

| half | source | tiles |
|---|---|---|
| human labels | PHerc. 1447 surfaces w058 and w060, with the team's refined ink labels (HF `YoussefMoNader/ink-8um-pherc1447-surfaces`, revision `7e4d9187`). Same scan protocol as PHerc. 0306B. | 1,512 + 1,206 |
| machine labels | five other surfaces of PHerc. 0306B, 26.7 cm² in all (`pool_regions.json`): a GrowPatch around another candidate row (A6), lane windings w052, w053 and w059, and w109 of the next band up | 23,331, of which 1,405 hold ink |

The machine labels come from four public models: v8-in, Reader V3, TRACE v3B and Reader v2 (`self/code/teachers.py`, `labels.py`). Each model's map becomes a z-score, and the four are averaged. Then:
- ink is a mean of 2.5 or more, kept only in blobs of 0.3 to 6 mm²;
- background is a mean of 0.3 or less, at least 20 px from ink;
- everything else is left out of the loss.

That gives 66.7 mm² of ink in 72 blobs and 1958.8 mm² of background (`pseudo_labels_stats.json`).

### Exclusion zone

The zone holds 2.39 million 3-D points: every surface point shown in any of our images of this line, plus 2 mm on the surface (`exclusion/`). A training pixel must lie at least 2.0 mm from the zone in 3-D, plus 0.2 mm for the render slab and 0.7 mm for the tile pad. Because the distance is in 3-D, the rule covers every winding. CT inside the 2.2 mm buffer was set to zero before any teacher or student read it.

All 23,331 PHerc. 0306B tiles were rebuilt and checked: centre and corners at the pad, a 5 × 5 grid at pad 0, and every 16th labelled pixel. All pass, and the nearest tile pixel is 2.925 mm from the zone (`exclusion_audit.json`). A6, the largest pool surface, reaches the claimed sheet at its far edge, and the rule drops its tiles there. The Docker `exclusion` mode repeats the check.

### Recipe

| setting | value |
|---|---|
| steps | 600 of 16 tiles: 8 PHerc. 1447 and 8 PHerc. 0306B (micro-batch 4, accumulation 4) |
| loss | 0.5 Dice + 0.5 BCE on labelled pixels; BCE targets smoothed to 0.25 and 0.75; PHerc. 0306B half weighted 0.5 |
| hard negatives | each labelled background pixel's BCE weight is 1 + 4 × the model's own ink probability there |
| optimiser | AdamW, learning rate 2e-5, 5 % warm-up, cosine decay to 2e-6, weight decay 1e-6 |
| augmentation | the organisers' PHerc. 1447 recipe: rotation, scale ±10 %, shift ±15 %, blur, dropout holes (full settings in each checkpoint's `meta.json`) |
| input | render layers 8 to 31, in stored order, 64 px tiles |
| saved stages | step 300 and step 600 (final) |
| compute | one rented GPU per run, about 12 minutes, peak 19 GB |
| code | `training/x_push3/m306/denoise/cloud_code/train_cloud_hn.py`; configs `m306/denoise/configs/hn4_s{0,1,2}.json` |

The decoy control is the same training with every PHerc. 0306B machine label moved 3.5 × 3.0 mm. Model A, a comparison, is the same training without hard negatives. Weights and checksums for all of them are listed in [MODELS_AND_DATA.md](MODELS_AND_DATA.md).

## 5. The image

The claim image is `PHerc0306B_20250521133212_w074own_touchup_V5Rc.png`, drawn by [`submission/code/make_claim_image_v2.py`](../submission/code/make_claim_image_v2.py) from `config_v7.json`. Its `.json` records every input checksum and level.

| element | rule |
|---|---|
| tone | linear from the median (page colour) to the 99.9th percentile (full ink) of the fused map over the shown area, u −30..+30 × v −20..+15 mm; no threshold; ink on a plain page; off-sheet areas blue-grey |
| orientation | flipped top to bottom into reading orientation |
| box | u −15..+23 × v −6.84..+2.72 mm: 38.0 × 9.56 mm, 3.633 cm² flat and 3.603 cm² on the mesh (`mesh/box_area.json`) |
| rows | peaks of the ink profile across the box, smoothed 0.6 mm and at least 3 mm apart, marked by ticks in the margins |
| scale | 1 cm bar; letter boxes are 1.2 to 3.8 mm tall (`method/checks/letter_dimensions.md`) |

## 6. Validation

| check | what was done | result | record |
|---|---|---|---|
| depth shuffle | render layers 8 to 31 shuffled around the box; every model run again | box pixels darker than half tone: 8.6 % → 0.00 % | `checks/depth_shuffle_variants_v7_trace_x_Ahn_product.json` |
| blank control R | a strip of the same sheet (u +25..+30, v −20..+3 mm), fixed in advance | 0.42 % darker than half tone | `checks/blank_control_views_v7_trace_x_Ahn_product.json` |
| decoy | TRACE v3B × the decoy model, same levels | blurrier strokes over about as much of the box (8.5 %), since TRACE v3B sets most shapes; blank R 0.21 % | `checks/ctrl_panel_v7.json` |
| exclusion | every training tile against the zone, in 3-D | all allowed; nearest 2.925 mm | `model/exclusion_audit.json` |
| Docker | `final`, `smoke`, `exclusion` | claim image redrawn byte for byte (20 of 20 views); A-hn re-run matches (r 0.99999); all tiles allowed | `checks/docker_tests/` |

Blank control R is the cleanest strip we could fix in advance. Its CT contrast is 0.48 of the line's, and it holds faint marks at the level of two text rows.

### Known text

PHerc. 0841 (volume `20260220213127`, 9.37 µm, 113 keV, organiser labels v2) is in no model's training data. TRACE v3B's authors did use it to choose their checkpoint, so rows with TRACE v3B are slightly optimistic, by about 0.01 AUC on their estimate. False ink is the share of far blank papyrus above the threshold that keeps 80 % of the labelled stroke centre-lines. Faint recall is the recall of the faintest third of strokes at that threshold (`checks/known_text/known_text_product_rule.json`).

| model | AUC | AP | false ink | faint recall |
|---|---|---|---|---|
| TRACE v3B × A-hn (claim rule) | 0.874 | 0.731 | 6.1 % | 0.778 |
| TRACE v3B × model A (no hard negatives) | 0.874 | 0.729 | 6.1 % | 0.779 |
| TRACE v3B × v8-in | 0.871 | 0.727 | 5.9 % | 0.757 |
| TRACE v3B | 0.870 | 0.713 | 8.5 % | 0.741 |
| A-hn | 0.845 | 0.699 | 12.3 % | 0.777 |
| v8-in | 0.840 | 0.674 | 12.1 % | 0.782 |

The product of two models gives most of the gain here, and the fine-tune adds little on known text. On PHerc. 0306B the claim image is cleaner on blank control R (0.42 %) than TRACE v3B + v8-in under the same tone rule (0.57 %). PHerc. 0009B is also scored in the records, but TRACE v3B, Reader v2 and Reader V3 trained on it, so it is a reference only.

## 7. Human choices

The team leader made three choices by hand, looking at the claimed region:
- adopting the surface touch-up on the two checks in section 1. The fibre-grid rule we had registered in advance rejected it, and we then found that rule does not track the written face. Before deciding, we had looked at ink on the alternative sheet at the right end;
- drawing the box, with its right edge just past the last letter of line 1;
- choosing A-hn, by comparing the claim region side by side with model A and, blind, with TRACE v3B + v8-in, knowing the papyrologist's reading.

The tone rule was written down on 15 October, before our papyrologist saw an image made with it; an exploratory grey view of the line existed before then. Blank control R and the checks were fixed before the images they judge.

Our papyrologist's tracings were used:
- as a no-harm check when we adopted the surface touch-up (ink agreement could not fall by more than 0.01 AP), and to reject a rival surface correction;
- to gate v8Q, our fine-tune shown only in supporting views and the letter grid (the grid labels it as a public model);
- to choose Reader V3 + TRACE v3B for an earlier version of the image;
- to measure the letter sizes;
- to place the letter-grid columns. Each edge starts at the narrow point between two of her letters and moves to the nearest gap in the ink, except that Ρ|Ε and Ε|Τ stay at her positions and Ο|Α and Α|Ι were set mid-gap by eye.

`code/models/v8Q/train_v8.py` also has "finder" arms that train on her tracing. v8Q itself trained only on PHerc. 1447 w058 and PHerc. 0343 (`training_done.json`), and no finder model is in any image or in A-hn's chain.

## 8. What did not help

We tried these on this line, and none made the letters more legible:

- Upsampling the render before inference gave thinner strokes but lost ink.
- Deconvolving the ink map invented strokes.
- Fine-tuning Reader V3 on thinned (centre-line) labels made strokes dimmer, not thinner.
- Other model variants all drew the same letters, with the end of line 1 still blobby. These included fine-tunes of v8-in on other scrolls, self-training on this scroll, Reader-family students of our models, and matched-voxel-scale training. The limit there is the scan.
- Snapping the surface to CT density peaks gave no improvement.
- Frozen features from dinovol, a self-supervised CT model, scored at chance for ink.

Three measurements that apply to screening in general:
- Reader V3 keeps a 0.5 mm gap between strokes open only about a third of the time. The gaps between letters here are 0.2 to 0.5 mm, so neighbouring letters merge.
- The share of marks lying over dark gaps in the CT does not separate text from artefacts (AUC 0.35 on 23 text and 22 artefact bands).
- Disagreement between models, and spread under test-time augmentation, are weak hallucination controls (AUC 0.50 to 0.66).
