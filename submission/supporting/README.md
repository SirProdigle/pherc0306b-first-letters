# Supporting images

All of these show the claim image's surface, frame and box in reading orientation. They use its grey rule (median to 99.9th percentile of the shown area; dark = ink) unless the name says otherwise. The claim image is in the package root.

| folder | contents |
|---|---|
| `letter_grids/` | line 1 in 15 equal windows (no reading used), and line 2 the same way, with a strip marking where her reading of line 2 begins. The per-letter grid of line 1 is in the root. |
| `models/` | one full page per model or combination, each marked as a reading aid in its header; `00_all_models_claim_box.jpg` stacks the box from every page |
| `noise_reduction/` | how we reduced false ink in models A, B and C (gates, filters, retraining), and seven ways of combining them with TRACE v3B |
| `controls/` | `ctrl_panel_v7.jpg`: the claim image beside its depth-shuffle control, the decoy control and version 5; the decoy page; the box mirrored left to right; `views_side_by_side.jpg`: the box in five views |
| `views/` | the claim ink under the threshold rule over pale CT, A-hn alone, TRACE v3B alone, TRACE v3B and A-hn averaged, and the claim ink in a 7 cm² context |
| `reading/` | our papyrologist's tracings: 23 October (used to place the letter boxes and grid columns) and 16 October (versions 3 to 5) |

## The models

| model | trained on |
|---|---|
| A | v8-in, with half of each batch PHerc1447 team labels and half machine labels on other parts of PHerc0306B; three seeds, averaged |
| A-hn | model A plus hard negatives (blank papyrus marked as ink costs more); three seeds, averaged; in the claim image |
| B, B-hn | as one seed of A and A-hn, with PHerc0343 and PHerc0139 labels added to the human half |
| C | public human labels only (PHerc1447, PHerc0343, PHerc0139) |

Pages in `models/`:

- 01–05: the noise-reduced models: TRACE v3B × A-hn (the claim rule), A-hn × √Reader V3, C × √Reader V3, B-hn × √Reader V3, A-hn alone.
- 06–08: A and C × √Reader V3 or √TRACE v3B.
- 09–11, 16–17, 19–20: TRACE v3B × or + models A, B, B-hn and C.
- 12–15, 18, 21: models A (and each seed), B and C alone.
- 22–23: the public models TRACE v3B and v8-in alone.

"× √Reader V3" multiplies each pixel by the square root of Reader V3's ink probability, a softer form of the claim rule. Each page's `.json` records its maps and grey levels. Training records and the exclusion check are in `../method/`.
