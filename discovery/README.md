# discovery

The scripts that found the line: the spiral-fit screening lane on PHerc. 0306B, the window triage that ranked it, and the follow-up checks. [`docs/METHOD.md`](../docs/METHOD.md#1-segment) summarises what they did. Everything after the line was found (surface, ink models, claim image) is in [`submission/`](../submission/).

These are the files as they ran. They hard-code our paths (`/home/liam/projects/vesuvius-work`, `/home/liam/projects/villa`) and need edits to run elsewhere. The folders mirror our working repository, and `MANIFEST.tsv` gives each file's original path and sha256. Most are the current working copies, a few commits newer than the runs (failover, retries and similar robustness fixes). `x_vast_plan/lane_box.sh` is the exact version that ran on the lane.

There is no data here. The only weights are the 2.6 MB glyph CNN `x_glyph/models/gA.pt` (CC BY-NC 4.0), trained on synthetic glyphs, public known text and text-free windows from our screening.

## Pipeline

| stage | what it does | files |
|---|---|---|
| band | picks the clearest 1,500-slice band from the Scan-Quality-Map atlas: z 8400–9900 for PHerc. 0306B | `x_vast_plan/sqm_bands.py`, `sqm_atlas_PHerc0306B_only.json`, `lanes_PHerc0306B_only.json` |
| axis | builds the scroll axis and smooths it over 75 slices | `x_recipe_erwin/tools/build_umb.py`, `smooth_umb.py`, `x_vast_plan/state/umb_PHerc0306B_s75.json` |
| rented box | rents a GPU box, pushes the code, runs the lane and pulls the results | `x_vast_plan/fleet.py`, `fl/remote/` |
| fit | villa `fit_spiral.py`, anticlockwise, 30,000 steps | `x_vast_plan/lane_box.sh` stage `03_fit` |
| snap, flatten, render, ink | snaps 40 windings (w035–w074) to the m7 prediction, flattens them with Lasagna, renders 31 layers, inks both faces with Hecate and Reader v2 | `lane_box.sh` stage `04_post`, `x_vast_plan/ink_lane.py`, `x_research2/speed/fastpipe.py`, `fastink.py` |
| triage | row-periodicity score, gates and ranking of every window | `fl/remote/spiral/window_score.py`, `fl/scripts/tri_rank.py`, `gap_gate.py`, `stroke_feats.py` |
| review | glyph CNN score per window, reading face, review cards; later the single-line finder | `x_vast_plan/review_lane.py`, `fl/scripts/glyph_win.py`, `x_glyph/`, `fl/scripts/line_find.py`, `x_vast_plan/cards_from_tri.py`, `CANDIDATES/_tools/card.py` |
| own-sheet patch | finds the line's own sheet along the normal and grows a GrowPatch from seed (3559, 3224, 9348) with step 20 and 140 generations | `fl/scripts/neighbour_seeds.py`, `x_focus306/grow_render.sh`, `fl/scripts/sqm_grow.py`, `fl/sqm/PHerc0306B_f306_w074own/` |
| max-quality read | Hecate (stride 16), Reader v2 and TRACE v3B, 8× test-time augmentation, depths −1 to +1, with a PHerc. 0343 positive control | `x_deepread/`, `x_zoom/` |
| checks | depth sweep, depth shuffle, lane against own-sheet patch, neighbouring sheets, a fourth model | `x_followup/A_depth/`, `x_submit/depth_shuffle.py`, `x_followup/own_compare.py`, `x_submit/v8loo.py` |

The own-sheet patch is surface `A` in [`submission/code/surface/p1geom.py`](../submission/code/surface/p1geom.py), which adds the touch-up that became the submitted mesh.

The screening scripts used across all eligible scrolls before the lane (`fl/scripts/screen.py`, `seeds.py`, `spectral_rows.py`, `fiber_align.py`, `deep.py` and others) are included as well, because the lane reuses them.

## Not included

These ran too but also cover other scrolls, so they are left out: `x_vast_plan/make_lanes.py` (one band per eligible scroll), `x_focus/focus_region.py` and `x_focus/make_card.py` with the gap-pocket helper `x_focus0813/permark.py` (the focus battery), and the later working copy of `lane_box.sh`.

Four files were cut down to PHerc. 0306B entries only; their entries are otherwise unchanged:
- `lanes_PHerc0306B_only.json`;
- `sqm_atlas_PHerc0306B_only.json` (read by `sqm_bands.py` as `sqm_atlas.json`);
- `x_deepread/push/jobs_b1_w074_only.tsv`;
- `jobs_b2_w074_only.tsv`.

## Dependencies

- villa (github.com/ScrollPrize/villa): `vesuvius/src` and Lasagna at `a329895ea5e9`; `fit_spiral.py` at `0fb38c45b` with three local patches (Erwin Nieuwlaar's shell-optional patch, a TrackGraph crossing-cache fix, and a port of villa PR #2028); the native modules in `volume-cartographer/python/vc/`.
- VC3D `vc_grow_seg_from_seed` and `vc_render_tifxyz`: release AppImage `fb8c2c4` on rented boxes; locally a build of villa main from about 10 October with PR #2007's grid-cache fix.
- Erwin Nieuwlaar's PHerc. 343 recipe (github.com/Nieuwlaar/pherc343-first-letters, `e8370f1f`): `mesh/snap_meshes.py`, `ink/scripts/run_ink.sh`, `ink/scripts/hecate_prep.py`.
- Models: Hecate 9.6 µm, Reader v2 and TRACE v3B (sha256 pins in the lane scripts); `YoussefMoNader/ink-8um-v8in-pherc1447-loo-w062` for the fourth-model check.
- Data, all public: the PHerc. 0306B volume, its m7 prediction, normal grids and Lasagna normals (Vesuvius Challenge open-data bucket); the spiral tracks (`dl.ash2txt.org/datasets/spiral_datasets/`); the Scan-Quality-Map atlas (github.com/jcooperkai-sys/Scan-Quality-Map).
- Vast.ai for the rented boxes: the `vastai` CLI with your own key, and an SSH key registered with Vast.
- Python 3.14 with torch 2.14.1 (cu130) and zarr 3.4.1 (`fl/remote/setup.sh`); a separate older torch env for the fitter (`erwin_setup.sh`).
