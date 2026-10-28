# Method records

The files behind each step of the claim. Paths are relative to the package root. [`docs/METHOD.md`](../../docs/METHOD.md) describes the method, with the same section numbers.

## 1. Segment

| record | contents |
|---|---|
| `mesh/PHerc0306B_20250521133212_w074own_touchup_V5Rc/` | the tifxyz mesh |
| `mesh/distortion_in_box.json`, `mesh/box_area.json` | stretch and area inside the box |
| `code/surface/DECISION_touchup.json` | the touch-up's adoption checks and their numbers |
| `code/surface/build_surface.sh` | rebuilds the mesh byte for byte from the inputs in `code/surface/inputs/` |

## 2. Render

The 41-layer render is `results/claim_surface_render_41layers.zarr` in the Hugging Face dataset repo.

## 3. Ink models and the claim rule

| record | contents |
|---|---|
| `code/ink_v2.py`, `code/ink_v8.py` | TRACE v3B and A-hn inference |
| `code/fuse_v7.py` | the product of the two maps |
| `method/model/checkpoints.json` | the sha256 of every weight file |
| `method/model/fusion_record.json` | the sha256 of each member map and of the fused map |

## 4. Training A-hn

| record | contents |
|---|---|
| `code/training/` | the training code, as it ran |
| `method/model/pool_regions.json`, `tile_regions.json` | the five pool surfaces and the training regions on them |
| `method/model/<region>_masks.json`, `<region>_teach.json` | per surface: pool masks and teacher maps (statistics; the arrays are in the dataset repo) |
| `method/model/pseudo_labels_stats.json` | label rule and label statistics |
| `method/model/training_config.json`, `run_config.json`, `training_sources.json` | the training configuration and data sources |
| `method/model/checkpoint_*_config.json`, `checkpoint_*_meta.json` | the config and metadata saved with each checkpoint |
| `method/model/train_log.jsonl`, `members/<run>/` | per-step logs and run records for each seed and the decoy |
| `method/model/exclusion/` | the zone (`zone.npz`), its build and check code, and its description |
| `method/model/exclusion_audit.json`, `training_tiles.npz` | the tile audit and every tile's position |
| `method/model/data_sources.json` | data sources and licences; its "CC BY-NC-SA" line is superseded, as all our files are CC BY-NC 4.0 |

## 5. The image

| record | contents |
|---|---|
| `code/make_claim_image_v2.py`, `code/config_v7.json` | draw the claim image |
| `PHerc0306B_20250521133212_w074own_touchup_V5Rc.json` | every input checksum, the levels and the box |
| `method/checks/letter_dimensions.md`, `.json` | letter sizes |
| `method/checks/letter_grids/` | the letter grids' column layouts |
| `code/supporting/` | the scripts that made the letter table, the grids and the reading-aid banners |

## 6. Validation

| check | record |
|---|---|
| depth shuffle | `method/checks/depth_shuffle_variants_v7_trace_x_Ahn_product.json` |
| blank control R | `method/checks/blank_control_views_v7_trace_x_Ahn_product.json` |
| decoy | `method/checks/ctrl_panel_v7.json` |
| known text, PHerc. 0841 | `method/checks/known_text/` |
| exclusion | `method/model/exclusion_audit.json` |
| Docker | `method/checks/docker_tests/` |

## 7. Human choices

[`docs/METHOD.md`](../../docs/METHOD.md#7-human-choices) lists the choices made by hand and each use of the papyrologist's tracings.
