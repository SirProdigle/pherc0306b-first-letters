# Models and data

The weights and data are in two Hugging Face repos. Each has a `SHA256SUMS` listing every file. All of it is CC BY-NC 4.0.

| repo | size | contents |
|---|---|---|
| [`Prodigle/pherc0306b-ink-ahn`](https://huggingface.co/Prodigle/pherc0306b-ink-ahn) | 6 GB | model weights, every saved stage, with their training records |
| [`Prodigle/pherc0306b-first-letters`](https://huggingface.co/datasets/Prodigle/pherc0306b-first-letters) (dataset) | 17 GB | training data at every stage, the claim's ink maps and render, and the Docker image |

## Weights

Each checkpoint folder holds `model.safetensors`, `config.json` and `meta.json`. The `meta.json` holds the full training config, the data sources and the sha256 of the starting weights. Each run's log, config and tile list is in `run_records/`.

| checkpoint | model | role | sha256 of `model.safetensors` |
|---|---|---|---|
| `hn4_s0_final` | A-hn, seed 0 | claim | `9a2812843924c50cafaa2b598f62c92ceca9f0d1f5d05b7bbf25764814ec7009` |
| `hn4_s1_final` | A-hn, seed 1 | claim | `5c2740c1f33ee5db73ca97425669607a2ffe4ae9af8b9d09d61f1c43165e60ad` |
| `hn4_s2_final` | A-hn, seed 2 | claim | `a294716b083769e8116d5f8d3cfa6a09378ca5c7e51c32225d04703266ad8ce7` |
| `hn4_s*_step_0300` | A-hn at step 300 of 600 | stage | see `SHA256SUMS` |
| `hn4_decoy_final` | decoy: PHerc. 0306B labels moved 3.5 × 3.0 mm | control | `b9cd0cbb61df991e2489c79bf085df1f1954143379abff044a641063ae31a1fe` |
| `self_main_final`, `self_main_s1_final`, `self_main_s2_final` | model A: no hard negatives | comparison | `835847af…`, `ae1bce5e…`, `b47912a8…` |
| `self_343h_final` | model B: adds PHerc. 0343 and 0139 labels to the human half | comparison | `458909dc…` |
| `hn4B_s0_final` | model B-hn: model B with hard negatives | comparison | `760bddaa…` |
| `v8306_343h_s0_final` | model C: public human labels only, no PHerc. 0306B data | comparison | `d468011f…` |
| `v8Q_final` | v8Q: v8-in fine-tuned on PHerc. 1447 w058 and the PHerc. 0343 tracings | reading-aid views | `4cb8e921…` |

The comparison models' mid-training stages are there too. Their maps of the line are in [`submission/supporting/models/`](../submission/supporting/models/). The v8-in loader (`ink8um`) is in `code/v8in_loader/`.

## Data

| path | size | contents |
|---|---|---|
| `data/0_exclusion_zone/` | 24 MB | the zone around the claimed line, its build and check code, every training tile, and the audit |
| `data/0_inputs/` | 4.0 GB | the five PHerc. 0306B surfaces used for machine labels: tifxyz meshes and 41-layer CT renders |
| `data/1_pool_masks/` | 16 MB | per surface: distance to the zone, on-sheet score, CT coverage |
| `data/2_teacher_maps/` | 677 MB | the four teachers' maps on those surfaces |
| `data/3_machine_labels/` | 125 MB | the machine labels (`main` was used; `sc` and `decoy` are variants), with statistics |
| `data/3b_cloud_packs/` | 2.9 GB | the exact arrays the seeds trained from, with the PHerc. 1447 anchor packs |
| `results/claim_surface_render_41layers.zarr` | 1.6 GB | the CT render of the claim surface |
| `results/member_maps/` | 1.0 GB | ink maps of TRACE v3B, the three A-hn seeds and the decoy, plus depth-shuffled versions (`shuf_*`); Reader v2, Reader V3, v8-in and v8Q for the reading-aid views |
| `results/fused/` | 0.4 GB | TRACE v3B × A-hn (`fwd_prod.npy`), the A-hn mean, and `fuse.json` with every input's sha256 |
| `results/claim_map_fwd_fin.npy`, `depth_shuffle_map_fwd_fin.npy` | 146 MB | the claim map and its depth-shuffle control |
| `results/ct_surface_layer20.png` | 19 MB | the surface CT layer under `ink_over_CT.jpg` |
| `docker/build/` | 357 MB | files only the from-scratch build needs: the VC3D AppImage, the smoke-test crop, two training-tree files and the reference views |
| `docker/fl-claim-a7_final.tar.zst` | 6.6 GB | the Docker image (sha256 `1832832330ab67b134795eaedd43d6d5c3dcf004596e1fd287d2e71b3af73536`); see [REPRODUCE.md](REPRODUCE.md) |

Maps are float16 arrays on the render raster (4900 × 7780 px; claim frame in [METHOD.md](METHOD.md#2-render)).

The PHerc. 1447 surfaces and labels come from HF `YoussefMoNader/ink-8um-pherc1447-surfaces`, revision `7e4d918712a8d8642a6fdeee45b80217c1e98317`. The CT is the public PHerc. 0306B volume `20250521133212`.
