# Reproduce

The Docker image checks the claim in a few minutes. The scripts rebuild everything from the public CT.

## Docker

You need Linux x86-64 with Docker and `zstd`, 4 CPU cores and about 20 GB of free disk (6.6 GB for the download, 12 GB for the loaded image). The three checks below need no GPU.

Download `docker/fl-claim-a7_final.tar.zst` and its `.sha256` from the [dataset repo](https://huggingface.co/datasets/Prodigle/pherc0306b-first-letters), then:

```bash
sha256sum -c fl-claim-a7_final.tar.zst.sha256
zstd -dc fl-claim-a7_final.tar.zst | docker load        # loads fl-claim-a7:final
mkdir -p out
docker run --rm --network none --cpus 4 -v $PWD/out:/out fl-claim-a7:final              # final
docker run --rm --network none --cpus 4 -v $PWD/out:/out fl-claim-a7:final smoke        # smoke
docker run --rm --network none --cpus 4 -v $PWD/out:/out fl-claim-a7:final exclusion    # exclusion
```

| mode | what it does | result | time on 4 CPUs |
|---|---|---|---|
| `final` | redraws the claim image and 19 other views from the shipped maps and checks each PNG's sha256 | MATCH, 20 of 20 | 126 s |
| `smoke` | re-runs the three A-hn seeds on a 169 × 169 px CT crop inside the box, compares them with the shipped maps, and rebuilds the product byte for byte, using the shipped TRACE v3B map | PASS: largest difference 0.00174, r 0.99999 | 297 s |
| `exclusion` | re-checks all 23,331 training tiles against the exclusion zone in 3-D | ALL ALLOWED | 44 s |
| `pipeline` (GPU) | runs the ink models on the full render, then `final` | not run end to end | |
| `train` (GPU) | repeats the training, from pool masks to the claim map; mount the inputs listed in `/opt/fl/train/INPUTS.tsv` (all in the dataset repo) | not run end to end | |

GPU kernels are not bit-exact across cards, so a re-trained model differs in sha256. The `train` mode reports how closely the new map matches the shipped one. Training with the shipped configs peaked at 19 GB of GPU memory.

**12 GB cards.** The trainer's own default, per-block activation checkpointing (`grad_ckpt "block"` in `train_cloud_hn.py`), fits the training on a single 12 GB card (RTX 4070, 16 GB RAM). The shipped hn4 configs turn it off (`grad_ckpt: false`, `vram_gb: 30.0`); `configs/<member>_12gb.json` are the same configs with `grad_ckpt: "block"` and `vram_gb: 10.5`, run with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. BatchNorm statistics under checkpointing equal the no-checkpoint run exactly. Stages 1 to 8 and the claim image were re-run this way: the whole card peaked at 9,706 MiB while training, each member took 47 to 54 minutes, the retrained members match the shipped maps at r 0.9925 to 0.9934 (the shipped seeds agree with each other at 0.932 to 0.942) and the fused map at r 0.9992. Write-up and scripts: https://github.com/comoke/first-letters-12gb.

**Windows / WSL2.** Under WSL2, GPU memory is charged to Windows commit memory. A logged training run took about 14.3 GB of Windows commit (8,698 MiB of GPU memory above idle plus about 5 GiB of packs and workers in RAM), so on a 16 GB PC the pagefile must be able to grow; a full system drive stops it growing and takes the WSL VM down.

To build the image yourself instead of downloading it, run `bash docker/build.sh` from the repository root; [`docker/README.md`](../docker/README.md) has the details. The recipe is in [`submission/docker/`](../submission/docker/). It pins the base image digest, the Ubuntu archive snapshot, Python and every pip package, the villa commit, the VC3D AppImage, and every public checkpoint, by revision and sha256. The image also holds Hecate's weights, which the claim does not use. [`THIRD_PARTY_NOTICES.md`](../submission/docker/THIRD_PARTY_NOTICES.md) lists everything inside it with its licence, including the GPLv3 VC3D AppImage. Our test logs are in [`submission/method/checks/docker_tests/`](../submission/method/checks/docker_tests/).

## From the CT

The scripts in [`submission/code/`](../submission/code/) are the files that ran. They use the absolute paths of our working tree (`/home/liam/projects/vesuvius-work`), which the Docker image recreates under `/opt/fl`. Outside Docker, set `FL_ROOT` where a script reads it, or edit the path constants at the top.

1. Take the mesh in [`submission/mesh/`](../submission/mesh/). `code/surface/build_surface.sh` rebuilds it byte for byte from the GrowPatch and the touch-up inputs in `code/surface/inputs/`.
2. Render it with VC3D `vc_render_tifxyz` against volume `20250521133212`: 41 layers, 1 voxel apart, 20 below to 20 above the surface. Our render is `results/claim_surface_render_41layers.zarr` in the dataset repo.
3. Run TRACE v3B with `code/ink_v2.py` (weights pinned in `docker/checkpoints.tsv`), and A-hn with `code/ink_v8.py --nsig 3`, once per seed (weights in the model repo).
4. Fuse with `code/fuse_v7.py`, which multiplies TRACE v3B by the mean of the three A-hn maps. Compare with `results/fused/fwd_prod.npy`; `fuse.json` has the sha256 of each input.
5. Draw the image with `code/make_claim_image_v2.py` and `code/config_v7.json`. The claim image's `.json` records the levels and the box.
6. To retrain, run `code/training/x_push3/m306/denoise/cloud_code/train_cloud_hn.py` with the configs `hn4_s{0,1,2}.json`. It reads the arrays in `data/3b_cloud_packs/` and needs one GPU with 24 GB or more, or 12 GB with the `hn4_*_12gb.json` configs described above. [METHOD.md](METHOD.md#4-training-a-hn) gives the recipe.
