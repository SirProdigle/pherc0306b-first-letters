# Docker recipe

The recipe for `fl-claim-a7:final`, the image that reproduces the claim. The image itself is `docker/fl-claim-a7_final.tar.zst` in the Hugging Face dataset repo (6.62 GB, sha256 `1832832330ab67b134795eaedd43d6d5c3dcf004596e1fd287d2e71b3af73536`). [`docs/REPRODUCE.md`](../../docs/REPRODUCE.md) shows how to run it, and our test logs are in [`../method/checks/docker_tests/`](../method/checks/docker_tests/).

| file | what it is |
|---|---|
| `Dockerfile` | the self-contained recipe; it pins the base image digest, Python, every pip package, the villa commit, the VC3D AppImage and every public checkpoint |
| `Dockerfile.from_v1` | the recipe we built with, on top of our first version's image; it ends in the same steps |
| `requirements.txt`, `constraints.txt`, `checkpoints.tsv` | the pip pins and the public checkpoints, by revision and sha256 |
| `run.sh` | the entry point, with modes `final`, `smoke`, `exclusion`, `pipeline`, `train` and `shell` |
| `build.sh`, `make_context.sh`, `train_tree.py` | assemble the build context from our working tree and build the image; as written they run only on our machine. [`docker/build.sh`](../../docker/build.sh) builds the same image from public sources |
| `THIRD_PARTY_NOTICES.md`, `GPL-3.0.txt` | the licences of everything inside the image |
