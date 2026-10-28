# Build the image from scratch

`build.sh` builds the claim's Docker image from public sources, so you don't need to download the 6.6 GB image from the dataset repo.

```bash
git clone https://github.com/SirProdigle/pherc0306b-first-letters
cd pherc0306b-first-letters
bash docker/build.sh                                         # builds fl-claim-a7:built
mkdir -p out
docker run --rm -v $PWD/out:/out fl-claim-a7:built            # redraw the claim images and compare bytes
docker run --rm -v $PWD/out:/out fl-claim-a7:built smoke      # re-run A-hn on a CT crop
docker run --rm -v $PWD/out:/out fl-claim-a7:built exclusion  # re-check the training tiles against the exclusion zone
```

You need Linux x86-64 with Docker and curl, and about 40 GB of free disk: 3.6 GB for the build context, 12 GB for the image and the rest for Docker's build cache.

`build.sh` assembles the build context in `./fl-claim-a7_context` (set `CTX` to put it elsewhere). The code, the mesh and small records come from this repository. The ink maps, weights, VC3D AppImage, smoke-test crop and reference views come from the two Hugging Face repos at pinned revisions, and v8-in and Reader V3 come from their authors' repos. [`context.tsv`](context.tsv) lists every file with its source and sha256, and the script stops if a download doesn't match. It then builds with [`submission/docker/Dockerfile`](../submission/docker/Dockerfile), which pins everything else: the base image, the Ubuntu archive snapshot, Python and every pip package, PyTorch, the villa commit and the public checkpoints.

We tested it on 28 October 2026 with no build cache; the build took 4 minutes once the context was downloaded. The image has the same Ubuntu packages, Python packages, VC3D, villa sources and checkpoints as the shipped `fl-claim-a7:final`. `final` matched all 20 views byte for byte, `smoke` gave the same numbers as our original test, and `exclusion` allowed all 23,331 tiles. Its files match the shipped image's, except the header comments of 22 scripts, which we edited for release, and some unused files the shipped image carries over from our first version's image.
