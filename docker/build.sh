#!/usr/bin/env bash
# build.sh [TAG] : build the claim's Docker image from public sources.
#   1. assembles the build context in $CTX from this repository, the two Hugging Face repos (at the revisions below) and two
#      upstream checkpoints, following context.tsv and checking the sha256 of every download;
#   2. builds it with submission/docker/Dockerfile.
# Env: CTX (default ./fl-claim-a7_context), HF_TOKEN (optional, sent to huggingface.co), NO_BUILD=1 (context only),
#      DOCKER_BUILD_ARGS (extra arguments for docker build, e.g. --no-cache). Re-running skips files that are already in place.
set -euo pipefail
TAG=${1:-fl-claim-a7:built}
REPO=$(cd "$(dirname "$0")/.." && pwd)
CTX=${CTX:-$PWD/fl-claim-a7_context}
HF_DATASET=https://huggingface.co/datasets/Prodigle/pherc0306b-first-letters/resolve/f435998b55269762942a979052044309b29650ac
HF_MODEL=https://huggingface.co/Prodigle/pherc0306b-ink-ahn/resolve/e57c4d541c239e06450a98c1cf2015d3de8d887e
MESH=PHerc0306B_20250521133212_w074own_touchup_V5Rc

for t in curl sha256sum docker; do command -v $t > /dev/null || { echo "build.sh needs $t"; exit 1; }; done
AUTH=(); [ -n "${HF_TOKEN:-}" ] && AUTH=(-H "Authorization: Bearer $HF_TOKEN")
mkdir -p "$CTX"
n=0; got=0
while IFS=$'\t' read -r path src sha; do
  case "$path" in ''|\#*) continue;; esac
  n=$((n + 1)); dst=$CTX/$path; mkdir -p "$(dirname "$dst")"
  case "$src" in
    repo:*) cp "$REPO/${src#repo:}" "$dst"; continue ;;
    hf-dataset:*) url=$HF_DATASET/${src#hf-dataset:}; hdr=("${AUTH[@]}") ;;
    hf-model:*) url=$HF_MODEL/${src#hf-model:}; hdr=("${AUTH[@]}") ;;
    url:*) url=${src#url:}; hdr=() ;;
    *) echo "context.tsv: unknown source $src"; exit 1 ;;
  esac
  if [ -s "$dst" ] && echo "$sha  $dst" | sha256sum -c --quiet 2> /dev/null; then continue; fi
  echo "fetch $path"
  curl -fsSL --retry 3 "${hdr[@]}" -o "$dst.part" "$url"
  echo "$sha  $dst.part" | sha256sum -c --quiet || { echo "sha256 mismatch: $path ($url)"; exit 1; }
  mv "$dst.part" "$dst"; got=$((got + 1))
done < "$REPO/docker/context.tsv"
chmod -R a+rX "$CTX"
# the input checksums the Dockerfile verifies after copying them into the image
( cd "$CTX" && sha256sum data/maps/* data/smoke/* mesh/$MESH/* data/members/* ) > "$CTX/data/inputs.sha256"
( cd "$CTX/code" && sha256sum claim/* x_zoom/zoomlib.py x_research2/speed/fastink.py ) >> "$CTX/data/inputs.sha256"
echo "context: $n files in $CTX ($got downloaded, $(du -sh "$CTX" | cut -f1))"
[ "${NO_BUILD:-0}" = 1 ] && exit 0
docker build ${DOCKER_BUILD_ARGS:-} -t "$TAG" "$CTX"
docker image inspect "$TAG" --format "built $TAG: {{.Id}}"
