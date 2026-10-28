#!/bin/bash
# build.sh [TAG] [CTX] : assemble the context (make_context.sh) and build the image, niced (version 7: default tag fl-claim-a7:draft).
#   default (owner's machine): Dockerfile.from_v1 = v1's image fl-claim-a:final + the package Dockerfile's v2 tail, --network none
#   FULL=1: the self-contained package Dockerfile (needs network on a machine without v1's layers: apt snapshot, uv/Python, torch,
#           the pinned pip set, villa sources, the public checkpoints)
set -uo pipefail
TAG=${1:-fl-claim-a7:draft}
CTX=${2:-$PWD/docker_ctx}                                     # the owner's test build used a scratch dir on /
HERE=$(cd "$(dirname "$0")" && pwd); LOG=$(dirname "$CTX")/build.log
bash "$HERE/make_context.sh" "$CTX" || exit 1
t0=$(date +%s)
if [ "${FULL:-0}" = 1 ]; then nice -n 15 docker build --progress=plain -t $TAG "$CTX" > $LOG 2>&1; rc=$?
else
  diff <(sed -n '/^# v8-in: our ink8um loader/,$p' "$HERE/Dockerfile") <(sed -n '/^# v8-in: our ink8um loader/,$p' "$HERE/Dockerfile.from_v1") > /dev/null \
    || { echo "Dockerfile.from_v1's v2 tail differs from the package Dockerfile"; exit 1; }
  cp "$HERE/Dockerfile.from_v1" "$CTX/"
  nice -n 15 docker build --network none --progress=plain -f "$CTX/Dockerfile.from_v1" -t $TAG "$CTX" > $LOG 2>&1; rc=$?
fi
echo "build rc=$rc seconds=$(( $(date +%s) - t0 ))" | tee -a $LOG
docker image inspect $TAG --format 'image {{.Id}} size_bytes={{.Size}}' 2>/dev/null | tee -a $LOG
