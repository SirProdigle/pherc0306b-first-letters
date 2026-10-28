#!/bin/bash
# pull.sh HOST PORT TAG SCROLL... : fetch overlays/quality/done markers + patch meshes into fl/<SCROLL>@<TAG>/screen2
set -uo pipefail
H=$1; P=$2; T=$3; shift 3
W=/home/liam/projects/vesuvius-work
E="ssh -i $HOME/.ssh/vast_ed25519 -p $P -o LogLevel=ERROR -o ConnectTimeout=20"
for S in "$@"; do
  D=$W/fl/$S@$T/screen2; mkdir -p $D/maps $D/grown
  rsync -az -e "$E" --include='*/' --include='*_overlay.png' --include='done_d0' --include='quality_tiles.npy' --include='quality.json' --include='spectral.json' \
        --exclude='*' root@$H:$W/fl/$S/screen2/maps/ $D/maps/
  rsync -az -e "$E" --include='*/' --include='meta.json' --include='x.tif' --include='y.tif' --include='z.tif' --exclude='*' \
        root@$H:$W/fl/$S/screen2/grown/ $D/grown/
  rsync -az -e "$E" root@$H:$W/fl/$S/seeds.json $W/fl/$S/seeds.json 2>/dev/null
  echo "$S@$T: $(ls $D/maps | wc -l) map dirs, $(grep -l . $D/maps/*/done_d0 2>/dev/null | wc -l) done"
done
