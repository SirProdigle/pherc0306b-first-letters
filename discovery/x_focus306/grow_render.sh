#!/bin/bash
# grow_render.sh TAG X Y Z GENS : GrowPatch at a sheet point (no snap), then a 41-layer real-CT render of it
W=/home/liam/projects/vesuvius-work; cd $W; T=$1
export PYTHONPATH=/home/liam/projects/villa/vesuvius/src VC3D_CONFIG_DIR=$W/x_focus306/vc3dcfg OMP_NUM_THREADS=4
m=$(nice -n 15 envs/vcpy/bin/python fl/scripts/sqm_grow.py PHerc0306B $2 $3 $4 f306_$T --gens $5 --nosnap | tail -1)
echo "grown $T -> $m"
[ "$m" = NONE ] && exit 1
O=$W/x_focus306/renders/$T.zarr
[ -f $O/0/.zarray ] || nice -n 15 $W/vc_build_main/bin/vc_render_tifxyz --volume $W/fl/vol/PHerc0306B.zarr \
  --remote-url https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0306B/volumes/20250521133212-8.640um-1.2m-116keV-masked.zarr \
  --prefetch-remote --group-idx 0 --scale 1 --segmentation $m --num-slices 41 --slice-step 1 --cache-gb 6 --zarr-output $O > $W/x_focus306/logs/render_$T.log 2>&1
echo "rendered $T rc=$? mesh=$m"
