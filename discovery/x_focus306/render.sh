#!/bin/bash
# render.sh wNNN : 41-layer real-CT render of a PHerc0306B cloud-lane winding (same grid as the lane z map)
W=/home/liam/projects/vesuvius-work; w=$1
T=$W/x_vast_plan/results/PHerc0306B_z8400/results/flat/$w/tifxyz/flatten.tifxyz
O=$W/x_focus306/renders/$w.zarr
[ -f $O/0/.zarray ] && { echo "have $O"; exit 0; }
export VC3D_CONFIG_DIR=$W/x_focus306/vc3dcfg OMP_NUM_THREADS=8
nice -n 15 $W/vc_build_main/bin/vc_render_tifxyz --volume $W/fl/vol/PHerc0306B.zarr \
  --remote-url https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0306B/volumes/20250521133212-8.640um-1.2m-116keV-masked.zarr \
  --prefetch-remote --group-idx 0 --scale 1 --segmentation $T --num-slices 41 --slice-step 1 --cache-gb 6 --zarr-output $O > $W/x_focus306/logs/render_$w.log 2>&1
echo "rendered $w rc=$?"
