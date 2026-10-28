#!/bin/bash
# run_focus.sh KEY wNNN BBOX TITLE : focus_region battery on a PHerc0306B cloud-lane winding (render = x_focus306/renders/wNNN.zarr)
W=/home/liam/projects/vesuvius-work; cd $W; K=$1; w=$2; B=$3; T=$4
L=x_vast_plan/results/PHerc0306B_z8400
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VC3D_CONFIG_DIR=$W/x_focus306/vc3dcfg
Z=$(ls $L/maps/L0306bacw_$w/*_d0_fwd_z.npy)
nice -n 15 envs/vcpy/bin/python x_focus/focus_region.py --key $K --render x_focus306/renders/$w.zarr --zmap $Z --face fwd --um 8.64 --bbox $B \
  --tifxyz $L/results/flat/$w/tifxyz/flatten.tifxyz --umb x_vast_plan/state/umb_PHerc0306B_s75.json --neighbours x_focus306/cfg/nb_$w.json \
  --depth --title "$T" --out $W/x_focus306/out > x_focus306/logs/focus_$K.log 2>&1
echo "focus $K rc=$?"
