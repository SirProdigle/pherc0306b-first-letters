#!/bin/bash
# focus battery (no depth: already measured) on the cloud deep-read max-quality maps of the w074 own-sheet patch
W=/home/liam/projects/vesuvius-work; cd $W
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
M=x_focus306/ink_maps/w074own_dr; T=fl/sqm/PHerc0306B_f306_w074own/auto_grown_20261012144243662
for K in ${KEYS:-own_row own_up42 own_up52}; do
nice -n 15 envs/vcpy/bin/python x_focus/focus_region.py --key w074ownDR_$K --render x_focus306/renders/w074own.zarr --zmap $M/fused_fwd_z.npy --face fwd --um 8.64 \
  --poly x_focus306/loops/$K.json --tifxyz $T --umb x_vast_plan/state/umb_PHerc0306B_s75.json --neighbours x_focus306/cfg/nb_own.json \
  --models hec=$M/hec.png,rv2=$M/rv2.png,trace=$M/tr.png \
  --title "PHerc0306B w074 own-sheet patch, MAX-QUALITY deep-read map, reading face fwd: $K" --out $W/x_focus306/out > x_focus306/logs/focus_w074ownDR_$K.log 2>&1
echo "focus $K rc=$?"
done
