#!/bin/bash
W=/home/liam/projects/vesuvius-work; cd $W
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VC3D_CONFIG_DIR=$W/x_focus306/vc3dcfg
M=x_focus306/ink_maps/w074own/w074own; T=fl/sqm/PHerc0306B_f306_w074own/auto_grown_20261012144243662
for K in own_row own_up42; do
nice -n 15 envs/vcpy/bin/python x_focus/focus_region.py --key w074own_$K --render x_focus306/renders/w074own.zarr --zmap ${M}_d0_fwd_z.npy --face fwd --um 8.64 \
  --poly x_focus306/loops/$K.json --tifxyz $T --umb x_vast_plan/state/umb_PHerc0306B_s75.json --neighbours x_focus306/cfg/nb_own.json \
  --models hec=${M}__hec_d0.png,rv2=${M}__rv2_d0.png,trace=${M}__trace_d0.png --depth \
  --title "PHerc0306B w074 line, own-sheet patch (independent surface), reading face fwd: $K" --out $W/x_focus306/out > x_focus306/logs/focus_w074own_$K.log 2>&1
echo "focus $K rc=$?"
done
echo OWN_DONE
