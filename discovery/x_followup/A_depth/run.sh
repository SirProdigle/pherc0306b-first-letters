#!/bin/bash
# A_depth/run.sh : depth sweep of lead A (0306B w074 line) on its own-sheet surface render (x_focus306/renders/w074own.zarr, 41
# layers, surface = layer 20).  Max-quality recipe (x_deepread/ink_full.py: Hecate s16 + Rv2 + TRACE, 8x TTA, d-1..+1 around
# the centre layer) centred on layers 17..23, i.e. -3..+3 vx off the fitted surface, over a 30 x 30 mm box around the line
# (render x 991-4463, y 1089-4561).  Faint letters whose ink sits off the fitted surface should sharpen at their own depth.
W=/home/liam/projects/vesuvius-work; cd $W; T=x_followup/A_depth
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
for pass in 1 2 3; do                                         # redo depths interrupted by a game
for c in 20 17 23 18 22 19 21; do
  [ -f $T/c$c/fwd_fused.npy ] && continue
  nice -n 15 flock x_discord/gpu.lock envs/vcpy/bin/python x_deepread/ink_full.py x_focus306/renders/w074own.zarr $T/c$c --c $c --um 8.64 --faces fwd --crop 1089,4561,991,4463 > $T/logs/ink_c$c.log 2>&1
  echo "$(date +%T) c$c rc=$?" >> $T/logs/run.log
done
done
echo "$(date +%T) DONE" >> $T/logs/run.log
