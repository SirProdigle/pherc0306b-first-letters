#!/bin/bash
# run_depth_shuffle.sh : depth-shuffle control for lead A (x_submit/depth_shuffle.py) under the GPU lock; retried if a game kills it
W=/home/liam/projects/vesuvius-work; cd $W; mkdir -p x_submit/logs
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
for pass in 1 2 3; do
  [ -f x_submit/depth_shuffle/metrics.json ] && break
  nice -n 15 flock x_discord/gpu.lock envs/vcpy/bin/python x_submit/depth_shuffle.py > x_submit/logs/depth_shuffle_$pass.log 2>&1
  echo "$(date +%T) pass $pass rc=$?" >> x_submit/logs/depth_shuffle_run.log
  [ -f x_submit/depth_shuffle/metrics.json ] || sleep 300
done
echo "$(date +%T) DONE" >> x_submit/logs/depth_shuffle_run.log
