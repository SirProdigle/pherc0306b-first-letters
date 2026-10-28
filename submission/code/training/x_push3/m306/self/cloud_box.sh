#!/bin/bash
# cloud_box.sh (runs ON the rented box) : SELF arms, one per GPU, from /work/self/configs/*.json listed in /work/self/arms.txt
# (line i -> GPU i).  Each arm retries on exit 3 (OOM -> activation checkpointing 'block', as self_main); logs /work/logs/<arm>.log.
P=/work/env/py/bin/python; export SELF_ROOT=/work/self V8_DIR=/work/v8in PYTHONPATH=/work/v8in OMP_NUM_THREADS=4
g=0
while read -r arm; do
  [ -z "$arm" ] && continue
  ( export CUDA_VISIBLE_DEVICES=$g
    for t in 1 2 3; do
      $P /work/code/train_cloud.py /work/self/configs/$arm.json >> /work/logs/$arm.log 2>&1; rc=$?
      [ $rc -eq 0 ] && break
      if grep -q "OOM" /work/logs/$arm.log; then export GCK_FORCE=block; fi
      echo "$(date '+%F %T') $arm rc=$rc retry $t (GCK_FORCE=${GCK_FORCE:-})" >> /work/logs/$arm.log
    done
    echo "$(date '+%F %T') $arm EXIT rc=$rc" >> /work/logs/arms.log ) &
  g=$((g + 1))
done < /work/self/arms.txt
wait
echo "$(date '+%F %T') ALL ARMS DONE" >> /work/logs/arms.log
