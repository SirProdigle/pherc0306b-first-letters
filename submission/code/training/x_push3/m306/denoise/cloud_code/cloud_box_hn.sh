#!/bin/bash
# cloud_box_hn.sh (ON the box) : DENOISE hard-negative arms, one per GPU, /work/self/configs/<arm>.json for each line of /work/self/arms.txt
P=/work/env/py/bin/python; export SELF_ROOT=/work/self V8_DIR=/work/v8in PYTHONPATH=/work/v8in OMP_NUM_THREADS=4
g=0
while read -r arm; do
  [ -z "$arm" ] && continue
  ( export CUDA_VISIBLE_DEVICES=$g
    for t in 1 2 3; do
      $P /work/code/train_cloud_hn.py /work/self/configs/$arm.json >> /work/logs/$arm.log 2>&1; rc=$?
      [ $rc -eq 0 ] && break
      if grep -q "OOM" /work/logs/$arm.log; then export GCK_FORCE=block; fi
      echo "$(date '+%F %T') $arm rc=$rc retry $t (GCK_FORCE=${GCK_FORCE:-})" >> /work/logs/$arm.log
    done
    echo "$(date '+%F %T') $arm EXIT rc=$rc" >> /work/logs/arms.log ) &
  g=$((g + 1))
done < /work/self/arms.txt
wait
echo "$(date '+%F %T') ALL ARMS DONE" >> /work/logs/arms.log
