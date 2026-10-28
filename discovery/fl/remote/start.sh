#!/bin/bash
# start.sh SCROLL N GROWERS : grids, then grow+render (1-thread growers) and one ink server per GPU
set -uo pipefail
W=/home/liam/projects/vesuvius-work; S=$1; N=$2; G=$3
cd $W
export PATH=$W/envs/vcpy/bin:$PATH
[ -f fl/$S/ngrid_remote/.complete ] || { bash fl/scripts/get_ngrids.sh $S > logs/ngrid_$S.log 2>&1 && touch fl/$S/ngrid_remote/.complete; }
NG=$(nvidia-smi -L | wc -l)
for i in $(seq 0 $((NG-1))); do
  [ -f fl/run/ink$i.pid ] && kill -0 $(cat fl/run/ink$i.pid) 2>/dev/null && continue
  CUDA_VISIBLE_DEVICES=$i INK_WORKERS=4 OMP_NUM_THREADS=4 bash fl/scripts/svc.sh start ink$i envs/vcpy/bin/python fl/scripts/ink_server.py $([ $i = 0 ] && echo --clear-stale)
  sleep 40
  [ -f fl/run/ink${i}b.pid ] && kill -0 $(cat fl/run/ink${i}b.pid) 2>/dev/null || CUDA_VISIBLE_DEVICES=$i INK_WORKERS=4 OMP_NUM_THREADS=4 bash fl/scripts/svc.sh start ink${i}b envs/vcpy/bin/python fl/scripts/ink_server.py
done
bash fl/scripts/svc.sh start s$S envs/vcpy/bin/python fl/scripts/screen2.py $S --n $N --growers $G --threads 1 --no-ink
[ -f fl/run/prune.pid ] && kill -0 $(cat fl/run/prune.pid) 2>/dev/null || bash fl/scripts/svc.sh start prune bash fl/scripts/prune.sh
[ -f fl/run/score.pid ] && kill -0 $(cat fl/run/score.pid) 2>/dev/null || OMP_NUM_THREADS=2 bash fl/scripts/svc.sh start score envs/vcpy/bin/python fl/scripts/score_loop.py
