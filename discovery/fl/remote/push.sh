#!/bin/bash
# push.sh HOST PORT SCROLL... : copy code/config/seeds/checkpoints to a worker (paths mirror local)
set -euo pipefail
H=$1; P=$2; shift 2
W=/home/liam/projects/vesuvius-work
SSH="ssh -i $HOME/.ssh/vast_ed25519 -p $P -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"
$SSH root@$H "mkdir -p $W/fl $W/models/ink9_ens $W/third_party/pherc343-first-letters/ink/scripts"
rsync -az -e "$SSH" $W/fl/scripts $W/fl/remote $W/fl/scrolls.json $W/fl/priority.txt $W/fl/aws_fast.cfg root@$H:$W/fl/
rsync -az -e "$SSH" $W/third_party/pherc343-first-letters/ink/scripts/hecate_prep.py root@$H:$W/third_party/pherc343-first-letters/ink/scripts/
rsync -az -e "$SSH" $W/models/ink9_ens/dense_native-016000.rebuilt.pth $W/models/ink9_ens/hecate_9.6um.pth $W/models/ink9_ens/hecate.py root@$H:$W/models/ink9_ens/
for S in "$@"; do $SSH root@$H "mkdir -p $W/fl/$S"; rsync -az -e "$SSH" $W/fl/$S/seeds.json root@$H:$W/fl/$S/; done
echo pushed
