#!/bin/bash
# prep_box_dn.sh TAG [train] CID... : push DENOISE code, start env setup, upload render/shuffle/v8-in/known volumes and checkpoints
# (cids from m306/cloud/ckpt_map.tsv); with 'train' also the zone-checked SELF packs + DENOISE arm configs + trainer.
H=$1; shift; TR=0; [ "${1:-}" = train ] && { TR=1; shift; }
W=/home/liam/projects/vesuvius-work; P3=$W/.claude/worktrees/leadA-push3/x_push3; C=$P3/m306/cloud; D=$P3/m306/denoise/cloud_code
E="ssh -F $D/ssh_config"; t0=$(date +%s)
scp -F $D/ssh_config -q $D/infer.py $D/render_leadA.py $D/render_leadA_dn.py $D/known_job_dn.py $D/known_sets_box.json $D/box_setup.sh $D/start_workers.sh $D/train_cloud_hn.py $D/v8306lib.py $H:/work/code/
scp -F $D/ssh_config -q $D/worker_dn.sh $H:/work/code/worker.sh
timeout 20 $E -n $H 'cd /work && (setsid nohup bash /work/code/box_setup.sh > /work/logs/setup.log 2>&1 < /dev/null &); echo setup started'
rsync -a -e "$E" $W/x_refine/trials/P1_surface_refine/renders/P1V5Rc.zarr/.zgroup $W/x_refine/trials/P1_surface_refine/renders/P1V5Rc.zarr/0 $H:/work/render.zarr/ && echo "render $(($(date +%s)-t0)) s" &
rsync -a -e "$E" --exclude __pycache__ $W/third_party/v8in/config.json $W/third_party/v8in/model.safetensors $W/third_party/v8in/ink8um $H:/work/v8in/ && echo "v8in $(($(date +%s)-t0)) s" &
rsync -a -e "$E" $W/x_req/20261020/A10/v4c_work/shuf_render.zarr/.zgroup $W/x_req/20261020/A10/v4c_work/shuf_render.zarr/0 $H:/work/shuf.zarr/ && echo "shuf $(($(date +%s)-t0)) s" &
for c in "$@"; do ck=$(awk -F'\t' -v c=$c '$1==c{print $3}' $C/ckpt_map.tsv); rsync -a -e "$E" $ck $H:/work/ckpt/$c.safetensors && echo "$c $(($(date +%s)-t0)) s" & done
rsync -aL -e "$E" $P3/aclick/bench/f9x/data/20250919125754/sv31.zarr/ $H:/work/known/k1.zarr/ && echo "k1 $(($(date +%s)-t0)) s" &
rsync -aL -e "$E" $W/x_push3_data/paired/PHerc0841/20260220213127/native.zarr/ $H:/work/known/k2.zarr/ && echo "k2 $(($(date +%s)-t0)) s" &
if [ $TR = 1 ]; then
  $E $H 'mkdir -p /work/self/packs /work/self/configs /work/self/runs'
  rsync -a -e "$E" $P3/m306/self/packs/ $H:/work/self/packs/ && echo "packs $(($(date +%s)-t0)) s" &
  rsync -a -e "$E" $P3/m306/denoise/configs/ $H:/work/self/configs/ && scp -F $D/ssh_config -q $P3/m306/denoise/configs/arms.txt $D/cloud_box_hn.sh $H:/work/self/ && echo "configs $(($(date +%s)-t0)) s" &
fi
wait; echo "ALL_UPLOADED $(($(date +%s)-t0)) s"
