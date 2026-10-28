#!/bin/bash
# lane_box.sh [LANE_ENV=/root/lane.env] : ONE scroll band end to end on a rented box (or a local dry run), idempotent and
# resumable: every stage leaves $LANE/stages/<stage>.ok and is skipped when re-run; a failure writes $LANE/FAILED and exits 1.
#
#   00 disk     free space >= NEED_GB, else FAILED (nothing downloaded yet)
#   01 setup    fl/remote/setup.sh + x_recipe_erwin/remote/erwin_setup.sh; FIT_SPIRAL_STOP_AT patch on the old fitter;
#               Reader v2 + TRACE checkpoints from their public URLs, sha256-checked
#   02 data     x_recipe_erwin/remote/erwin_data.sh for this band only (published tracks, or pushed local tracks: TRACKS=local)
#   03 fit      SENSE=CW|ACW: one full fit (catalog sense).  SENSE=AB: both senses to SHORT_N steps (FIT_SPIRAL_STOP_AT), snap test
#               (sense_geo.py), then the winner CONTINUES from its checkpoint to the full horizon (same LR schedule, resumed state)
#   04 post     snap all windings (Erwin's snap_meshes.py), then MAXW windings middle-out: lasagna flatten + 31-layer render,
#               inked as they land by ink_lane.py (Hecate+Rv2 at 8.64 um, Hecate+TRACE at 9.36 um), chunk cache capped at CACHE_GB
#   05 score    window_score.py (row angles for EVERY window)
#   06 triage   tri_rank.py --min -99 --flat (hard + soft gates)
#   07 cards    cards_from_tri.py: REVIEW tier + first 5 REVIEW-2
#   08 pack     $LANE/results.tar (JSON, TSV, logs, JPG cards, half-size overlay JPGs, CT previews, flat tifxyz) -> DONE
#
# Per-stage wall times go to $LANE/timing.tsv; $LANE/heartbeat is touched every 60 s with the current stage.
# Config (LANE_ENV, shell syntax): SCROLL Z0 Z1 TB UM SENSE TRACKS UMB [MAXW=40] [WCENTER=<median>] [SHORT_N=8000] [NEED_GB=150] [CACHE_GB=30] [STEPS=30000]
#   [FACE_GATE=0|1: ink only the reading face, x_recto/reading_face.py] [QUALITY_THR: --mask tile threshold, default 0.10]
# Local dry run: LANE_DRYRUN=1 LANE=<scratch dir> DS_DIR=<existing dataset root> FIT_RUN=<existing fit run dir> lane_box.sh
#   -> skips setup/data/fit (prints what it would run), then post/ink/score/triage/cards/pack for MAXW (default 2) windings.
set -uo pipefail
W=/home/liam/projects/vesuvius-work; X=$W/x_recipe_erwin; V=/home/liam/projects/villa
P=$(cd "$(dirname "$0")" && pwd)                                   # x_vast_plan (this script's dir)
LANE=${LANE:-/root/lane}; LANE_ENV=${LANE_ENV:-/root/lane.env}
[ -f "$LANE_ENV" ] && source "$LANE_ENV"
: "${SCROLL:?}" "${Z0:?}" "${Z1:?}" "${TB:?}" "${UM:?}" "${SENSE:?}" "${UMB:?}"
TRACKS=${TRACKS:-published}; MAXW=${MAXW:-40}; WCENTER=${WCENTER:-}; SHORT_N=${SHORT_N:-8000}; NEED_GB=${NEED_GB:-150}; CACHE_GB=${CACHE_GB:-30}; STEPS=${STEPS:-30000}
DRY=${LANE_DRYRUN:-}; export INK_MAX_BATCH=${INK_MAX_BATCH:-}   # small-VRAM dry runs only
[ -n "${QUALITY_THR:-}" ] && export FL_QUALITY_THR=$QUALITY_THR   # one tile threshold for ink --mask AND window_score (set before any stage)
[ -n "$DRY" ] && MAXW=${DRY_MAXW:-2}
S=$SCROLL; mkdir -p $LANE/stages $LANE/logs $LANE/results
op=$(cat $LANE/pid 2>/dev/null); if [ -n "$op" ] && [ "$op" != "$$" ] && kill -0 $op 2>/dev/null; then echo "lane_box already running (pid $op)"; exit 0; fi
echo $$ > $LANE/pid   # single instance per box; fleet.py checks this pid
DS=${DS_DIR:-$LANE/ds}; mkdir -p $DS
SP=${SPIRAL_DIR:-$X/spiral_old}; SPY=${SPIRAL_PY:-$W/envs/spiral_old/bin/python}
CORES=$(nproc); TH=${THREADS:-$CORES}
export OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH OPENBLAS_NUM_THREADS=$TH NUMBA_NUM_THREADS=$TH PYTHONUNBUFFERED=1 WANDB_MODE=disabled
mkdir -p $LANE/vc3dcfg $LANE/vcache; printf '[viewer]\nremote_cache_dir=%s/vcache\n' $LANE > $LANE/vc3dcfg/VC3D.ini   # lane-local chunk cache
export PATH=$W/envs/vcpy/bin:$HOME/.local/bin:$PATH VC3D_CONFIG_DIR=$LANE/vc3dcfg AWS_CONFIG_FILE=$W/fl/aws_fast.cfg
MEMB=$(cat /sys/fs/cgroup/memory.max 2>/dev/null); [[ "$MEMB" =~ ^[0-9]+$ ]] || MEMB=$(awk '/MemTotal/{print $2*1024}' /proc/meminfo)
MEM_GB=$(( MEMB / 1073741824 )); SDTW=$(( CORES > 4 ? CORES - 2 : 2 )); SDTM=$(( MEM_GB * 10 / 25 )); [ $SDTM -lt 2 ] && SDTM=2
[ $SDTW -gt $SDTM ] && SDTW=$SDTM   # make_surf_sdt: ~2.5 GB per worker (384^3 tile EDT); CORES-2 workers OOM-killed 64-core boxes (BrokenProcessPool)
export DSROOT=$DS SPIRAL_DIR=$SP SPIRAL_PY=$SPY SDT_WORKERS=$SDTW SNAP_PROCS=$(( CORES > 8 ? 8 : CORES ))
VOL=$(python3 -c "import json;print(json.load(open('$W/fl/scrolls.json'))['$S']['volume_url'])")
NICE=""; [ -n "$DRY" ] && NICE="nice -n 15"
CUR=init
echo "$(date +%s) start $S $Z0-$Z1 sense $SENSE tag $TB dry=${DRY:-0}" >> $LANE/logs/lane.log
( while true; do echo "$(date +%s) $(cat $LANE/cur_stage 2>/dev/null)" > $LANE/heartbeat; sleep 60; done ) & HB=$!
trap 'kill $HB 2>/dev/null' EXIT

fail() { echo "$CUR: $*" > $LANE/FAILED; echo "$(date +%s) FAILED $CUR: $*" >> $LANE/logs/lane.log; exit 1; }
stage() {   # stage NAME FUNC : run FUNC once, time it
  CUR=$1; [ -f $LANE/stages/$1.ok ] && return 0
  echo "$1" > $LANE/cur_stage; local t0=$(date +%s)
  "$2" || fail "stage function returned $?"
  local t1=$(date +%s); printf '%s\t%s\t%s\t%s\n' "$1" $t0 $t1 $((t1 - t0)) >> $LANE/timing.tsv; touch $LANE/stages/$1.ok
}
dry_skip() { echo "DRYRUN skip $CUR: $*" | tee -a $LANE/logs/lane.log; }
cache_trim() {   # size-capped LRU for VC3D's remote chunk cache (instead of prune-and-refetch)
  local C=$LANE/vcache; [ -d $C ] || return 0
  local used=$(du -s --block-size=1G $C 2>/dev/null | cut -f1)
  [ "${used:-0}" -le $CACHE_GB ] && return 0
  find $C -type f -printf '%A@ %s %p\n' | sort -n | awk -v cap=$(( CACHE_GB * 8 / 10 * 1073741824 )) -v used=$(( used * 1073741824 )) \
    '{ if (used <= cap) exit; used -= $2; print $3 }' | xargs -r rm -f
}

# ---------------------------------------------------------------------------------------------------------------------------
s_disk() {
  local free=$(df --output=avail -BG $LANE | tail -1 | tr -dc 0-9)
  echo "free ${free} GB, need $NEED_GB GB" >> $LANE/logs/lane.log
  [ -n "$DRY" ] && return 0
  [ "$free" -ge "$NEED_GB" ] || { echo "only ${free} GB free, need $NEED_GB" > $LANE/logs/disk.txt; return 1; }
}
s_setup() {
  [ -n "$DRY" ] && { dry_skip "fl/remote/setup.sh; erwin_setup.sh; patch_fitter.py $SP; checkpoint fetch"; return 0; }
  bash $W/fl/remote/setup.sh > $LANE/logs/setup.log 2>&1   # its last line prints vc_render_tifxyz --help, which exits 1
  grep -q 'cuda True' $LANE/logs/setup.log || return 1
  bash $X/remote/erwin_setup.sh > $LANE/logs/erwin_setup.log 2>&1 || return 1
  python3 $P/patch_fitter.py $SP >> $LANE/logs/setup.log 2>&1 || return 1
  [ -d $X/native_pkg/vc ] || bash $X/native/build.sh > $LANE/logs/native.log 2>&1 || return 1
  mkdir -p $W/models/community/reader-v2 $W/models/community/trace
  getck() {   # getck PATH URL SHA256
    [ -s $1 ] && echo "$3  $1" | sha256sum -c --quiet 2>/dev/null && return 0
    curl -fsSL --retry 3 -o $1 "$2" && echo "$3  $1" | sha256sum -c --quiet || { rm -f $1; return 1; }
  }
  getck $W/models/community/reader-v2/reader-v2-step040000.pth https://huggingface.co/domenicor046/reader-v2/resolve/main/reader-v2-step040000.pth \
        654ec5acec2b6c4788d9cb326e2f0b8c2c730584949a934ef775db7f6ab5d3a6 || return 1
  getck $W/models/community/trace/v3B_best.pth https://github.com/Jashann/vesuvius-scrolling/releases/download/v1.0/v3B_best.pth \
        812ffe2b42b3293974052ebbee60a0fe939ca2bfc3f03a517a0bc2e2339e5da2 || return 1
  $W/envs/vcpy/bin/python -c "import torch;assert torch.cuda.is_available();print(torch.cuda.get_device_name(0))" >> $LANE/logs/setup.log 2>&1
}
s_data() {
  [ -n "$DRY" ] && { dry_skip "erwin_data.sh $S $Z0 $Z1 CW $UMB (DSROOT=$DS, TRACKS=$TRACKS)"; return 0; }
  local nt=""; [ "$TRACKS" = local ] && nt=1
  NOTRACKS=$nt bash $X/remote/erwin_data.sh $S $Z0 $Z1 CW $UMB > $LANE/logs/data.log 2>&1 || return 1
  grep -q DATA_OK $LANE/logs/data.log || return 1
  local A=$DS/${S}_ACW; mkdir -p $A                              # ACW twin: shares everything but the scroll spec
  for f in tracks lasagna_inputs umbilicus.json; do ln -sfn $DS/$S/$f $A/$f; done
  python3 -c "import json;d=json.load(open('$DS/$S/spiral-scroll.json'));d['spiral_outward_sense']='ACW';json.dump(d,open('$A/spiral-scroll.json','w'))"
}
runfit() {   # runfit DSNAME TAG [extra env assignments...]
  local dsn=$1 tag=$2; shift 2
  [ -n "$(ls -d $LANE/fits/*_$tag 2>/dev/null)" ] && [ -f "$(ls -d $LANE/fits/*_$tag | head -1)/checkpoint_fitted.ckpt" ] && return 0
  ( cd $SP && env "$@" FIT_SPIRAL_OUT_DIR=$LANE/fits FIT_SPIRAL_RUN_TAG=$tag PYTHONPATH=$X/native_pkg${PYTHONPATH:+:$PYTHONPATH} \
      FIT_SPIRAL_CONFIG_OVERRIDES="{\"z_begin\": $Z0, \"z_end\": $Z1, \"input_disable_patches\": true, \"loss_weight_shell_outer\": 0, \"loss_weight_shell_patch_radius\": 0, \"dense_spacing_mode\": \"phase\", \"optimizer_num_training_steps\": $STEPS}" \
      $SPY fit_spiral.py --dataset $DS/$dsn --cache $LANE/cache ) > $LANE/logs/fit_$tag.log 2>&1
}
run_of() { ls -d $LANE/fits/*_$1 2>/dev/null | head -1; }
s_fit() {
  # drop corrupt .npz caches (e.g. a track-crossing cache cut off by a killed run) so the fitter rebuilds them instead of dying
  find $DS -name '*.npz' -size +0 2>/dev/null | while read f; do python3 -c "import sys,zipfile; sys.exit(0 if zipfile.is_zipfile('$f') else 1)" || { echo "corrupt cache removed: $f" >> $LANE/logs/lane.log; rm -f "$f"; }; done
  if [ -n "$DRY" ]; then
    dry_skip "SENSE=$SENSE: $([ $SENSE = AB ] && echo "runfit both senses with FIT_SPIRAL_STOP_AT=$SHORT_N, snap test, resume winner" || echo "runfit ${S}$([ $SENSE = ACW ] && echo _ACW) ${TB}$(echo $SENSE | tr A-Z a-z)")"
    mkdir -p $LANE/fits; ln -sfn ${FIT_RUN:?} $LANE/fits/dry_$TB${DRY_SENSE:-cw}; echo $TB${DRY_SENSE:-cw} > $LANE/best_tag; return 0
  fi
  case $SENSE in
    CW)  runfit $S ${TB}cw || return 1; echo ${TB}cw > $LANE/best_tag ;;
    ACW) runfit ${S}_ACW ${TB}acw || return 1; echo ${TB}acw > $LANE/best_tag ;;
    AB)
      local par=0; [ "$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)" -ge 46000 ] && par=1
      if [ $par = 1 ]; then
        runfit $S ${TB}cws FIT_SPIRAL_STOP_AT=$SHORT_N & local p1=$!; sleep 90
        runfit ${S}_ACW ${TB}acws FIT_SPIRAL_STOP_AT=$SHORT_N & local p2=$!; wait $p1 || return 1; wait $p2 || return 1
      else
        runfit $S ${TB}cws FIT_SPIRAL_STOP_AT=$SHORT_N || return 1; runfit ${S}_ACW ${TB}acws FIT_SPIRAL_STOP_AT=$SHORT_N || return 1
      fi
      snap_tag ${TB}cws || return 1; snap_tag ${TB}acws || return 1
      local g=$(python3 $X/tools/sense_geo.py $LANE/out/${TB}cws $LANE/out/${TB}acws 2>> $LANE/sense.txt)
      if [ -z "$g" ]; then   # not decisive: lower mean sheet-jump rejection wins, flagged weak
        g=$(python3 -c "
import csv
m=lambda d: sum(float(r['frac_rejected']) for r in csv.DictReader(open(d+'/snap/snap_report.tsv'),delimiter='\t'))
print('ACW' if m('$LANE/out/${TB}acws') < m('$LANE/out/${TB}cws') else 'CW')"); echo "weak: tie-break by mean rejections -> $g" >> $LANE/sense.txt
      fi
      echo "sense=$g (short fits at $SHORT_N steps)" >> $LANE/sense.txt
      local s=$(echo $g | tr A-Z a-z); local dsn=$S; [ $g = ACW ] && dsn=${S}_ACW
      runfit $dsn ${TB}$s FIT_SPIRAL_RESUME_PATH=$(run_of ${TB}${s}s)/checkpoint_fitted.ckpt || return 1
      echo ${TB}$s > $LANE/best_tag ;;
    *) return 1 ;;
  esac
  [ -f "$(run_of $(cat $LANE/best_tag))/checkpoint_fitted.ckpt" ]
}
link_meshes() {   # link_meshes TAG -> $LANE/out/TAG/fitted/wNNN
  local tag=$1 run=$(run_of $1); local src=$run/meshes/fitted_$tag; local o=$LANE/out/$tag; mkdir -p $o/fitted $o/snap $o/flat $o/logs $o/ct
  for d in $src/w[0-9][0-9][0-9]_$tag; do [ -f $d/x.tif ] && ln -sfn $d $o/fitted/$(basename $d _$tag); done
  ls $o/fitted | grep -E '^w[0-9]{3}$' | sort > $o/windings_all.txt
}
middle_out() {   # middle_out TAG N -> $LANE/out/TAG/windings.txt: the N windings nearest WCENTER (default: the median winding number)
  local o=$LANE/out/$1; python3 -c "
ws=[l.strip() for l in open('$o/windings_all.txt') if l.strip()]; n=sorted(int(w[1:]) for w in ws); m=int('$WCENTER') if '$WCENTER' else n[len(n)//2]
open('$o/windings.txt','w').write('\n'.join(sorted(ws, key=lambda w: (abs(int(w[1:])-m), w))[:$2])+'\n')"
}
snap_tag() {   # snap_tag TAG : link the fit's meshes, snap the middle-out MAXW windings (the snap test reads snap/snap_report.tsv)
  local tag=$1 o=$LANE/out/$1; link_meshes $tag; middle_out $tag $MAXW; [ -f $o/snap/.done ] && return 0
  SURF_ZARR=$DS/m7_$S $NICE python3 $W/third_party/pherc343-first-letters/mesh/snap_meshes.py $o/fitted $S $o/snap \
    --windings $o/windings.txt --z0 $Z0 --z1 $Z1 --level 0 --procs $SNAP_PROCS > $o/logs/snap.log 2>&1 && touch $o/snap/.done
}
one_winding() {   # flatten + render one winding into $LANE/renders/<TAG>_<w>.zarr
  local w=$1 tag=$2 o=$LANE/out/$2; local R=$LANE/renders/${tag}_$w.zarr; [ -f $R/.done ] && return 0
  local Sn=$o/snap/$w; [ -f $Sn/x.tif ] || { echo "$w no snap" >> $o/logs/post.log; return 0; }
  local F=$o/flat/$w/tifxyz/flatten.tifxyz
  if [ ! -f $F/x.tif ]; then
    echo "{\"external_surfaces\": [{\"path\": \"$Sn\"}]}" > $o/flat/$w.json
    (cd $V/lasagna && PYTHONPATH=$V/vesuvius/src:$V/lasagna timeout 3600 $NICE python3 fit.py configs/flatten_fast_nofilter.json $o/flat/$w.json \
       --out-dir $o/flat/$w --device cuda > $o/logs/flat_$w.log 2>&1)
  fi
  [ -f $F/x.tif ] || { echo "$w flatten failed" >> $o/logs/post.log; return 0; }
  rm -rf $R; mkdir -p $LANE/renders
  $NICE $W/vc_build_main/bin/vc_render_tifxyz --volume $W/fl/vol/$S.zarr --remote-url $VOL --prefetch-remote --group-idx 0 --scale 1 \
     --segmentation $F --num-slices 31 --slice-step 1 --cache-gb 4 --zarr-output $R > $o/logs/render_$w.log 2>&1 || { echo "$w render failed" >> $o/logs/post.log; return 0; }
  python3 -c "
import zarr,cv2; a=zarr.open('$R',mode='r')['0'][15]; cv2.imwrite('$o/ct/$w.jpg', a[::3,::3], [cv2.IMWRITE_JPEG_QUALITY, 85])" 2>/dev/null
  touch $R/.done; echo "$w rendered $(date +%s)" >> $o/logs/post.log
}
s_post() {
  local tag=$(cat $LANE/best_tag) o; o=$LANE/out/$tag
  snap_tag $tag || return 1
  rm -f $LANE/post.done; mkdir -p $LANE/rv2
  local vram=${FAST_VRAM_GB:-$(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) * 7 / 10240 ))}
  [ -n "$DRY" ] && vram=${FAST_VRAM_GB:-3}
  local face=""; [ "${FACE_GATE:-0}" = 1 ] && face="--face-from $o/flat $DS/$S/umbilicus.json"   # ink only the face read towards the umbilicus
  [ -n "${QUALITY_THR:-}" ] && export FL_QUALITY_THR=$QUALITY_THR                                  # --mask tile threshold (default 0.10)
  $NICE $W/envs/vcpy/bin/python $P/ink_lane.py $LANE/renders $LANE/rv2 $UM --post-done $LANE/post.done --vram-gb $vram $face > $LANE/logs/ink.log 2>&1 & local ink=$!
  rm -f $LANE/ink.done   # window_score is CPU-heavy (~3 min/winding at 4 threads): score inked windings while the rest render/ink
  ( while [ ! -f $LANE/ink.done ]; do OMP_NUM_THREADS=$(( TH > 3 ? TH / 2 : 1 )) $NICE $W/envs/vcpy/bin/python $W/fl/remote/spiral/window_score.py $LANE/rv2 \
      >> $LANE/logs/window_score.log 2>&1; sleep 60; done ) & local sc=$!
  local par=${POST_PAR:-$(( CORES >= 24 ? 3 : CORES >= 12 ? 2 : 1 ))}; [ -n "$DRY" ] && par=1
  export -f one_winding cache_trim; export LANE DS S Z0 Z1 W V VOL NICE CACHE_GB
  while read w; do echo "$w $tag"; done < $o/windings.txt | xargs -P $par -L 1 bash -c 'one_winding "$0" "$1"; cache_trim'
  touch $LANE/post.done; wait $ink; touch $LANE/ink.done; wait $sc   # score loop finishes its current pass, never killed mid-write
  local nr=$(ls -d $LANE/renders/*.zarr 2>/dev/null | wc -l) ni=$(ls $LANE/rv2/*/done_d0 2>/dev/null | wc -l)
  echo "rendered $nr inked $ni" >> $LANE/logs/lane.log; [ $ni -ge 1 ] && [ $(( ni * 10 )) -ge $(( nr * 8 )) ]   # tolerate a few ink failures
}
s_score() { OMP_NUM_THREADS=$TH $NICE $W/envs/vcpy/bin/python $W/fl/remote/spiral/window_score.py $LANE/rv2 > $LANE/logs/window_score.log 2>&1; }
s_triage() {
  local tag=$(cat $LANE/best_tag)
  $NICE $W/envs/vcpy/bin/python $W/fl/scripts/tri_rank.py $LANE/rv2 --tag $tag --min -99 --um $UM --flat $LANE/out/$tag/flat \
     --json $LANE/results/tri_rank.json > $LANE/results/tri_rank.txt 2> $LANE/logs/tri_rank.err
}
s_cards() {
  $NICE $W/envs/vcpy/bin/python $P/cards_from_tri.py $LANE/results/tri_rank.json $LANE/rv2 $LANE/renders $LANE/results/cards $UM \
     --label "$S z$Z0-$Z1" > $LANE/logs/cards.log 2>&1
}
s_pack() {
  local tag=$(cat $LANE/best_tag) R=$LANE/results o; o=$LANE/out/$tag
  mkdir -p $R/maps $R/ct $R/flat $R/logs
  cp $LANE/timing.tsv $LANE/sense.txt $LANE/best_tag $LANE_ENV $R/ 2>/dev/null
  for d in $LANE/rv2/*/; do n=$(basename $d); mkdir -p $R/maps/$n; cp $d/spectral_win.json $d/*_timing.json $d/done_d0 $R/maps/$n/ 2>/dev/null
    for f in $d/*_overlay.png; do [ -f $f ] && python3 -c "
import cv2; a=cv2.imread('$f'); cv2.imwrite('$R/maps/$n/'+'$(basename $f .png)'+'.jpg', cv2.resize(a,(a.shape[1]//2,a.shape[0]//2),interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY,85])"; done; done
  cp $o/ct/*.jpg $R/ct/ 2>/dev/null; cp $o/snap/snap_report.tsv $R/ 2>/dev/null
  for w in $(cat $o/windings.txt); do [ -d $o/flat/$w/tifxyz ] && { mkdir -p $R/flat/$w; cp -r $o/flat/$w/tifxyz $R/flat/$w/; rm -f $R/flat/$w/tifxyz/*/model*.pt; }; done
  for f in lane.log ink.log window_score.log cards.log; do tail -c 200000 $LANE/logs/$f > $R/logs/$f 2>/dev/null; done
  tail -c 50000 $LANE/logs/data.log > $R/logs/data.log 2>/dev/null; for f in $LANE/logs/fit_*.log; do [ -f $f ] && tail -c 20000 $f > $R/logs/$(basename $f); done
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > $R/gpu.txt 2>/dev/null; nproc > $R/cores.txt
  tar -C $LANE -cf $LANE/results.tar results && du -sh $LANE/results.tar >> $LANE/logs/lane.log
}

stage 00_disk s_disk
stage 01_setup s_setup
stage 02_data s_data
stage 03_fit s_fit
stage 04_post s_post
stage 05_score s_score
stage 06_triage s_triage
stage 07_cards s_cards
stage 08_pack s_pack
echo "$(date +%s) DONE" >> $LANE/logs/lane.log; date +%s > $LANE/DONE
