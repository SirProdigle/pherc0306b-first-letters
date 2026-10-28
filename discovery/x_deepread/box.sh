#!/bin/bash
# box.sh : max-quality deep read on a rented Vast box (x_deepread/README.md).  Resumable; stage markers in /root/dr.
#   setup -> checkpoints (sha256-checked) -> 0343 positive control (AP must be .693 +- .01) -> for each job in
#   push/jobs.txt: render 41 layers (S3 stream) -> ink_full.py (max-quality recipe) -> out/<name>/  -> DONE
set -uo pipefail
W=/home/liam/projects/vesuvius-work; D=$W/x_deepread; J=/root/dr
mkdir -p $J/out $J/renders $J/logs $J/vc3dcfg $J/vcache
op=$(cat $J/pid 2>/dev/null); if [ -n "$op" ] && [ "$op" != "$$" ] && kill -0 $op 2>/dev/null; then echo "already running ($op)"; exit 0; fi
echo $$ > $J/pid; rm -f $J/FAILED
( while true; do echo "$(date +%s) $(cat $J/stage 2>/dev/null | tr ' ' _)" > $J/heartbeat; sleep 60; done ) & HB=$!
trap 'kill $HB 2>/dev/null' EXIT
st() { echo "$*" > $J/stage; echo "$(date +%s) $*" >> $J/logs/box.log; }
fail() { echo "$*" > $J/FAILED; echo "$(date +%s) FAILED $*" >> $J/logs/box.log; exit 1; }

st setup
if [ ! -f $J/setup.ok ]; then
  bash $W/fl/remote/setup.sh > $J/logs/setup.log 2>&1      # its last line prints vc_render_tifxyz --help (exit 1)
  grep -q 'cuda True' $J/logs/setup.log || fail setup; touch $J/setup.ok
fi
NC=$(nproc); T=$(( NC > 16 ? 16 : NC ))
export PATH=$W/envs/vcpy/bin:$PATH VC3D_CONFIG_DIR=$J/vc3dcfg OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T
printf '[viewer]\nremote_cache_dir=%s/vcache\n' $J > $J/vc3dcfg/VC3D.ini

st checkpoints
getck() { [ -s $1 ] && echo "$3  $1" | sha256sum -c --quiet 2>/dev/null && return 0
          mkdir -p $(dirname $1); curl -fsSL --retry 3 -o $1 "$2" && echo "$3  $1" | sha256sum -c --quiet || { rm -f $1; return 1; }; }
getck $W/models/community/reader-v2/reader-v2-step040000.pth https://huggingface.co/domenicor046/reader-v2/resolve/main/reader-v2-step040000.pth \
      654ec5acec2b6c4788d9cb326e2f0b8c2c730584949a934ef775db7f6ab5d3a6 || fail rv2
getck $W/models/community/trace/v3B_best.pth https://github.com/Jashann/vesuvius-scrolling/releases/download/v1.0/v3B_best.pth \
      812ffe2b42b3293974052ebbee60a0fe939ca2bfc3f03a517a0bc2e2339e5da2 || fail trace
getck $W/models/ink9_ens/hecate_9.6um.pth https://huggingface.co/scrollprize/hecate/resolve/main/hecate_9.6um.pth \
      809f4f10f7cb7afa19b4bee0f7d2ab31edd7e11b664f9f210cc9c17f22fcfe5d || fail hecate
getck $W/models/ink9_ens/hecate.py https://huggingface.co/scrollprize/hecate/resolve/main/hecate.py \
      c232c18a1a86cfb91257a00db13288202bb8ae33732ba26f7291a9b8adb8da59 || fail hecate_py

st control343
if [ ! -f $J/control.json ]; then
  python3 $D/control343.py $D/push/control343.npz $J/control.json > $J/logs/control.log 2>&1 || fail control_run
fi
python3 -c "import json,sys; sys.exit(0 if json.load(open('$J/control.json'))['ok'] else 1)" || fail "control AP off (see control.json)"

mkdir -p $W/fl/vol
while read -r name mesh faces scroll recipe; do
  [ -z "$name" ] && continue; [ -f $J/out/$name/.done ] && continue
  scroll=${scroll:-PHerc0306B}                                         # 4th column (drive.py), default the lead scroll
  VOL=$(python3 -c "import json; print(json.load(open('$W/fl/scrolls.json'))['$scroll']['volume_url'])") || { echo "$(date +%s) $name no volume for $scroll" >> $J/logs/box.log; continue; }
  UMV=$(python3 -c "import json; print(json.load(open('$W/fl/scrolls.json'))['$scroll']['voxel_um'])")
  st render $name; R=$J/renders/$name.zarr
  if [ ! -f $R/.done ]; then
    rm -rf $R
    if $W/vc_build_main/bin/vc_render_tifxyz --volume $W/fl/vol/$scroll.zarr --remote-url $VOL --prefetch-remote --group-idx 0 \
         --scale 1 --segmentation $mesh --num-slices 41 --slice-step 1 --cache-gb 8 --zarr-output $R > $J/logs/render_$name.log 2>&1; then
      touch $R/.done
    else
      echo "$(date +%s) $name render failed" >> $J/logs/box.log; continue
    fi
  fi
  st ink $name
  if [ "${recipe:-mq}" = fast ]; then                                 # 5th column 'fast': first-pass recipe in 4 overlapping quarters
    read H Wd < <(python3 -c "import zarr; a=zarr.open('$R',mode='r')['0']; print(a.shape[1], a.shape[2])")
    hy=$((H / 2)); hx=$((Wd / 2)); ok=1
    for q in "0,$((hy + 256)),0,$((hx + 256))" "0,$((hy + 256)),$((hx - 256)),$Wd" "$((hy - 256)),$H,0,$((hx + 256))" "$((hy - 256)),$H,$((hx - 256)),$Wd"; do
      qn=${q//,/_}; mkdir -p $J/out/${name}_q$qn; echo "$q" > $J/out/${name}_q$qn/crop.txt
      [ -f $J/out/${name}_q$qn/.done ] && continue
      python3 $D/ink_full.py $R $J/out/${name}_q$qn --c 20 --um $UMV --faces $faces --fast --crop $q > $J/logs/ink_${name}_q$qn.log 2>&1 \
        && touch $J/out/${name}_q$qn/.done || { ok=0; echo "$(date +%s) $name q$qn ink failed" >> $J/logs/box.log; }
    done
    [ $ok = 1 ] && mkdir -p $J/out/$name && touch $J/out/$name/.done
  else
  python3 $D/ink_full.py $R $J/out/$name --c 20 --um $UMV --faces $faces > $J/logs/ink_$name.log 2>&1 \
    && touch $J/out/$name/.done || echo "$(date +%s) $name ink failed" >> $J/logs/box.log
  fi
  rm -rf $R                                                            # free disk; maps + ct20 are kept
done < $D/push/jobs.txt
st done; date +%s > $J/DONE
