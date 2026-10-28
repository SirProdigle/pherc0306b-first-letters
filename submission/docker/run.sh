#!/bin/bash
# run.sh MODE : container entrypoint for the claim images (versions 2 to 6) (PHerc0306B, mesh <MESH> = own-sheet GrowPatch + CT-driven right-end
# touch-up, ARAP re-flattened).  Output goes to $OUT (default /out; mount a host directory there).
#
#   final         (default) every display variant of the package from the ink maps shipped in the image: CPU only, no network, ~4 min.
#                 Writes final/variants/<file>.png + .json for each variant (the registry of /opt/fl/claim/config_v7.json, version 7) and
#                 final/<MESH>.png + .json = the LEAD variant; checks every PNG against the sha256 of the package's files (MATCH / exit 3).
#                 MAPS / CT / MESH (below) replace the shipped inputs (then there is no sha256 check).
#   pipeline      mesh -> 41-layer CT render (vc_render_tifxyz) -> Reader v2 + TRACE + Reader V3 over the context (ink_v2.py, ink_v3.py)
#                 -> v8-in models (ink_v8.py; default only the tiles over the 4 cm2 box; version 5 adds v8g3 = v8-in base with the
#                 +-3 sigma stitch window) -> every variant again -> compare_v2.py and a pixel comparison of every variant against the
#                 package (compare_*.json, compare_variants.json).
#   smoke         (version 7) CPU only, ~2-10 min: the version-7 model's weights (sha256-checked) re-run on a 169 x 169 px CT crop inside
#                 the claim box (data/smoke/crop.npz) with the claim recipe, compared with the shipped map fwd_fin.npy -> PASS / exit 1.
#                 Product rule (finalist.json fusion = product): every member's weights vs its shipped map, then fuse_v7.py rebuilds A,
#                 TRACE x A, its depth-shuffle map and TRACE x decoy from the shipped maps and checks them byte for byte.
#   exclusion     (version 7) CPU only, ~1 min: every PHerc0306B training tile of the model (train/exclusion/training_tiles.npz) re-checked
#                 in 3-D against the exclusion zone with the shipped zone, meshes and check.py -> ALL ALLOWED / exit 1.
#   train         (version 7) GPU: the model's full training run again, from the CT and labels mounted read-only at the paths they were
#                 read from (/opt/fl/train/INPUTS.tsv; docker/README.md), then its claim map; results and comparisons in $OUT/train/.
#   shell         bash inside the container.
#
# Environment (all optional):
#   LEAD        a view key of the registry (default: the package's lead, data/final/LEAD): the variant drawn as <MESH>.png
#   MESH        tifxyz dir (default: the package mesh in the image)
#   VOLUME      OME-Zarr of the CT: the public S3 URL (default) or a mounted local zarr (-v /data/vol.zarr:/vol:ro -e VOLUME=/vol)
#   VCACHE      VC3D chunk cache dir (default $OUT/vcache)
#   RENDER      an existing 41-layer render zarr (pipeline skips the render step)
#   STEPS       pipeline steps, comma-separated (default render,ink,v8,image,compare); each step skips finished outputs
#   INK_CROP    y0,y1,x0,x1 render px for Reader v2 / TRACE / Reader V3 (default 116,4167,694,7639 = the context u -30..+30 x v -20..+15 mm)
#   V8_MODELS   v8-in models for the v8 step: any of v8 (released base, +-1 sigma), v8q (our fine-tune), v8g3 (released base, +-3 sigma;
#               version 5) and fin (version 7's model, +-3 sigma), comma-separated (default: those the registry uses)
#   V8_REGION   box (default: only the tiles that touch the 4 cm2 box; inside the box the map is value-for-value the full run's,
#               outside it is NaN, and the variants are then drawn with the shipped map outside the box) | ctx (the whole context:
#               ~61k tile positions x 4 flips per model, ~1.5 h on an RTX 4090) | u0,u1,v0,v1 (claim-frame mm)
#   V8_MAX_MIN  minutes per ink_v8.py call before it stops between parts (default 28; re-run to continue: parts are kept)
#   INK_DEVICE  cuda (default) | cpu (fp32; slow: small regions only)
#   MODELS_DIR  a mounted dir of checkpoints (same file names) used instead of the baked-in ones; sha256-checked
#   CACHE_GB    VC3D in-memory chunk cache (default 4)
#   PNG_ZLIB    zlib-ng (default: libz = zlib-ng-compat 2.3.3, byte-identical PNGs) | system (Ubuntu zlib: same pixels, other bytes)
#   MAPS, CT    final mode: a dir with fwd_<model>.npy on the mesh's render raster (the registry's models), and the CT surface layer
set -euo pipefail
FL=/opt/fl; OUT=${OUT:-/out}; C=$FL/claim; CFG=$C/config_v7.json
FIN_W=$FL/third_party/v8in/finalist/model.safetensors; FIN_SHA=$(cut -d' ' -f1 $FL/third_party/v8in/finalist/SHA256SUMS 2>/dev/null || true)
MESHNAME=PHerc0306B_20250521133212_w074own_touchup_V5Rc
SHIPPED_MAPS=$FL/data/maps; SHIPPED_CT=$FL/data/maps/ct20.png; SHIPPED_IMG=$FL/data/final
BOXPX=$(python -c "import json, math; b = json.load(open('$CFG'))['box']; K = 1000 / 8.64
print(','.join(str(v) for v in (math.ceil((b['v0'] + 21) * K - 0.5), math.floor((b['v1'] + 21) * K - 0.5) + 1, math.ceil((b['u0'] + 36) * K - 0.5), math.floor((b['u1'] + 36) * K - 0.5) + 1)))")
#   render px (y0,y1,x0,x1) whose centres lie in the 4 cm2 box (version 4c and the version 5 candidates: 1639,2745,2431,6829)
LEAD=${LEAD:-$(cat $SHIPPED_IMG/LEAD)}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
mkdir -p $OUT; T0=$(date +%s); log() { echo "[$(( $(date +%s) - T0 ))s] $*"; }
MODE=${1:-final}; [ $# -gt 0 ] && shift
KEYS=$(python -c "import sys; sys.path.insert(0, '$C'); import build_variants as b; print(' '.join(b.registry('$CFG')[0]))")
case " $KEYS " in *" $LEAD "*) ;; *) echo "LEAD must be one of: $KEYS"; exit 2;; esac

# build_variants.py (-> make_claim_image_v2.py) with zlib-ng-compat 2.3.3 as libz (the PNG encoder of the machine that wrote the package)
variants() {   # variants MAPS CT MESH DEST
  if [ "${PNG_ZLIB:-zlib-ng}" = system ]; then python $C/build_variants.py --config $CFG --maps $1 --ct $2 --mesh $3 --out $4 --lead $LEAD --layout flat
  else LD_LIBRARY_PATH=/opt/zlib-ng${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} python $C/build_variants.py --config $CFG --maps $1 --ct $2 --mesh $3 --out $4 --lead $LEAD --layout flat; fi
}
leadfile() { python -c "import sys; sys.path.insert(0, '$C'); import build_variants as b; print(b.registry('$CFG')[0]['$LEAD']['file'])"; }
usedv8() { python -c "import sys, json; sys.path.insert(0, '$C'); import build_variants as b; v, p, _ = b.registry('$CFG')
print(','.join(m for m in ('v8', 'v8q', 'v8g3', 'fin') if any(m in p[x['pair']] for x in v.values())))"; }

case "$MODE" in
final)
  M=${MAPS:-$SHIPPED_MAPS}; CTI=${CT:-$SHIPPED_CT}; ME=${MESH:-$FL/mesh/$MESHNAME}
  log "final images: lead $LEAD, maps ${MAPS:-shipped}, CT ${CT:-shipped}, mesh ${MESH:-shipped}"
  variants $M $CTI $ME $OUT/final
  if [ -z "${MAPS:-}${CT:-}${MESH:-}" ] && [ "${PNG_ZLIB:-zlib-ng}" != system ]; then
    log "sha256 against the package's images:"
    (cd $OUT/final && sha256sum -c $SHIPPED_IMG/expected.sha256) && cmp -s $OUT/final/$MESHNAME.png $OUT/final/variants/$(leadfile).png \
      && log "MATCH: byte-identical to the package PNGs; $MESHNAME.png = $LEAD" || { log "DIFFERENT from the package PNGs"; exit 3; }
  fi
  log "done: $OUT/final" ;;

pipeline)
  ME=${MESH:-$FL/mesh/$MESHNAME}; STEPS=${STEPS:-render,ink,v8,image,compare}; V8R=${V8_REGION:-box}; V8M=${V8_MODELS:-$(usedv8)}
  VOLUME=${VOLUME:-https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0306B/volumes/20250521133212-8.640um-1.2m-116keV-masked.zarr}
  if [ -n "${MODELS_DIR:-}" ]; then                                  # mounted checkpoints replace the baked-in ones
    while read -r rel url sha; do case "$rel" in ''|\#*) continue;; esac
      src=$MODELS_DIR/$(basename $rel); [ -s $src ] || continue
      echo "$sha  $src" | sha256sum -c --quiet && ln -sf $src $FL/$rel && log "model from MODELS_DIR: $rel"
    done < $FL/env/checkpoints.tsv
  fi
  R=${RENDER:-$OUT/render.zarr}
  # 1. render: 41 layers, 1 voxel apart, layer 20 = the mesh surface, scale 1 (20 render px per 20-voxel grid cell)
  if [[ ",$STEPS," == *,render,* ]] && [ -z "${RENDER:-}" ] && [ ! -f $R/.done ]; then
    rm -rf $R; mkdir -p $OUT/vc3dcfg; printf '[viewer]\nremote_cache_dir=%s\n' "${VCACHE:-$OUT/vcache}" > $OUT/vc3dcfg/VC3D.ini
    PF=(); case "$VOLUME" in http*|s3*) PF=(--prefetch-remote);; esac
    log "render $ME from $VOLUME (chunk cache ${VCACHE:-$OUT/vcache})"; t=$(date +%s)
    VC3D_CONFIG_DIR=$OUT/vc3dcfg /opt/vc3d/AppRun vc_render_tifxyz --volume "$VOLUME" "${PF[@]}" --group-idx 0 --scale 1 \
      --segmentation $ME --num-slices 41 --slice-step 1 --cache-gb ${CACHE_GB:-4} --zarr-output $R > $OUT/render.log 2>&1
    touch $R/.done; log "render done ($(( $(date +%s) - t )) s)"
  fi
  # 2. Reader v2 + TRACE: 8x dihedral TTA, each averaged over layers 19-21, over the context window
  if [[ ",$STEPS," == *,ink,* ]]; then
    log "Reader v2 + TRACE on ${INK_DEVICE:-cuda}, window ${INK_CROP:-116,4167,694,7639}"; t=$(date +%s)
    python $C/ink_v2.py $R $OUT/ink --models rv2,tr --crop ${INK_CROP:-116,4167,694,7639} 2>&1 \
      | (grep -v -E "^---|Using|Patch size|Task|NetworkFrom|FutureWarning|torch.jit|guide_" || true) | tee -a $OUT/ink.log
    log "Reader v2 + TRACE done ($(( $(date +%s) - t )) s)"
  fi
  # 2b. Reader V3 (v3 package): the same recipe (ink_v3.py: 8x dihedral TTA, layers 19-21, same window), weights sha256-checked and
  #     loaded with torch.load(weights_only=True); writes fwd_v3.npy next to fwd_rv2 / fwd_tr
  if [[ ",$STEPS," == *,ink,* ]]; then
    log "Reader V3 on ${INK_DEVICE:-cuda}"; t=$(date +%s); set +e
    while true; do python $C/ink_v3.py $R $OUT/ink --crop ${INK_CROP:-116,4167,694,7639} 2>&1 | (grep -v -E "^---|Using|Patch size|guide_" || true) | tee -a $OUT/ink.log
      rc=${PIPESTATUS[0]}; [ $rc = 3 ] || break; done; set -e
    log "Reader V3 done ($(( $(date +%s) - t )) s)"
  fi
  # 3. v8-in models: stride 21, weight-normalised stitch, mirror4, d0 (layers 8..31, stored order); region V8_REGION.
  #    v8 = base, +-1 sigma; v8q = our fine-tune, +-1 sigma; v8g3 = base, +-3 sigma window (version 5: the map of the fusion)
  if [[ ",$STEPS," == *,v8,* ]]; then
    for mdl in ${V8M//,/ }; do
      WA=(); [ $mdl = v8q ] && WA=(--weights $FL/third_party/v8in/v8Q_final.safetensors --name v8q); [ $mdl = v8g3 ] && WA=(--nsig 3 --name v8g3)
      [ $mdl = fin ] && WA=(--weights $FIN_W --sha $FIN_SHA --nsig 3 --name fin)
      log "$mdl on ${INK_DEVICE:-cuda}, region $V8R"; t=$(date +%s); set +e
      python $C/ink_v8.py $R $OUT/ink --region=$V8R --max-minutes ${V8_MAX_MIN:-28} "${WA[@]}" 2>&1 | tee -a $OUT/ink_v8.log; rc=${PIPESTATUS[0]}; set -e
      [ $rc = 3 ] && { log "$mdl stopped at its time cap: run the v8 step again to continue (finished parts are kept)"; exit 3; }
      [ $rc = 0 ] || exit $rc
      log "$mdl done ($(( $(date +%s) - t )) s)"
    done
  fi
  # 4. every variant from the new maps (with V8_REGION=box: the new v8-in values inside the box, the shipped ones outside)
  if [[ ",$STEPS," == *,image,* ]]; then
    MN=$OUT/maps_new; mkdir -p $MN; for f in fwd_rv2.npy fwd_tr.npy fwd_v3.npy ct20.png; do ln -sf $OUT/ink/$f $MN/$f; done
    for mdl in v8 v8q v8g3 fin; do
      [ -s $SHIPPED_MAPS/fwd_$mdl.npy ] || continue
      [ -s $OUT/ink/fwd_$mdl.npy ] || { ln -sfn $SHIPPED_MAPS/fwd_$mdl.npy $MN/fwd_$mdl.npy; log "variants: shipped $mdl map (not re-run)"; continue; }
      if [ "$V8R" = ctx ]; then ln -sfn $OUT/ink/fwd_$mdl.npy $MN/fwd_$mdl.npy
      else python -c "import numpy as np,sys,os; s=np.load(sys.argv[1]); n=np.load(sys.argv[2]); m=np.isfinite(n); s[m]=n[m]; os.path.lexists(sys.argv[3]) and os.unlink(sys.argv[3]); np.save(sys.argv[3], s)
print(sys.argv[4], 'map for the variants: new values at', int(m.sum()), 'px (the box), shipped values elsewhere')" $SHIPPED_MAPS/fwd_$mdl.npy $OUT/ink/fwd_$mdl.npy $MN/fwd_$mdl.npy $mdl; fi
    done
    if [ -d $FL/data/members ] && [ -s $SHIPPED_MAPS/fwd_prod.npy ]; then           # version 7 product rule: re-fuse with the new TRACE map
      M=$FL/data/members; MEM=$(python -c "import json; print(' '.join('$M/fwd_' + m['name'] + '.npy' for m in json.load(open('$FL/third_party/v8in/finalist/finalist.json'))['members']))")
      python $C/fuse_v7.py $OUT/fused --trace $MN/fwd_tr.npy --members $MEM --decoy $M/fwd_decoy.npy
      for k in ensA prod prodd; do ln -sfn $OUT/fused/fwd_$k.npy $MN/fwd_$k.npy; done
      log "product maps re-fused with the new TRACE map and the shipped member maps (members: train mode re-makes them)"
    fi
    variants $MN $MN/ct20.png $ME $OUT/final; log "variants: $OUT/final"
  fi
  # 5. compare with the shipped maps, CT layer and images
  if [[ ",$STEPS," == *,compare,* ]]; then
    python $C/compare_v2.py --test-maps $OUT/ink --ref-maps $SHIPPED_MAPS --pair rt --test-ct $OUT/ink/ct20.png --ref-ct $SHIPPED_CT --out $OUT/compare_rt.json > /dev/null
    python $C/compare_v2.py --test-maps $OUT/ink --ref-maps $SHIPPED_MAPS --pair v3t --test-ct $OUT/ink/ct20.png --ref-ct $SHIPPED_CT --out $OUT/compare_v3t.json > /dev/null
    for mdl in v8 v8q v8g3 fin; do [ -s $OUT/ink/fwd_$mdl.npy ] && python $C/compare_v2.py --test-maps $OUT/ink --ref-maps $SHIPPED_MAPS --models $mdl --crop $BOXPX \
        --out $OUT/compare_${mdl}_box.json > /dev/null; done
    python - $OUT $SHIPPED_IMG <<'EOF'
import json, sys, glob, os, hashlib
import numpy as np, cv2
o, ref = sys.argv[1:3]; res = {}
for f in sorted(glob.glob(f'{ref}/variants/*.png')):
    n = os.path.basename(f); t = f'{o}/final/variants/{n}'
    if not os.path.exists(t): continue
    a, b = cv2.imread(t).astype(np.int16), cv2.imread(f).astype(np.int16); d = np.abs(a - b).max(-1)
    h = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
    res[n] = dict(png_identical=h(t) == h(f), pixels_differing=int((d > 0).sum()), max_abs_diff=int(d.max()))
json.dump(res, open(f'{o}/compare_variants.json', 'w'), indent=1)
for k in ('rt', 'v3t', 'v8_box', 'v8q_box', 'v8g3_box', 'fin_box'):
    p = f'{o}/compare_{k}.json'
    if os.path.exists(p):
        r = json.load(open(p)); print(f'[{k}] maps vs shipped (max |diff| x255, identical, r):', ', '.join(f"{m} {v['max_abs_x255']} {v['identical_frac']} {v['pearson_r']}" for m, v in r['models'].items()))
for n, v in res.items(): print(f'[image] {n}: PNG identical {v["png_identical"]}, px differing {v["pixels_differing"]}, max |diff| {v["max_abs_diff"]}')
EOF
    log "compare done: $OUT/compare_*.json"
  fi
  python - "$OUT" <<'EOF'
import json, sys, platform, torch, zarr, cv2, numpy
json.dump(dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.cuda.is_available(), zarr=zarr.__version__,
               cv2=cv2.__version__, numpy=numpy.__version__), open(f'{sys.argv[1]}/run_info.json', 'w'), indent=1)
EOF
  log "done: $OUT" ;;

smoke)
  T=$(python -c "import json; print(json.load(open('$FL/third_party/v8in/finalist/finalist.json'))['tta'])"); FU=$(python -c "import json; print(json.load(open('$FL/third_party/v8in/finalist/finalist.json')).get('fusion', 'zmean'))")
  mkdir -p $OUT/smoke; set +e; rc=0
  if [ "$FU" = product ]; then
    M=$FL/data/members; FD=$FL/third_party/v8in/finalist
    for row in $(python -c "import json; print(' '.join(m['name'] + ':' + m['sha256'] for m in json.load(open('$FL/third_party/v8in/finalist/finalist.json'))['members']))"); do
      n=${row%%:*}; h=${row#*:}
      log "smoke: member $n (sha256 ${h:0:12}) on the CT crop, CPU fp32, TTA $T, vs its shipped map"
      INK_DEVICE=cpu python $C/smoke_v7.py run $FL/data/smoke/crop.npz --weights $FD/$n.safetensors --sha $h --map $M/fwd_$n.npy --tta $T \
        --out $OUT/smoke/smoke_$n.json || rc=1
    done
    log "smoke: the product rule rebuilt from the shipped maps (fuse_v7.py), byte for byte"
    MEM=$(python -c "import json; print(' '.join('$M/fwd_' + m['name'] + '.npy' for m in json.load(open('$FL/third_party/v8in/finalist/finalist.json'))['members']))")
    SHM=$(python -c "import json; print(' '.join('$M/shuf_' + m['name'] + '.npy' for m in json.load(open('$FL/third_party/v8in/finalist/finalist.json'))['members']))")
    python $C/fuse_v7.py $OUT/smoke/fused --trace $SHIPPED_MAPS/fwd_tr.npy --members $MEM --shuf-trace $M/shuf_tr.npy --shuf-members $SHM \
      --decoy $M/fwd_decoy.npy --expect ensA=$SHIPPED_MAPS/fwd_ensA.npy prod=$SHIPPED_MAPS/fwd_prod.npy prodd=$SHIPPED_MAPS/fwd_prodd.npy \
      shuf_prod=$M/shuf_prod.npy shuf_ensA=$M/shuf_ensA.npy || rc=1
    cp $OUT/smoke/fused/fuse.json $OUT/smoke/fuse.json; rm -rf $OUT/smoke/fused
    [ $rc = 0 ] && log "smoke: PASS (every member + the fusion)" || log "smoke: FAIL"
  else
    log "smoke: version-7 model (sha256 ${FIN_SHA:0:12}) on the CT crop, CPU fp32, TTA $T, vs the shipped map"
    INK_DEVICE=cpu python $C/smoke_v7.py run $FL/data/smoke/crop.npz --weights $FIN_W --sha $FIN_SHA \
      --map $SHIPPED_MAPS/fwd_fin.npy --tta $T --out $OUT/smoke/smoke_result.json; rc=$?
  fi; set -e
  log "smoke done (rc $rc)"; exit $rc ;;

exclusion)
  log "exclusion: the model's PHerc0306B training tiles against the zone (shipped zone, meshes, check.py)"
  X=$FL/train/exclusion; set +e
  if [ ! -s $X/training_tiles.npz ]; then python -c "import json; a = json.load(open('$X/exclusion_audit.json')); print(a.get('note'), a.get('objects')); assert a['all_allowed']"; exit $?; fi
  python $C/exclusion_audit_v7.py verify $X/training_tiles.npz $X/tile_regions.json --mesh-root $X/meshes --zone-dir $X; rc=$?; set -e
  log "exclusion done (rc $rc)"; exit $rc ;;

train)
  # the training run again (GPU).  Mount every row of /opt/fl/train/INPUTS.tsv read-only at /home/liam/projects/vesuvius-work/<path>.
  TW=/home/liam/projects/vesuvius-work; FT=$FL/train/finalist_train.json
  P3=$TW/$(python -c "import json; print(json.load(open('$FT'))['P3'])"); LANE=$(python -c "import json; print(json.load(open('$FT'))['lane'])")
  TC=$TW/$(python -c "import json; print(json.load(open('$FT'))['train_config'])"); RUN=$(python -c "import json; print(json.load(open('$FT'))['run_name'])")
  STEPS=${STEPS:-$(python -c "import json; print(','.join(json.load(open('$FT'))['steps']))")}
  mkdir -p $OUT/train; log "train: lane $LANE, config $TC, steps $STEPS"
  miss=0; while IFS=$'\t' read -r kind path n b h; do case "$kind" in ''|\#*) continue;; esac
    have=$(find "$TW/$path" -type f 2>/dev/null | wc -l)
    [ "$have" -ge "$n" ] || { echo "not mounted or incomplete: $TW/$path ($kind: $have of $n files, $b bytes)"; miss=1; continue; }
    if [ "$kind" = file ] && [ "$h" != - ]; then echo "$h  $TW/$path" | sha256sum -c --quiet || miss=1; fi
  done < $FL/train/INPUTS.tsv
  [ $miss = 0 ] || { log "inputs missing or changed: see above (docker/README.md, mode train)"; exit 2; }
  again() { set +e; while true; do "$@"; rc=$?; [ $rc = 3 ] || break; log "chunk end (exit 3): continuing"; done; set -e; return $rc; }
  S=$P3/m306/self/code
  if [[ ",$STEPS," == *,pool,* ]]; then log "pool masks"; python $S/masks.py; fi
  if [[ ",$STEPS," == *,teachers,* ]]; then log "teachers: Reader V3, TRACE v3B, Reader v2"; again python $S/teachers.py readers
    log "teacher: v8-in base (stride 32, as run)"; V8_STRIDE=32 V8_BAND=${V8_BAND:-512} again python $S/teachers.py v8; fi
  if [[ ",$STEPS," == *,labels,* ]]; then log "machine labels"; python $S/labels.py; fi
  if [[ ",$STEPS," == *,exclusion,* ]]; then log "exclusion audit (recomputed from the new labels)"
    python $C/exclusion_audit_v7.py recompute $TC $OUT/train/exclusion --lane $LANE --p3 $P3; fi
  MEMB=$(python -c "import json; print(' '.join(m['name'] + ':' + m['train_config'] + ':' + (m.get('trainer') or '') for m in json.load(open('$FT')).get('members', [])))")
  if [ -n "$MEMB" ] && [[ ",$STEPS," == *,train,* ]]; then          # product rule: every member (seeds 1, 2 ran train_cloud.py = train_self.py's recipe)
    for row in $MEMB; do n=${row%%:*}; rest=${row#*:}; tcfg=${rest%%:*}; trn=${rest#*:}   # version 7: a member's own trainer (hard negatives)
      if [ -n "$trn" ]; then log "fine-tune member $n with $trn"; SELF_ROOT=$P3/m306/self V8_DIR=$TW/third_party/v8in again python $TW/$trn $TW/$tcfg
      else log "fine-tune member $n"; again python $S/train_self.py $TW/$tcfg; fi
      mkdir -p $OUT/train/ckpt_$n && cp -r $P3/m306/self/runs/$n/final $OUT/train/ckpt_$n/ 2>/dev/null || true; done
    STEPS=${STEPS//train/}; fi
  if [ -n "$MEMB" ] && [[ ",$STEPS," == *,leadA,* ]]; then
    for row in $MEMB; do n=${row%%:*}; CK=$OUT/train/ckpt_$n/final/model.safetensors; [ -s $CK ] || CK=$FL/third_party/v8in/finalist/$n.safetensors
      again python $P3/m306/eval/code/render_leadA.py --ckpt $CK --tta 1 --out $OUT/train/leadA/fwd_$n.npy
      again python $P3/m306/eval/code/render_leadA.py --ckpt $CK --tta 1 --render $TW/x_req/20261020/A10/v4c_work/shuf_render.zarr --region box --out $OUT/train/leadA/shuf_$n.npy; done
    MEM=$(for row in $MEMB; do echo -n "$OUT/train/leadA/fwd_${row%%:*}.npy "; done); SHM=$(for row in $MEMB; do echo -n "$OUT/train/leadA/shuf_${row%%:*}.npy "; done)
    python $C/fuse_v7.py $OUT/train/fused --trace $SHIPPED_MAPS/fwd_tr.npy --members $MEM --shuf-trace $FL/data/members/shuf_tr.npy --shuf-members $SHM
    python -c "
import sys, numpy as np
a, b = np.load(sys.argv[1]).astype(np.float32), np.load(sys.argv[2]).astype(np.float32); m = np.isfinite(a) & np.isfinite(b)
print(f'new TRACE x A vs shipped: r {np.corrcoef(a[m][::7], b[m][::7])[0, 1]:.4f}, mean |diff| {np.abs(a[m] - b[m]).mean():.5f}, px {int(m.sum())}')
" $OUT/train/fused/fwd_prod.npy $SHIPPED_MAPS/fwd_prod.npy
    STEPS=${STEPS//leadA/}; fi
  if [[ ",$STEPS," == *,train,* ]]; then log "fine-tune $RUN"
    if [ $LANE = self ]; then again python $S/train_self.py $TC; R=$P3/m306/self/runs/$RUN; else again python $P3/m306/train/train_v8306.py $TC; R=$P3/m306/train/runs/$RUN; fi
    mkdir -p $OUT/train/ckpt && cp -r $R/final $R/train_log.jsonl $R/done.json $OUT/train/ckpt/ 2>/dev/null || true
    log "new final sha256 $(sha256sum $R/final/model.safetensors | cut -c1-12) (shipped ${FIN_SHA:0:12}; GPU kernels may differ bit for bit)"; fi
  if [[ ",$STEPS," == *,leadA,* ]]; then log "claim map of the trained model (TTA 4) on the V5Rc render"
    CK=$OUT/train/ckpt/final/model.safetensors; [ -s $CK ] || CK=$FIN_W
    again python $P3/m306/eval/code/render_leadA.py --ckpt $CK --tta 4 --out $OUT/train/leadA/fwd_fin.npy
    python -c "
import sys, numpy as np
a, b = np.load(sys.argv[1]).astype(np.float32), np.load(sys.argv[2]).astype(np.float32); m = np.isfinite(a) & np.isfinite(b)
print(f'new claim map vs shipped: r {np.corrcoef(a[m][::7], b[m][::7])[0, 1]:.4f}, mean |diff| {np.abs(a[m] - b[m]).mean():.4f}, px {int(m.sum())}')
" $OUT/train/leadA/fwd_fin.npy $SHIPPED_MAPS/fwd_fin.npy
  fi
  log "train done: $OUT/train" ;;

shell) exec bash "$@" ;;
*) echo "usage: run.sh [final | pipeline | smoke | exclusion | train | shell]  (see the header of /opt/fl/claim/run.sh)"; exit 2 ;;
esac
