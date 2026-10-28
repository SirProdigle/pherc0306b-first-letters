#!/bin/bash
# ink4.sh <surface.zarr> <n_layers> <d> <out_dir> [models...]  -- run 9um ink models at depth d, ensemble both sides
set -uo pipefail
W=/home/liam/projects/vesuvius-work
Z=$1; N=$2; D=$3; OUT=$4; shift 4
MODELS=${*:-"hybrid_3d2d-seed42 hybrid_3d2d-seed43 dnative hecate"}
export CKPT_DIR=$W/models/ink9_ens PY=$W/envs/vcpy/bin/python TMPDIR=$W/fl/tmp N_LAYERS=$N NATIVE_UM=${NATIVE_UM:-8.64}
export VILLA_BATCH=${VILLA_BATCH:-8} HEC_BATCH=${HEC_BATCH:-4}
mkdir -p "$OUT"
NAME=$(basename "$Z" .zarr)
for m in $MODELS; do
  bash $W/third_party/pherc343-first-letters/ink/scripts/run_ink.sh "$Z" $m $D "$OUT" > "$OUT/${NAME}__${m}__d$D.log" 2>&1 || echo "FAILED $m (see $OUT/${NAME}__${m}__d$D.log)"
done
S=$(( N / 2 + D - 8 )); ZC=$(awk -v n="$N" -v d="$D" 'BEGIN{printf "%.2f", (n - 1) / 2 + d}')
LAYER=$(( N / 2 + D ))
for SIDE in "" _reverse; do
  maps=()
  for m in $MODELS; do
    if [[ $m == hecate ]]; then f=$OUT/${NAME}__hecate__L${N}zc${ZC}${SIDE}.png; else f=$OUT/${NAME}__${m}__L${N}s${S}${SIDE}.tif; fi
    [[ -s $f ]] && maps+=("$f")
  done
  [[ ${#maps[@]} -gt 0 ]] && $PY $W/fl/scripts/ens_view.py "$OUT/${NAME}_d${D}${SIDE:-_fwd}" "$Z" $LAYER "${maps[@]}" --down ${DOWN:-2}
done
