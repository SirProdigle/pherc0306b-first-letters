#!/bin/bash
# build_surface.sh OUT [FROM] : how the claim mesh (CT-driven touch-up of the lead-A GrowPatch, re-flattened) is built.
# Lane P1's code (copied by lane I2 with portable paths; FL_ROOT = repo root, SURF_WORK = work dir for chunk caches and eval/).
# CPU only, niced, 4 threads.  No ink map, label or tracing is read by any step.
#
#   FROM=tail (default)  steps 4-5 from the shipped intermediate arrays (inputs/): seconds, byte-identical to the claim mesh
#   FROM=all             steps 1-5 from the surfaces and the volume (needs the inputs listed under step 1; CT and the m7 surface
#                        prediction stream from the public open-data bucket or a local chunk cache)
#
# 1. carrier.py --out OUT/carrier --stage carrier        claim-frame carrier grid (u -36..+31, v -21..+21 mm, 20 vx cells) from
#      the lead-A GrowPatch (in-plane parametrisation) and the larger own-sheet GrowPatch w074big, smoothed 0.5 mm; crossings of
#      every reference surface with the carrier normals (A, w074big, two more GrowPatches R1 / L1, the spiral-fit lanes z8400 /
#      z8400o windings w072-w076; their tifxyz are our own cloud outputs, listed in p1geom.SURF)
#    prof_par: carrier.py --stage prof --rows r0,r1 (x3), --stage merge    CT profiles along the normals (t = -60..+60 vx, 3x3 footprint)
#    m7prof.py OUT/carrier                                m7 surface-prediction profiles (published m7-L0-th0.2 zarr), same nodes
# 2. track.py --carrier OUT/carrier --out OUT             -> track.npz: V0_orig = A's (else w074big's) crossing offsets = the
#      original surface on the carrier grid (also the CT-tracked arms V1/V2, which were rejected)
# 3. track.py --carrier OUT/carrier --out OUT --prof prof_m7.npy --minprom 5 --seed-target 0 --tag _m7   -> track_m7.npz
#      (V2_trackP = the m7-tracked sheet, "V3");  touchup.py computes x_z99_w074.npy (lane z9900 w074 crossings) on first use
# 4. touchup.py CARRIER TRACK V5R_rightend 12,30,-4,4 OUT        the consensus touch-up, kept only in the pre-registered region
#      "line row right" (u +12..+30, |v| <= 4 mm): one component, 85.9 mm2  -> OUT/V5R_rightend (tifxyz) + track_V5R_rightend.npz
# 5. reflatten.py OUT/V5R_rightend OUT/V5Rf_ctx --region=-31,31,-21,16 --anchor=-10,8,-2.5,2.5 --iters 60 --fold-weight 0.02 --fold-thr 1.25
#      fold-weighted ARAP re-flattening -> OUT/V5Rf_ctx = the claim mesh (x/y/z.tif)
# Checks: adopt_check.py OUT/track_V5R_rightend.npz CARRIER TRACK OUT (sheet identity, seam); recto.py NAME=RENDER.zarr (recto score
#   on a render of the mesh); metric.py MESH (distortion).
set -euo pipefail
OUT=${1:?usage: build_surface.sh OUT [tail|all]}; FROM=${2:-tail}
HERE=$(cd "$(dirname "$0")" && pwd); PY=${PY:-python}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 SURF_WORK=${SURF_WORK:-$OUT}
mkdir -p $OUT/eval
if [ $FROM = all ]; then
  nice -n 15 $PY $HERE/carrier.py --out $OUT/carrier --stage carrier
  for r in 0,82 82,164 164,245; do nice -n 15 $PY $HERE/carrier.py --out $OUT/carrier --stage prof --rows $r & done; wait
  nice -n 15 $PY $HERE/carrier.py --out $OUT/carrier --stage merge
  nice -n 15 $PY $HERE/m7prof.py $OUT/carrier
  nice -n 15 $PY $HERE/track.py --carrier $OUT/carrier --out $OUT
  nice -n 15 $PY $HERE/track.py --carrier $OUT/carrier --out $OUT --prof prof_m7.npy --minprom 5 --seed-target 0 --tag _m7
  C=$OUT/carrier; T=$OUT
else
  C=$HERE/inputs; T=$HERE/inputs                               # carrier.npz, track.npz, track_m7.npz, x_z99_w074.npy
fi
nice -n 15 $PY $HERE/touchup.py $C $T V5R_rightend 12,30,-4,4 $OUT
nice -n 15 $PY $HERE/reflatten.py $OUT/V5R_rightend $OUT/V5Rf_ctx --region=-31,31,-21,16 --anchor=-10,8,-2.5,2.5 --iters 60 --fold-weight 0.02 --fold-thr 1.25
nice -n 15 $PY $HERE/adopt_check.py $OUT/track_V5R_rightend.npz $C $T $OUT/eval
echo "mesh: $OUT/V5Rf_ctx"; (cd $OUT/V5Rf_ctx && sha256sum x.tif y.tif z.tif)
