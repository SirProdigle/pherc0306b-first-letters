#!/bin/bash
# set_lead.sh VARIANT [MAPS_DIR] : make VARIANT (rt_primary | rt_secondary | rv8q_secondary | rv8q_grey = views 1 / 2 / 4 / 5) the main image
# of the v2 DRAFT package and redo everything that depends on it (lane I2).  ~2-3 min, CPU:
#   1. build_variants.py --lead VARIANT: <MESH>.png = VARIANT, every other variant in supplementary/ (the old lead's file comes back
#      there; the new lead's supplementary copy is removed, listed in DELETED_manifest.txt)
#   2. letter_table_v2.py on the new main image (measurement only)
#   3. assemble_v2.py (README filled for the new lead, SHA256SUMS)
#   4. the Docker image rebuilt with the new lead layout (offline, Dockerfile.from_v1) and test (a) re-run: MATCH required
# MAPS_DIR: fwd_rv2 / fwd_tr / fwd_v8 / fwd_v8q.npy + ct20.png (default: the owner's staging dir of this lane).
set -euo pipefail
SCR=/tmp/claude-1000/-home-liam-projects-vesuvius-work/7d89d68e-474d-4112-ba44-511303204b24/scratchpad
V=${1:?usage: set_lead.sh rt_primary|rt_secondary|rv8q_secondary|rv8q_grey [MAPS_DIR]}
W=/home/liam/projects/vesuvius-work; C=$W/x_refine/I2_package; T=$W/x_refine/trials/I2_package; PKG=${PKG:-$T/SUBMISSION_v2_DRAFT_owner_only}
TAG=${TAG:-fl-claim-a2:draft}; CTXD=${CTXD:-$SCR/docker_ctx}; MANIFEST=${MANIFEST:-$T/DELETED_manifest.txt}   # PKG / TAG / CTXD / MANIFEST: dry runs on a copy
MAPS=${2:-$SCR/maps_v5rc}; MESH=PHerc0306B_20250521133212_w074own_touchup_V5Rc; PY="nice -n 15 $W/envs/vcpy/bin/python -I"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
F=$($PY -c "import sys; sys.path.insert(0, '$C'); import build_variants as b; print(b.VARIANTS['$V']['file'])")
$PY $C/build_variants.py --maps $MAPS --ct $MAPS/ct20.png --mesh $PKG/mesh/$MESH --out $PKG --lead $V
for e in png json; do f=$PKG/supplementary/$F.$e; [ -f $f ] || continue
  echo "$(date '+%F %T')  $(stat -c %s $f) B  $f  sha256 $(sha256sum $f | cut -c1-64)  reason: set_lead.sh $V (now <MESH>.$e)" >> $MANIFEST; rm $f; done
$PY $C/letter_table_label3.py $PKG $W/x_followup/A_label/papyrologist_label3_conservative_2026-10-16.png > /dev/null
$PY $C/mirror_control.py $PKG > /dev/null
$PY $C/compare_views.py $PKG > /dev/null
I2_PKG=$PKG $PY $C/assemble_v2.py $T/docker_test
PKG=$PKG bash $C/docker/build.sh $TAG $CTXD
O=$SCR/test_final_$V; rm -rf $O; mkdir -p $O
nice -n 15 docker run --rm --network none --cpus 4 --user $(id -u):$(id -g) -v $O:/out $TAG > $O/run.log 2>&1 && tail -2 $O/run.log
grep -q "MATCH" $O/run.log && echo "set_lead $V: done (package, letter table, README, SHA256SUMS, Docker image + test (a) MATCH)" || { echo "set_lead $V: Docker test FAILED"; exit 3; }
