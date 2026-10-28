#!/bin/bash
# erwin_data.sh SCROLL Z0 Z1 SENSE UMBILICUS_JSON : dataset ds/SCROLL for the old fitter (Erwin's PHerc0343 recipe):
#   tracks (+ placeholder .dbm), lasagna nx/ny/grad_mag group 2 (band), m7 level 0 (band, for snapping) + level 1 (band, SDT),
#   CT level 4 (whole) + level 1 (band, ct-zero), surf-SDT (m7 L1, threshold 128, z band +-64, CT L4 occupancy dilated 1),
#   resident-pool sidecars, old-format spiral-scroll.json.
set -euo pipefail
W=/home/liam/projects/vesuvius-work; X=$W/x_recipe_erwin; S=$1; Z0=$2; Z1=$3; SENSE=$4; UMB=$5
DS=${DSROOT:-$W/ds}; SP=${SPIRAL_DIR:-$W/x_recipe_erwin/spiral_old}; SPY=${SPIRAL_PY:-$W/envs/spiral_old/bin/python}   # local runs override these
export PATH=$W/envs/vcpy/bin:$PATH AWS_CONFIG_FILE=$W/fl/aws_fast.cfg
read VID UM ZS YS XS SURF VOLK < <(python3 -c "
import json;c=json.load(open('$W/fl/scrolls.json'))['$S'];v=c['volume_url'].rstrip('/').split('/')[-1]
print(v.split('-')[0], c['voxel_um'], *c['shape'], c['surface_url'].split('amazonaws.com/')[1], c['volume_url'].split('amazonaws.com/')[1])")
D=$DS/${DSNAME:-$S}; M=$DS/m7_$S; C=$DS/ct_$S; mkdir -p $D/tracks $D/lasagna_inputs $M $C   # DSNAME: own dataset dir for a 2nd band
cp $UMB $D/umbilicus.json
T=https://dl.ash2txt.org/datasets/spiral_datasets/$S/$VID/tracks; P=${S}_${VID}_surface_m7_L0_th0.2
if [ -z "${NOTRACKS:-}" ]; then   # NOTRACKS=1: tracks are extracted locally (scroll without published tracks)
cd $D/tracks; mkdir -p $P.dbm.vctracks
for f in arclengths.f64 coordinates.i32 family_codes.i8 header.bin metadata.json offsets.i64 source_ids.u64 tortuosities.f64 z_bounds.i32; do
  [ -s $P.dbm.vctracks/$f ] || curl -sf -o $P.dbm.vctracks/$f $T/$P.dbm.vctracks/$f &
done
[ -s $P.dbm.crossings.npz ] || curl -sf -o $P.dbm.crossings.npz $T/$P.dbm.crossings.npz &
wait
python3 - <<PY
import json, os, urllib.request
m = json.load(open('$P.dbm.vctracks/metadata.json')); name, size, mt = m['source_db_signature'][0]
if not os.path.exists(name):
    req = urllib.request.Request('$T/' + name, headers={'Range': 'bytes=0-65535'}); head = urllib.request.urlopen(req).read()
    with open(name, 'wb') as f: f.write(head); f.truncate(size)
    os.utime(name, ns=(mt, mt)); print('placeholder', name, size, len(head))
PY
[ -s $P.dbm.vctracks/z_bounds.i32 ] || python3 - <<PY
import json, numpy as np
d='$P.dbm.vctracks'; m=json.load(open(d+'/metadata.json')); n=m['track_count']; p=m['point_count']
off=np.memmap(d+'/offsets.i64',mode='r',dtype='<i8',shape=(n+1,)); c=np.memmap(d+'/coordinates.i32',mode='r',dtype='<i4',shape=(p,3))
zb=np.empty((n,2),'<i4'); B=2_000_000
for i in range(0,n,B):
    j=min(n,i+B); s=off[i:j]; z=np.asarray(c[off[i]:off[j],0]); st=(s-off[i]).astype(np.int64)
    zb[i:j,0]=np.minimum.reduceat(z,st); zb[i:j,1]=np.maximum.reduceat(z,st)
zb.tofile(d+'/z_bounds.i32'); print('z_bounds written')
PY
fi
# lasagna group 2 (4x down, chunks 32 = 128 working) for the band +-1 chunk
LP=$( (aws s3 ls --no-sign-request s3://vesuvius-challenge-open-data/$S/representations/predictions/lasagna/ || true) | awk '{print $2}' | head -1 | tr -d /)
[ "${NORMALS:-lasagna}" = syn ] && LP=""
if [ -n "$LP" ]; then
  L=s3://vesuvius-challenge-open-data/$S/representations/predictions/lasagna/$LP
  for ch in nx ny grad_mag; do
    src=$L/${S}_${ch}.ome.zarr; dst=$D/lasagna_inputs/las_008_${ch}.ome.zarr; mkdir -p $dst/2
    for f in .zattrs .zgroup 2/.zarray; do aws s3 cp --no-sign-request --only-show-errors $src/$f $dst/$f; done
    for zi in $(seq $((Z0/128-1)) $((Z1/128+1))); do aws s3 sync --no-sign-request --only-show-errors $src/2/$zi $dst/2/$zi & done; wait
  done
else
  echo "no Lasagna used for $S: synthetic normals from m7 (after the m7 download)"; touch $D/SYN_NORMALS
fi
# m7: level 0 band +-1 chunk (snap), level 1 over the SDT tile windows
MS=s3://vesuvius-challenge-open-data/$SURF
for f in .zattrs .zgroup 0/.zarray 1/.zarray; do aws s3 cp --no-sign-request --only-show-errors $MS/$f $M/$f; done
for zi in $(seq $((Z0/192-1)) $((Z1/192+1))); do aws s3 sync --no-sign-request --only-show-errors $MS/0/$zi $M/0/$zi & done; wait
read A B < <(python3 -c "print(((($Z0-64)//2)//256*256-64)//192, ((($Z1+64)//2+255)//256*256+64)//192)")
for zi in $(seq $A $B); do aws s3 sync --no-sign-request --only-show-errors $MS/1/$zi $M/1/$zi & done; wait
# CT: level 4 whole, level 1 over the SDT tile windows (chunks 128)
CS=s3://vesuvius-challenge-open-data/$VOLK
for f in .zattrs .zgroup 1/.zarray 4/.zarray; do aws s3 cp --no-sign-request --only-show-errors $CS/$f $C/$f; done
aws s3 sync --no-sign-request --only-show-errors $CS/4 $C/4
read A B < <(python3 -c "print(((($Z0-64)//2)//256*256-64)//128, ((($Z1+64)//2+255)//256*256+64)//128)")
for zi in $(seq $A $B); do aws s3 sync --no-sign-request --only-show-errors $CS/1/$zi $C/1/$zi & done; wait
if [ -f $D/SYN_NORMALS ] && [ ! -f $D/lasagna_inputs/.syn_done ] && [ -z "${SKIP_NORMALS:-}" ]; then   # SKIP_NORMALS: CPU/download-only prep
  $W/envs/vcpy/bin/python $X/tools/make_normals.py $M $D/lasagna_inputs --z $((Z0-128)) $((Z1+128)) --tile 512 > $D/make_normals.log 2>&1 && touch $D/lasagna_inputs/.syn_done
  tail -1 $D/make_normals.log
fi
du -sh $D $M $C
# old-format scroll spec (0fb38c45 rejects unknown keys)
cat > $D/spiral-scroll.json <<JS
{"schema_version": 1, "name": "$S", "voxel_size_um": $UM, "spiral_outward_sense": "$SENSE",
 "normal_zarr_group": "2", "surf_sdt_zarr_group": "1", "lasagna_scale": 4, "paths": {"tracks_dbm": "tracks/$P.dbm"}}
JS
# surf-SDT (Erwin: m7 level 1, threshold 128, z band +-64, CT occupancy at level 4 dilated by 1)
PY=$SPY
OUT=$D/lasagna_inputs/las_008_surf_sdt.ome.zarr
if [ ! -f $OUT/.sdt_done ]; then
  (cd $SP && $PY make_surf_sdt.py --surf $M --group 1 --out $OUT --threshold 128 --working-voxel-um $UM \
     --z-range $((Z0-64)):$((Z1+64)) --ct $C --ct-group 4 --ct-dilate 1 --workers ${SDT_WORKERS:-16} --edt-threads 2 $([ -d $OUT ] && echo --resume)) && touch $OUT/.sdt_done
fi
python3 -c "import json;a=json.load(open('$OUT/.zattrs'));print({k:a[k] for k in a if k!='multiscales'})" | head -c 2000; echo
# resident-pool sidecars (what the fitter would build itself; no CT masking)
WHAT=sdt,normals; [ -f $D/SYN_NORMALS ] || WHAT=sdt,normals,grad_mag
[ -n "${SKIP_NORMALS:-}" ] && { echo "prep only: normals + pools later"; exit 0; }
(cd $SP && $PY pack_resident_pools.py $D/lasagna_inputs --what $WHAT --sdt-group 1 --normal-group 2)
ls $D/lasagna_inputs
echo DATA_OK
