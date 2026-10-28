#!/bin/bash
# make_context.sh CTX : assemble the Docker build context of the claim image, version 7 (fl-claim-a7).
# Version 5's context (v1 pins, VC3D AppImage, zlib-ng, ink8um loader + v8-in base + v8Q weights, Reader V3, the package mesh, every
# shipped map, every view PNG + expected.sha256), with these version-7 changes:
#   code/claim/config_v7.json        the package's build config (registry, models incl. the finalist 'fin', pairs, box)
#   data/maps/fwd_<model>.npy        copied from the config's own model map paths (exactly the files the build read), fwd_fin.npy incl.
#   finalist/model.safetensors       the version-7 model's weights + finalist/SHA256SUMS (checked in the build)
#   data/smoke/                      a 24-layer CT crop (169 x 169 px inside the claim box) for the CPU smoke test (smoke_v7.py)
#   train/W/...                      the training code at the paths it ran from, the pool and region registries, the exclusion zone and
#                                    check code, the pool meshes, the training tile list (mode 'exclusion'), the training configs
#   train/INPUTS.tsv                 every training input the 'train' mode needs mounted (path under W, sha256 or manifest)
# Inputs (env): PKG (the assembled package; default this docker/ dir's package), SMOKE_CROP (default the v6 kit's crop), V1_DOCKER.
set -euo pipefail
CTX=${1:?usage: make_context.sh CTX}
HERE=$(cd "$(dirname "$0")" && pwd); CODE=$(cd "$HERE/../code" && pwd)
W=${FL_DATA:-/home/liam/projects/vesuvius-work}; P3=$W/.claude/worktrees/leadA-push3/x_push3
PKG=${PKG:-$CODE/..}
CFG=$CODE/config_v7.json
[ -s "$CFG" ] || { echo "no $CFG: assemble the package with build_v7.py first"; exit 1; }
V1=${V1_DOCKER:-$W/x_submit/docker_final}
MESH=PHerc0306B_20250521133212_w074own_touchup_V5Rc
PY=${PY:-$W/envs/vcpy/bin/python}
rm -rf "$CTX"; mkdir -p "$CTX"/{reader-v3,code/claim,code/x_zoom,code/x_research2/speed,mesh,vendor,zlib-ng,v8in/ink8um,data/maps,data/final,data/smoke,data/members,finalist,train/W}
cp "$HERE"/Dockerfile "$CTX"/; cp "$V1"/{constraints.txt,requirements.txt} "$CTX"/; cp "$HERE"/checkpoints.tsv "$CTX"/
cp "$V1"/vendor/libz.so.1.zlib-ng-2.3.3 "$CTX"/zlib-ng/libz.so.1
cp "$V1"/code/x_zoom/zoomlib.py "$CTX"/code/x_zoom/; cp "$V1"/code/x_research2/speed/fastink.py "$CTX"/code/x_research2/speed/
cp "$HERE"/run.sh "$CODE"/{make_claim_image_v2.py,claimlib_v2.py,ink_v2.py,ink_v3.py,ink_v8.py,compare_v2.py,build_variants.py,config_v7.json,smoke_v7.py,exclusion_audit_v7.py,fuse_v7.py,controls_panel_v7.py,controls_v5.py} "$CTX"/code/claim/
cp "$W"/third_party/v8in/ink8um/{__init__.py,modeling.py,inference.py} "$CTX"/v8in/ink8um/
cp "$W"/third_party/v8in/{config.json,model.safetensors} "$CTX"/v8in/
cp "${V8Q_CKPT:-$W/x_refine/trials/P2_inkft_0306B/ckpt/v8Q/final.safetensors}" "$CTX"/v8in/v8Q_final.safetensors
cp "${READER_V3_CKPT:-$W/models/community/reader-v3/reader-v3.pth}" "$CTX"/reader-v3/reader-v3.pth
cp -r "$PKG"/mesh/$MESH "$CTX"/mesh/
# every map the config names (the build's own inputs) + the CT layer; the finalist's weights
"$PY" - "$CFG" "$CTX" << 'PYEOF'
import json, sys, shutil, hashlib, os
cfg, ctx = json.load(open(sys.argv[1])), sys.argv[2]
used = sorted({m for v in cfg['views'] for m in cfg['pairs'][v['pair']]})
for k in used:
    shutil.copy2(cfg['models'][k]['map'], f'{ctx}/data/maps/fwd_{k}.npy')
    if cfg['models'][k].get('crop'): json.dump(dict(box=cfg['models'][k]['crop']), open(f'{ctx}/data/maps/fwd_{k}.json', 'w'))
shutil.copy2(cfg['ct'], f'{ctx}/data/maps/ct20.png')
F = cfg['finalist']; D = '/opt/fl/third_party/v8in/finalist'
def put(src, name, want):
    h = hashlib.sha256(open(src, 'rb').read()).hexdigest()
    if want and h != want: raise SystemExit(f'{src}: sha256 {h} != {want}')
    shutil.copy2(src, f'{ctx}/finalist/{name}'); os.chmod(f'{ctx}/finalist/{name}', 0o644); return h
fj = dict(name=F['name'], display=F['display'], tta=F.get('expect', {}).get('tta', 4), lane=F['lane'], dry_run=cfg.get('dry_run', False),
          fusion=F.get('fusion', 'zmean'), train_config=os.path.basename(F['train_config']))
if F.get('fusion') == 'product':                     # every member's weights, maps and shuffle maps; the decoy; TRACE's shuffle map
    os.makedirs(f'{ctx}/data/members', exist_ok=True); fj['members'] = []; sums = []
    for m in F['members']:
        h = put(m['ckpt'], f'{m["name"]}.safetensors', m['sha256']); sums.append(f'{h}  {D}/{m["name"]}.safetensors')
        shutil.copy2(m['map'], f'{ctx}/data/members/fwd_{m["name"]}.npy'); shutil.copy2(m['shuf_map'], f'{ctx}/data/members/shuf_{m["name"]}.npy')
        fj['members'].append(dict(name=m['name'], seed=m['seed'], file=f'{m["name"]}.safetensors', sha256=h))
    shutil.copy2(F['decoy']['map'], f'{ctx}/data/members/fwd_decoy.npy'); shutil.copy2(F['trace_shuf_map'], f'{ctx}/data/members/shuf_tr.npy')
    for k in ('shuf_prod', 'shuf_ensA'): shutil.copy2(f'{F["fused_dir"]}/{k}.npy', f'{ctx}/data/members/{k}.npy')
    fj['decoy'] = dict(name=F['decoy']['name'], sha256=F['decoy']['sha256'])
    open(f'{ctx}/finalist/SHA256SUMS', 'w').write('\n'.join(sums) + '\n'); h = 'product of ' + ', '.join(x['sha256'][:12] for x in fj['members'])
else:
    c = F['ckpts'][0]; h = put(c['path'], 'model.safetensors', c.get('sha256'))
    open(f'{ctx}/finalist/SHA256SUMS', 'w').write(f'{h}  {D}/model.safetensors\n'); fj['sha256'] = h
json.dump(fj, open(f'{ctx}/finalist/finalist.json', 'w'), indent=1)
print('maps', used, 'finalist', F['name'], h[:60])
PYEOF
chmod 644 "$CTX"/v8in/*.safetensors "$CTX"/v8in/*.json "$CTX"/reader-v3/reader-v3.pth
cp "${SMOKE_CROP:-$P3/v7/smoke/v5rc_crop_y2110_x4495_n5.npz}" "$CTX"/data/smoke/crop.npz
# every display variant of the package, by its registry file name (the lead's from <MESH>.png): data/final/variants/ + expected.sha256 + LEAD
mkdir -p "$CTX"/data/final/variants
LEAD=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['variant']['key'])" "$PKG"/$MESH.json); echo $LEAD > "$CTX"/data/final/LEAD
"$PY" - "$PKG" "$CTX" "$MESH" "$CODE" "$CFG" << 'PYEOF'
import sys, shutil, importlib.util
pkg, ctx, mesh, code, cfg = sys.argv[1:6]
spec = importlib.util.spec_from_file_location('bv', f'{code}/build_variants.py'); bv = importlib.util.module_from_spec(spec); spec.loader.exec_module(bv)
lead = open(f'{ctx}/data/final/LEAD').read().strip()
for k, v in bv.registry(cfg)[0].items():
    src = f'{pkg}/{mesh}' if k == lead else f'{pkg}/supplementary/{v["file"]}'
    for ext in ('png', 'json'): shutil.copy2(f'{src}.{ext}', f'{ctx}/data/final/variants/{v["file"]}.{ext}')
PYEOF
(cd "$CTX"/data/final && sha256sum variants/*.png) > "$CTX"/data/final/expected.sha256
# version 7: the training reproduction tree (code at its original paths under W; data is mounted at run time, train/INPUTS.tsv)
"$PY" "$HERE"/train_tree.py "$CFG" "$PKG" "$CTX"/train
( cd "$CTX" && sha256sum data/maps/* data/smoke/* mesh/$MESH/* $(ls data/members/* 2>/dev/null) ) > "$CTX"/data/inputs.sha256
( cd "$CTX"/code && sha256sum claim/* x_zoom/zoomlib.py x_research2/speed/fastink.py ) >> "$CTX"/data/inputs.sha256
AI=${VC3D_APPIMAGE:-$W/x_req/20261020/A10/v4c_work/docker_ctx/vendor/VC3D.AppImage}
if [ "${FULL:-0}" = 1 ] && [ -s "$AI" ]; then cp "$AI" "$CTX"/vendor/VC3D.AppImage; fi      # only the self-contained build needs it
wc -l "$CTX"/data/final/expected.sha256; du -sh "$CTX"
