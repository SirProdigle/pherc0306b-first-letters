"""train_tree.py CONFIG_V7.json PKG OUT : the training-reproduction part of the version-7 Docker context.

OUT/W/<path under W>   the training code exactly as it ran (byte-identical copies), the registries it reads (regions, renders,
                       pool regions), the exclusion zone + check code, the pool meshes (small tifxyz), the lane's configs.
OUT/INPUTS.tsv         every training input the image does NOT carry (CT renders, labels of other scrolls): kind, path under W,
                       files, bytes, sha256 (files) -- mount each at /home/liam/projects/vesuvius-work/<path> (read only).
OUT/links.tsv          symlinks the image creates under W to files it already has (model weights, villa helpers).
OUT/exclusion/         the package's tile list, tile regions, audit and the meshes (mode 'exclusion' re-verifies the tile list).
OUT/finalist_train.json  the lane, the configs, the chain of steps (read by run.sh mode 'train')."""
import os, sys, json, shutil, hashlib, glob
W = '/home/liam/projects/vesuvius-work'; P3 = f'{W}/.claude/worktrees/leadA-push3/x_push3'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()


def put(src, out):
    rel = os.path.relpath(src, W); d = f'{out}/W/{rel}'; os.makedirs(os.path.dirname(d), exist_ok=True)
    if os.path.isdir(src): shutil.copytree(src, d, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
    else: shutil.copy2(src, d)
    return rel


def du(p):
    if os.path.isfile(p): return 1, os.path.getsize(p)
    n = b = 0
    for r, _, fs in os.walk(p):
        for f in fs: n += 1; b += os.path.getsize(os.path.join(r, f))
    return n, b


def main():
    cfg_path, pkg, out = sys.argv[1:4]
    cfg = json.load(open(cfg_path)); F = cfg['finalist']; lane = F['lane']
    os.makedirs(f'{out}/W', exist_ok=True)
    code = [f'{P3}/aclick/p306/exclusion/check.py', f'{P3}/aclick/p306/exclusion/build_zone.py', f'{P3}/aclick/p306/exclusion/zone.json',
            f'{P3}/aclick/p306/exclusion/zone.npz', f'{P3}/aclick/p306/exclusion/renders.json', f'{P3}/aclick/bench/infer.py',
            f'{P3}/m306/eval/code/render_leadA.py', f'{P3}/m306/train/v8306lib.py', f'{P3}/m306/train/train_v8306.py', F['train_config']]
    if lane == 'self':
        code += [f'{P3}/m306/self/code/{f}' for f in ('pool.py', 'masks.py', 'teachers.py', 'labels.py', 'train_self.py')]
        code += [f'{W}/x_req/20261020/V3/run_maps.py', f'{W}/x_research2/speed/fastink.py', f'{P3}/aclick/p306/regions.json',
                 f'{P3}/m306/self/pool/regions_pool.json']
        R = json.load(open(f'{P3}/m306/self/pool/regions_pool.json')); R.pop('_note', None)
        sys.path.insert(0, f'{P3}/aclick/p306/exclusion'); import check
        meshes, crops = [], []
        for r, g in R.items():
            m, off = check.resolve(g['zarr']); meshes.append(m)
            cj = f'{os.path.realpath(g["zarr"])}/crop.json'
            if os.path.exists(cj): crops.append(cj)
        code += meshes + crops
        if F.get('fusion') == 'product':                                           # every member's (and the decoy's) config; the cloud trainer
            code += [m['train_config'] for m in F['members'] + [F['decoy']]]
            code += [f'{P3}/m306/self/code/{f}' for f in ('train_cloud.py', 'make_packs.py')]
            code += sorted({m['trainer_path'] for m in F['members'] + [F['decoy']] if m.get('trainer_path')})   # version 7: hard-negative trainer
            code += [p for p in F.get('extra_training_code', []) if p not in code]
    rels = [put(p, out) for p in code if os.path.exists(p)]
    missing = [p for p in code if not os.path.exists(p)]
    if missing: raise SystemExit(f'train_tree: missing {missing}')
    # inputs the image does not carry (mounted read-only at the same path)
    inp = []
    if lane == 'self':
        for g in R.values():
            z = os.path.realpath(g['zarr']); inp.append(('zarr', z))
        inp += [('dir', f'{W}/data/pherc1447/w058'), ('dir', f'{W}/data/pherc1447/w060'), ('file', f'{W}/data/pherc1447/SHA256SUMS')]
        if any(m.get('trainer_path') for m in F.get('members', [])):                  # the hard-negative trainer reads the zone-checked packs
            inp.append(('dir', f'{P3}/m306/self/packs'))
    else:
        tc = json.load(open(F['train_config']))
        for s in tc['sources']:
            if s['kind'] == '1447': inp.append(('dir', f'{W}/data/pherc1447/{s["winding"]}'))
            elif s['kind'] == '0343': inp += [('file', f'{W}/x_px343/label_render.npy'), ('file', f'{W}/x_px343/mask_render.npy'), ('zarr', f'{W}/fl/0343/concat_66.zarr')]
            elif s['kind'] == 'human_v2':
                tj = f'{P3}/aclick/data/labels_train/{s["object"].replace("PHerc", "")}_{s["segment"]}.json'
                inp += [('file', tj), ('file', f'{P3}/aclick/data/labels_registered/{s["segment"]}_human_v2.npz'),
                        ('zarr', json.load(open(tj))['native_zarr'])]
        inp.append(('file', f'{W}/data/pherc1447/SHA256SUMS'))
    inp += [('zarr', f'{W}/x_refine/trials/P1_surface_refine/renders/P1V5Rc.zarr'), ('zarr', f'{W}/x_req/20261020/A10/v4c_work/shuf_render.zarr')]
    rows = ['# kind\tpath under W (mount read-only at /home/liam/projects/vesuvius-work/<path>)\tfiles\tbytes\tsha256 (files only)']
    seen = set()
    for k, p in inp:
        if p in seen: continue
        seen.add(p); n, b = du(p) if os.path.exists(p) else (0, 0)
        rows.append(f'{k}\t{os.path.relpath(p, W)}\t{n}\t{b}\t{sha(p) if k == "file" and os.path.exists(p) else "-"}')
    open(f'{out}/INPUTS.tsv', 'w').write('\n'.join(rows) + '\n')
    links = ['# path under W\ttarget in the image',
             'third_party/v8in\t/opt/fl/third_party/v8in',
             'models/community/reader-v2\t/opt/fl/models/community/reader-v2',
             'models/community/reader-v3\t/opt/fl/models/community/reader-v3',
             'models/community/trace\t/opt/fl/models/community/trace']
    open(f'{out}/links.tsv', 'w').write('\n'.join(links) + '\n')
    X = f'{out}/exclusion'; os.makedirs(X, exist_ok=True)
    MD = f'{pkg}/methodology/model_v7'
    for f in ('exclusion_audit.json', 'training_tiles.npz', 'tile_regions.json'):
        if os.path.exists(f'{MD}/{f}'): shutil.copy2(f'{MD}/{f}', f'{X}/{f}')
    if os.path.isdir(f'{MD}/exclusion/meshes'): shutil.copytree(f'{MD}/exclusion/meshes', f'{X}/meshes', dirs_exist_ok=True)
    for f in ('check.py', 'zone.npz', 'zone.json'): shutil.copy2(f'{P3}/aclick/p306/exclusion/{f}', f'{X}/{f}')
    steps = (['pool', 'teachers', 'labels', 'exclusion', 'train', 'leadA'] if lane == 'self' else ['exclusion', 'train', 'leadA'])
    if F.get('fusion') == 'product': steps = ['pool', 'teachers', 'labels', 'exclusion', 'train', 'leadA']
    json.dump(dict(lane=lane, name=F['name'], train_config=os.path.relpath(F['train_config'], W), run_name=json.load(open(F['train_config']))['name'],
                   ckpt_sha256=F['ckpts'][0].get('sha256'), steps=steps, code=rels, P3=os.path.relpath(P3, W), fusion=F.get('fusion', 'zmean'),
                   members=[dict(name=m['name'], train_config=os.path.relpath(m['train_config'], W), sha256=m['sha256'], seed=m['seed'],
                                 trainer=os.path.relpath(m['trainer_path'], W) if m.get('trainer_path') else None)
                            for m in F.get('members', [])]),
              open(f'{out}/finalist_train.json', 'w'), indent=1)
    print(f'train tree: {len(rels)} files, {len(seen)} inputs to mount, steps {steps}')


if __name__ == '__main__':
    main()
