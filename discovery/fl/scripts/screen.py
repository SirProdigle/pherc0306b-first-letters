"""First Letters screening: grow local GrowPatch patches across a scroll, render them,
run the 9 um ink models on both sides, and write ensemble overlays + a score per patch.

usage: screen.py SCROLL [--n N] [--gens G] [--models m1,m2] [--depths 0] [--zs z1,z2,...]
Seeds come from level-3 slices of the scroll's m7 surface prediction (cached as .npy).
Everything lives under fl/<SCROLL>/screen/. Safe to re-run: finished patches are skipped.
"""
import argparse, json, os, queue, subprocess, sys, threading, time, glob, random
from pathlib import Path
import numpy as np

W = Path('/home/liam/projects/vesuvius-work')
BIN = W / 'vc_build_main/bin'
PY = W / 'envs/vcpy/bin/python'
BUCKET = 'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com'
ENV = dict(os.environ, VC3D_CONFIG_DIR=str(W / 'fl/vc3dcfg'))

SCROLLS = json.load(open(W / 'fl/scrolls.json'))


def sh(cmd, log, timeout=None, env=ENV):
    with open(log, 'w') as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=timeout).returncode


def level3_slice(cfg, sdir, z):
    p = sdir / f'surf_z{z}_L3.npy'
    if not p.exists():
        rc = sh([str(PY), str(W / 'fl/scripts/xsec.py'), cfg['surface_url'], '3', str(sdir / 'surf'), str(z)], sdir / f'xsec_{z}.log')
        if rc != 0:
            raise RuntimeError(f'xsec failed z={z}')
    return np.load(p)


def pick_seeds(cfg, sdir, zs, n, spacing_l3=230, seed=0):
    rng = random.Random(seed)
    out = []
    per = max(1, n // len(zs) + 1)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda z: level3_slice(cfg, sdir, z), zs))
    for z in zs:
        a = level3_slice(cfg, sdir, z) > 0
        ys, xs = np.nonzero(a)
        if len(ys) == 0:
            continue
        cy, cx = ys.mean(), xs.mean()
        r = np.hypot(ys - cy, xs - cx)
        keep = r < np.percentile(r, 97)          # skip the outermost rim
        ys, xs = ys[keep], xs[keep]
        idx = list(range(len(ys)))
        rng.shuffle(idx)
        chosen = []
        for i in idx:
            if all((ys[i] - y) ** 2 + (xs[i] - x) ** 2 > spacing_l3 ** 2 for y, x in chosen):
                chosen.append((ys[i], xs[i]))
            if len(chosen) >= per:
                break
        out += [(int(x * 8 + 4), int(y * 8 + 4), int(z)) for y, x in chosen]
    rng.shuffle(out)
    return out[:n]


def grow_and_render(cfg, sdir, pid, seed, gens):
    pdir = sdir / 'patches' / pid
    pdir.mkdir(parents=True, exist_ok=True)
    if (pdir / 'r31.zarr' / '0' / '.zarray').exists():
        return pdir
    params = {"mode": "seed", "step_size": 20, "min_area_cm": 0.3, "generations": gens, "thread_limit": 0,
              "normal_grid_path": str(W / 'fl' / cfg['name'] / 'ngrid_remote'), "voxelsize": cfg['voxel_um'],
              "cache_size": 6000000000}
    json.dump(params, open(pdir / 'params.json', 'w'))
    segs = sorted(glob.glob(str(pdir / 'auto_grown_*')))
    if not segs:
        rc = sh([str(BIN / 'vc_grow_seg_from_seed'), '-v', cfg['surface_url'], '-t', str(pdir), '-p', str(pdir / 'params.json'),
                 '--seed', *map(str, seed)], pdir / 'grow.log', timeout=1800)
        segs = sorted(glob.glob(str(pdir / 'auto_grown_*')))
        if rc != 0 or not segs:
            raise RuntimeError(f'grow failed rc={rc}')
    meta = json.load(open(Path(segs[0]) / 'meta.json'))
    if meta.get('area_cm2', 0) < 1.0:
        raise RuntimeError(f"patch too small {meta.get('area_cm2')}")
    rc = sh([str(BIN / 'vc_render_tifxyz'), '--volume', str(W / 'fl/vol' / (cfg['name'] + '.zarr')), '--remote-url', cfg['volume_url'],
             '--prefetch-remote', '--group-idx', '0', '--scale', '1', '--segmentation', segs[0], '--num-slices', '31',
             '--slice-step', '1', '--cache-gb', '6', '--zarr-output', str(pdir / 'r31.zarr')], pdir / 'render.log', timeout=1800,
            env=dict(ENV, OMP_NUM_THREADS='12'))
    if rc != 0:
        raise RuntimeError(f'render failed rc={rc}')
    return pdir


def ink(pdir, models, depths):
    out = pdir / 'maps'
    for d in depths:
        tag = f'd{d}'
        if (pdir / f'done_{tag}').exists():
            continue
        rc = sh(['bash', str(W / 'fl/scripts/ink4.sh'), str(pdir / 'r31.zarr'), '31', str(d), str(out), *models], pdir / f'ink_{tag}.log',
                env=dict(ENV, NATIVE_UM=str(json.load(open(pdir / 'params.json'))['voxelsize']), DOWN='3'))
        (pdir / f'done_{tag}').write_text(str(rc))


def score(pdir):
    """crude text-likeness: count letter-sized high-z blobs and how row-like they are, per side"""
    import cv2
    res = {}
    for f in glob.glob(str(pdir / 'maps' / '*_z.npy')):
        z = np.load(f).astype(np.float32)
        b = os.path.basename(f)
        side = 'rev' if '_reverse' in b else 'fwd'
        d = b.split('_d')[-1].split('_')[0]
        m = (z > 1.5).astype(np.uint8)
        n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
        areas = st[1:, cv2.CC_STAT_AREA]
        letter = (areas > 2000) & (areas < 60000)          # ~0.15 - 4.5 mm^2 at 8.64 um
        valid = (z != 0).mean()
        res[f'{side}_d{d}'] = dict(n_letterblobs=int(letter.sum()), blob_frac=float(m.mean() / max(valid, 1e-6)),
                                   top_mean=float(np.sort(z[z != 0])[-max(1, int(0.02 * (z != 0).sum())):].mean()) if (z != 0).any() else 0.0)
    json.dump(res, open(pdir / 'score.json', 'w'), indent=1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('scroll')
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--gens', type=int, default=70)
    ap.add_argument('--models', default='hybrid_3d2d-seed42,hybrid_3d2d-seed43,dnative')
    ap.add_argument('--depths', default='0')
    ap.add_argument('--zs', default='')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    cfg = SCROLLS[a.scroll]
    sdir = W / 'fl' / cfg['name'] / 'screen'
    sdir.mkdir(parents=True, exist_ok=True)
    (W / 'fl' / cfg['name'] / 'ngrid_remote').mkdir(parents=True, exist_ok=True)
    json.dump({"url": cfg['ngrid_url']}, open(W / 'fl' / cfg['name'] / 'ngrid_remote' / 'normal-grids-remote.json', 'w'))
    Z = cfg['shape'][0]
    zs = [int(v) for v in a.zs.split(',')] if a.zs else [int(Z * f) // 8 * 8 for f in np.linspace(0.15, 0.85, 8)]
    seeds_file = sdir / f'seeds_{a.seed}.json'
    if seeds_file.exists():
        seeds = json.load(open(seeds_file))
    else:
        seeds = pick_seeds(cfg, sdir, zs, a.n, seed=a.seed)
        json.dump(seeds, open(seeds_file, 'w'))
    models = a.models.split(',')
    depths = [int(v) for v in a.depths.split(',')]
    q = queue.Queue(maxsize=2)

    def producer():
        for i, s in enumerate(seeds):
            pid = f's{a.seed}_{i:03d}_x{s[0]}_y{s[1]}_z{s[2]}'
            t = time.time()
            try:
                pdir = grow_and_render(cfg, sdir, pid, s, a.gens)
                print(f'[prep] {pid} {time.time()-t:.0f}s', flush=True)
                q.put(pdir)
            except Exception as e:
                print(f'[prep] {pid} FAILED {e}', flush=True)
        q.put(None)

    th = threading.Thread(target=producer, daemon=True)
    th.start()
    while True:
        pdir = q.get()
        if pdir is None:
            break
        t = time.time()
        ink(pdir, models, depths)
        r = score(pdir)
        print(f'[ink] {pdir.name} {time.time()-t:.0f}s {json.dumps(r)}', flush=True)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
