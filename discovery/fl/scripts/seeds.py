"""seeds.py SCROLL N [--spacing-mm 18] [--nz 8] : write/extend fl/<SCROLL>/seeds.json with explicit GrowPatch seeds.
Seeds = surface-prediction pixels (L3) whose CT (L3, same slice) is papyrus-bright and not at the rim,
spaced >= spacing from each other and from seeds already in the file."""
import sys, json, os, subprocess, numpy as np, random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
W = Path('/home/liam/projects/vesuvius-work')
S, N = sys.argv[1], int(sys.argv[2])
sp_mm = float(sys.argv[sys.argv.index('--spacing-mm') + 1]) if '--spacing-mm' in sys.argv else 18.0
nz = int(sys.argv[sys.argv.index('--nz') + 1]) if '--nz' in sys.argv else 8
cfg = json.load(open(W / 'fl/scrolls.json'))[S]
d = W / 'fl' / S / 'screen'; d.mkdir(parents=True, exist_ok=True)
Z = cfg['shape'][0]
zs = [int(Z * f) // 8 * 8 for f in np.linspace(0.12, 0.88, nz)]
env = dict(os.environ, VC3D_CONFIG_DIR=str(W / 'fl/vc3dcfg'))
def fetch(args):
    url, pre, z = args
    if not (d / f'{pre}_z{z}_L3.npy').exists():
        subprocess.run([str(W / 'envs/vcpy/bin/python'), str(W / 'fl/scripts/xsec.py'), url, '3', str(d / pre), str(z)],
                       env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
jobs = [(cfg['surface_url'], 'surf', z) for z in zs] + [(cfg['volume_url'], 'ct', z) for z in zs]
with ThreadPoolExecutor(8) as ex:
    list(ex.map(fetch, jobs))
sf = W / 'fl' / S / 'seeds.json'
seeds = json.load(open(sf)) if sf.exists() else []
sp_l0 = sp_mm * 1000 / cfg['voxel_um']
rng = random.Random(len(seeds))
cands = []
for z in zs:
    try:
        s = np.load(d / f'surf_z{z}_L3.npy') > 0; c = np.load(d / f'ct_z{z}_L3.npy').astype(np.float32)
    except FileNotFoundError:
        continue
    inside = c > 0
    if inside.sum() < 1000:
        continue
    thr = np.percentile(c[inside], 40)                      # papyrus-bright
    import cv2
    core = cv2.erode(inside.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0   # >=100 L0 vox from the mask edge
    ys, xs = np.nonzero(s & (c > thr) & core)
    for i in rng.sample(range(len(ys)), min(len(ys), 4000)):
        cands.append((int(xs[i]) * 8 + 4, int(ys[i]) * 8 + 4, z))
rng.shuffle(cands)
added = 0
for x, y, z in cands:
    if added >= N:
        break
    if all((x - a) ** 2 + (y - b) ** 2 + (z - c) ** 2 > sp_l0 ** 2 for a, b, c, *_ in seeds):
        seeds.append([x, y, z, 'new']); added += 1
json.dump(seeds, open(sf, 'w'))
print(S, 'seeds total', len(seeds), 'added', added, 'from', len(cands), 'candidates')
