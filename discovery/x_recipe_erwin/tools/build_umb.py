"""build_umb.py SCROLL Z0 Z1 STEP OUT.json : xy normal grids every STEP slices in [Z0, Z1] -> vc_gen_umbilicus (5 repeats,
median) -> drop points > 400 vox from a running median -> savgol smooth -> VC3D control_points json."""
import sys, os, json, subprocess, numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.signal import savgol_filter
W = '/home/liam/projects/vesuvius-work'
S, Z0, Z1, STEP, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
cfg = json.load(open(f'{W}/fl/scrolls.json'))[S]
url = cfg['ngrid_url']; s3 = 's3://vesuvius-challenge-open-data/' + url.split('amazonaws.com/')[1]
D = f'{W}/x_recipe_erwin/ngrid_{S}'; os.makedirs(f'{D}/xy', exist_ok=True)
subprocess.run(['aws', 's3', 'cp', '--no-sign-request', '--only-show-errors', f'{s3}/metadata.json', f'{D}/metadata.json'], check=True)
zs = list(range(Z0, Z1 + 1, STEP))
def get(z):
    f = f'{D}/xy/{z:06d}.grid'
    if not os.path.exists(f):
        subprocess.run(['aws', 's3', 'cp', '--no-sign-request', '--only-show-errors', f'{s3}/xy/{z:06d}.grid', f], check=False)
with ThreadPoolExecutor(8) as ex: list(ex.map(get, zs))
csv = OUT.replace('.json', '_all.csv'); raw = OUT.replace('.json', '_raw.json')
subprocess.run(['nice', '-n', '15', f'{W}/vc_build_main/bin/vc_gen_umbilicus', '-i', D, '-o', raw, '--csv', csv, '--seed', '7',
                '--repeats', '5', '--threads', '4', '--voxelsize-um', str(cfg['voxel_um']),
                '--volume-width', str(cfg['shape'][2]), '--volume-height', str(cfg['shape'][1]), '--volume-slices', str(cfg['shape'][0])],
               check=True, stdout=subprocess.DEVNULL)
cp = json.load(open(raw))['control_points']
P = np.array([[c['z'], c['x'], c['y']] for c in cp if c.get('inside', True)], float)
P = P[np.argsort(P[:, 0])]
keep = np.ones(len(P), bool)
for it in range(3):
    med = np.array([np.median(P[max(0, i - 5):i + 6][keep[max(0, i - 5):i + 6]], axis=0) if keep[max(0, i - 5):i + 6].any() else P[i]
                    for i in range(len(P))])
    d = np.hypot(P[:, 1] - med[:, 1], P[:, 2] - med[:, 2]); keep = d < 400
Q = P[keep]
wl = min(15, len(Q) - (1 - len(Q) % 2)); wl = wl if wl % 2 else wl - 1
xs = savgol_filter(Q[:, 1], wl, 2) if wl >= 5 else Q[:, 1]; ys = savgol_filter(Q[:, 2], wl, 2) if wl >= 5 else Q[:, 2]
out = {'control_points': [{'x': float(x), 'y': float(y), 'z': float(z)} for z, x, y in zip(Q[:, 0], xs, ys)],
       'metadata': {'tool': 'build_umb.py', 'scroll': S, 'kept': int(keep.sum()), 'total': int(len(P))}}
json.dump(out, open(OUT, 'w'), indent=1)
print(S, 'kept', int(keep.sum()), 'of', len(P), 'x range', round(xs.min()), round(xs.max()), 'y range', round(ys.min()), round(ys.max()))
