"""sqm_grow.py SCROLL X Y Z TAG [--gens 100] : grow a large GrowPatch patch at a Scan-Quality-Map starting spot.
Snaps (X,Y,Z) (level-0 voxels) to the nearest m7 surface-prediction voxel on the level-3 slice at Z, then runs
vc_grow_seg_from_seed with the scroll's normal grids. Output: fl/sqm/<SCROLL>_<TAG>/auto_grown_*/ (prints its path)."""
import sys, json, glob, subprocess, argparse
from pathlib import Path
import numpy as np
W = Path('/home/liam/projects/vesuvius-work')
ap = argparse.ArgumentParser(); ap.add_argument('scroll'); ap.add_argument('x', type=int); ap.add_argument('y', type=int); ap.add_argument('z', type=int)
ap.add_argument('tag'); ap.add_argument('--gens', type=int, default=100); ap.add_argument('--nosnap', action='store_true')
a = ap.parse_args()
cfg = json.load(open(W / 'fl/scrolls.json'))[a.scroll]
od = W / 'fl/sqm' / f'{a.scroll}_{a.tag}'; od.mkdir(parents=True, exist_ok=True)
segs = sorted(glob.glob(str(od / 'auto_grown_*')))
if not segs:
    sp = od / f'surf_z{a.z}_L3.npy'
    if not sp.exists() and not a.nosnap:
        subprocess.run([str(W / 'envs/vcpy/bin/python'), str(W / 'fl/scripts/xsec.py'), cfg['surface_url'], '3', str(od / 'surf'), str(a.z)], check=True,
                       stdout=open(od / 'xsec.log', 'w'), stderr=subprocess.STDOUT)
    if a.nosnap:
        seed = [a.x, a.y, a.z]
    else:
        s = np.load(sp) > 0
        ys, xs = np.nonzero(s)
        i = int(np.argmin((xs * 8 + 4 - a.x) ** 2 + (ys * 8 + 4 - a.y) ** 2))
        seed = [int(xs[i] * 8 + 4), int(ys[i] * 8 + 4), a.z]
    dist = float(np.hypot(seed[0] - a.x, seed[1] - a.y))
    json.dump({'spot': [a.x, a.y, a.z], 'seed': seed, 'snap_dist_vox': dist}, open(od / 'seed.json', 'w'))
    json.dump({'mode': 'seed', 'step_size': 20, 'min_area_cm': 0.5, 'generations': a.gens, 'thread_limit': 4,
               'normal_grid_path': str(W / 'fl' / a.scroll / 'ngrid_remote'), 'voxelsize': cfg['voxel_um'], 'cache_size': 4000000000},
              open(od / 'params.json', 'w'))
    subprocess.run([str(W / 'vc_build_main/bin/vc_grow_seg_from_seed'), '-v', cfg['surface_url'], '-t', str(od), '-p', str(od / 'params.json'),
                    '--seed', *map(str, seed)], stdout=open(od / 'grow.log', 'w'), stderr=subprocess.STDOUT, timeout=5400)
    segs = sorted(glob.glob(str(od / 'auto_grown_*')))
print(segs[0] if segs else 'NONE', flush=True)
