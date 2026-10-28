"""neighbour_seeds.py SCROLL X Y Z NX NY NZ [--reach 120] : sheets next to a spot, along its surface normal.
Samples the m7 surface prediction (level 0) on the line spot + t*normal, t in [-reach, reach]; runs of surface voxels = sheets.
Prints JSON: {"own": t0, "sheets": [[t, x, y, z], ...]} with t = signed offset in voxels (own sheet ~ 0)."""
import sys, json, argparse, numpy as np, vc
ap = argparse.ArgumentParser(); [ap.add_argument(k) for k in ('scroll',)]; [ap.add_argument(k, type=float) for k in ('x', 'y', 'z', 'nx', 'ny', 'nz')]
ap.add_argument('--reach', type=int, default=120); a = ap.parse_args()
cfg = json.load(open('/home/liam/projects/vesuvius-work/fl/scrolls.json'))[a.scroll]
v = vc.Volume.open_url(cfg['surface_url'])
p = np.array([a.x, a.y, a.z]); n = np.array([a.nx, a.ny, a.nz]); n /= np.linalg.norm(n)
ts = np.arange(-a.reach, a.reach + 1)
pts = p[None] + ts[:, None] * n[None]
lo = np.floor(pts.min(0)).astype(int) - 2; hi = np.ceil(pts.max(0)).astype(int) + 3
box = np.asarray(v.read_zyx([int(lo[2]), int(lo[1]), int(lo[0])], [int(hi[2] - lo[2]), int(hi[1] - lo[1]), int(hi[0] - lo[0])], level=0, missing_policy='all_fill'))
idx = np.round(pts - lo).astype(int)
prof = np.array([box[k[2], k[1], k[0]] > 0 for k in idx])
runs = []; i = 0
while i < len(ts):
    if prof[i]:
        j = i
        while j + 1 < len(ts) and prof[j + 1]: j += 1
        tc = int(round((ts[i] + ts[j]) / 2)); q = p + tc * n
        runs.append([tc, int(round(q[0])), int(round(q[1])), int(round(q[2]))]); i = j + 1
    else:
        i += 1
own = min(runs, key=lambda r: abs(r[0]))[0] if runs else None
print(json.dumps({'own': own, 'sheets': runs, 'profile_on_frac': float(prof.mean())}))
