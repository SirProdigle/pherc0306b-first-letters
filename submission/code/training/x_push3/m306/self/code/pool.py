"""pool.py : SELF lane pool of PHerc0306B surfaces outside the lead-A exclusion zone.

    envs/vcpy/bin/python m306/self/code/pool.py dist [REGION,..]     # distance-to-zone maps (step 4) for pool regions
    envs/vcpy/bin/python m306/self/code/pool.py survey               # whole-render survey (step 8) of the 0306B renders on disk

Regions = P306's (aclick/p306/regions.json) plus any added in m306/self/pool/regions_extra.json.  For each region:
  pool/<r>_dist.npz : d (h/4+1, w/4+1) float32 = mm from the zone (check.distance_mm, nan = no mesh), step, roi
Masks are derived with dmask(): a full-res pixel is admitted for pad p iff all 4 surrounding step-4 samples have
d >= 2.0 + 0.2 (slab) + p + step * 0.71 * 8.64 um (the same conservative rule as check.allowed_mask).
"""
import sys, os, json
import numpy as np
P3 = '/home/liam/projects/vesuvius-work/.claude/worktrees/leadA-push3/x_push3'
SELF = f'{P3}/m306/self'; POOL = f'{SELF}/pool'
sys.path.insert(0, f'{P3}/aclick/p306/exclusion'); import check   # noqa: E402
VX = 0.00864


def regions():
    R = json.load(open(f'{P3}/aclick/p306/regions.json')); R.pop('_note', None)
    for r in R.values(): r['maps'] = f'{P3}/aclick/p306/maps'
    ex = f'{POOL}/regions_extra.json'
    if os.path.exists(ex):
        E = json.load(open(ex)); E.pop('_note', None); R.update(E)
    return R


def dist_grid(zarr, roi, step=4, cap=8.0):
    y0, y1, x0, x1 = roi
    gy = np.arange(y0, y1 + step, step); gx = np.arange(x0, x1 + step, step)
    YY, XX = np.meshgrid(gy, gx, indexing='ij')
    return check.distance_mm(zarr, YY, XX, cap_mm=cap).astype(np.float32)


def dmask(d, shape, pad, step=4):
    thr = check.RADIUS_MM + check.SLAB_MM + pad + step * 0.71 * VX
    ok = np.nan_to_num(d, nan=-1.0) >= thr
    q = ok[:-1, :-1] & ok[1:, :-1] & ok[:-1, 1:] & ok[1:, 1:]
    full = np.repeat(np.repeat(q, step, 0), step, 1)
    return full[:shape[0], :shape[1]]


def load_dist(r):
    z = np.load(f'{POOL}/{r}_dist.npz'); return z['d'], int(z['step'])


if __name__ == '__main__':
    os.makedirs(POOL, exist_ok=True)
    R = regions()
    if sys.argv[1] == 'dist':
        names = sys.argv[2].split(',') if len(sys.argv) > 2 else list(R)
        for r in names:
            out = f'{POOL}/{r}_dist.npz'
            if os.path.exists(out): continue
            g = R[r]; d = dist_grid(g['zarr'], g['roi'])
            np.savez_compressed(out, d=d, step=4, roi=np.array(g['roi']), zarr=g['zarr'])
            h, w = g['roi'][1] - g['roi'][0], g['roi'][3] - g['roi'][2]
            fr = {p: round(float(dmask(d, (h, w), p).mean()), 3) for p in (0.0, 0.4, 0.7, 1.6)}
            print(r, g['roi'], 'dist min %.2f med %.2f' % (np.nanmin(d), np.nanmedian(d)), 'allowed frac by pad', fr, flush=True)
    elif sys.argv[1] == 'survey':
        import zarr
        W = '/home/liam/projects/vesuvius-work'
        reg = json.load(open(f'{P3}/aclick/p306/exclusion/renders.json'))
        res = {}
        for z in reg:
            if 'zone test only' in reg[z].get('note', '') or not os.path.exists(z): continue
            a = zarr.open(z, mode='r'); a = a if hasattr(a, 'shape') else a['0']
            H, Wd = a.shape[1:]; st = 8
            d = dist_grid(z, (0, H, 0, Wd), step=st)
            m = {p: dmask(d, (H, Wd), p, st) for p in (0.0, 0.7)}
            mid = a.shape[0] // 2
            ct = np.asarray(a[mid, ::st, ::st]) > 0
            k = lambda mm: float((mm[::st, ::st][:ct.shape[0], :ct.shape[1]] & ct).sum() * (st * VX) ** 2 / 100)
            res[z.replace(W, 'W')] = dict(shape=[H, Wd], layers=a.shape[0], ct_cm2=round(float(ct.sum() * (st * VX) ** 2 / 100), 2),
                                         allowed0_cm2=round(k(m[0.0]), 2), allowed07_cm2=round(k(m[0.7]), 2))
            np.savez_compressed(f'{POOL}/survey_{os.path.basename(z).replace(".zarr", "")}_dist8.npz', d=d, step=st)
            print(z, res[z.replace(W, 'W')], flush=True)
        json.dump(res, open(f'{POOL}/survey.json', 'w'), indent=1)
