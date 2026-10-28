"""carrier.py : the claim-frame grid and the smooth carrier surface over lead A's column, plus its CT depth profiles.

Grid: node (i, j) <-> claim frame (u, v) = (U0 + j * CELL, V0 + i * CELL), CELL = 20 vx = 0.1728 mm, so a render of a
surface on this grid is already levelled: render px (x, y) = (20 j, 20 i) <-> (u, v) = (U0 + x / K, V0 + y / K).

Carrier (the parametrisation the refined surfaces are built on, NOT a sheet estimate):
  1. P_big(u, v) = w074big at its N7 affine (u, v) -> grid position (bilinear);
  2. inside A's patch the in-plane part of (P_A - P_big) is added (smoothed 1 mm, feathered 1.5 mm at A's border), so the carrier's
     (u, v) labels equal A's exact claim frame where A exists (the normal part is dropped: the tracker decides the sheet);
  3. Gaussian smoothing of the positions (sigma --smooth-mm, normalised convolution), normals from the smoothed grid
     (renderer convention n = normalize(dP/dcol x dP/drow); +n = A's +n = towards A's reading face).
Profiles: CT along the carrier normal, t = -T..+T vx (1 vx steps), each the mean of a 3 x 3 in-plane footprint (+-7 vx along
the carrier's tangents), trilinear.  Also: where A, big, R1, L1 and the lane windings cross each node's normal.

  carrier.py --out trials/P1_surface_refine/carrier   [--smooth-mm 1.5] [--T 60]
writes carrier.npz (P, n, uv grid, crossings) and prof.npy (NR, NC, 2T+1) float16.
"""
import os, sys, argparse, time
import numpy as np
from scipy import ndimage as ndi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p1geom as g                                                                   # noqa: E402

U0, U1, V0, V1 = -36.0, 31.0, -21.0, 21.0
CELL = g.G / g.K
NC = int(np.ceil((U1 - U0) / CELL)) + 1
NR = int(np.ceil((V1 - V0) / CELL)) + 1


def grid_uv():
    j = np.arange(NC); i = np.arange(NR)
    return np.meshgrid(U0 + j * CELL, V0 + i * CELL)          # u (NR, NC), v (NR, NC)


def nconv(X, w, sig):
    num = ndi.gaussian_filter(np.where(w[..., None] > 0, X, 0) * w[..., None], (sig, sig, 0), mode='nearest')
    den = ndi.gaussian_filter(w.astype(float), sig, mode='nearest')
    return num / np.maximum(den, 1e-9)[..., None], den


def build_carrier(smooth_mm=1.5):
    uu, vv = grid_uv()
    big = g.load(g.SURF['big']); A = g.load(g.SURF['A'])
    rb, cb = g.uv_to_grid('big', uu, vv); Pb = g.bilinear(big, rb, cb)
    ra, ca = g.a_uv_to_grid(uu, vv); Pa = g.bilinear(A, ra, ca)
    okb = np.isfinite(Pb).all(-1); oka = np.isfinite(Pa).all(-1)
    # in-plane part of A - big (normal from big's smoothed grid)
    Pbs, _ = nconv(np.nan_to_num(Pb), okb.astype(float), 1.0 / CELL)
    nb = g.normals(np.where(okb[..., None], Pbs, np.nan))
    D = np.where((oka & okb)[..., None], Pa - Pb, 0.0)
    D_in = D - (D * np.nan_to_num(nb)).sum(-1, keepdims=True) * np.nan_to_num(nb)
    both = (oka & okb).astype(float)
    Ds, den = nconv(D_in, both, 1.0 / CELL)
    feather = ndi.gaussian_filter(ndi.binary_erosion(oka, iterations=int(1.0 / CELL)).astype(float), 1.5 / CELL / 2)
    wA = np.clip(feather, 0, 1) * (den > 0.2)
    P0 = np.where(okb[..., None], Pb + wA[..., None] * Ds, np.nan)
    ok = np.isfinite(P0).all(-1)
    Ps, den = nconv(np.nan_to_num(P0), ok.astype(float), smooth_mm / CELL)
    Ps[~ok] = np.nan
    n = g.normals(Ps)
    return dict(u=uu, v=vv, P=Ps, n=n, ok=ok & np.isfinite(n).all(-1), wA=wA.astype(np.float32),
                Pb=Pb, Pa=Pa, D_in=Ds.astype(np.float32))


def tangents(P):
    tu = np.gradient(P, axis=1); tv = np.gradient(P, axis=0)
    tu /= np.linalg.norm(tu, axis=-1, keepdims=True) + 1e-9; tv /= np.linalg.norm(tv, axis=-1, keepdims=True) + 1e-9
    return tu, tv


def profiles(C, T=60, foot=7, rows_per=12, log=print):
    V = g.Vol(threads=24)
    P, n, ok = C['P'], C['n'], C['ok']
    tu, tv = tangents(np.nan_to_num(P))
    ts = np.arange(-T, T + 1, dtype=float)
    out = np.zeros(P.shape[:2] + (len(ts),), np.float16)
    offs = [(a, b) for a in (-foot, 0, foot) for b in (-foot, 0, foot)]
    t0 = time.time()
    for r0 in range(0, P.shape[0], rows_per):
        sl = slice(r0, min(P.shape[0], r0 + rows_per))
        p = P[sl]; nn = n[sl]; m = ok[sl]
        acc = np.zeros(p.shape[:2] + (len(ts),), np.float32)
        for a, b in offs:
            q = p + a * tu[sl] + b * tv[sl]
            L = q[:, :, None, :] + ts[None, None, :, None] * nn[:, :, None, :]
            L[~m] = np.nan
            acc += V.sample(L)
        out[sl] = (acc / len(offs)).astype(np.float16)
        log(f'  rows {r0}-{sl.stop} / {P.shape[0]}  {time.time() - t0:.0f}s', flush=True)
    return ts, out


def crossings(C, T=60, names=None):
    names = names or ['A', 'big', 'R1', 'L1', 'z84_w072', 'z84_w073', 'z84_w074', 'z84o_w073', 'z84o_w074', 'z84o_w075', 'z84o_w076']
    P, n, ok = C['P'], C['n'], C['ok']
    flat = P.reshape(-1, 3); fn = n.reshape(-1, 3)
    lo = np.nanmin(flat, 0) - 100; hi = np.nanmax(flat, 0) + 100
    out = {}
    for s in names:
        S = g.Surf(g.SURF[s], s, up=4, box=(lo, hi))
        t, miss = S.cross(np.where(ok.reshape(-1, 1), flat, np.nan), fn, span=T)
        out[s] = t.reshape(P.shape[:2]).astype(np.float32); print(s, 'crossing share', np.isfinite(out[s]).mean().round(3), flush=True)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True)
    ap.add_argument('--smooth-mm', type=float, default=0.5); ap.add_argument('--T', type=int, default=60)
    ap.add_argument('--stage', default='all', help='carrier | prof (with --rows r0,r1) | merge | all')
    ap.add_argument('--rows', default='')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    if a.stage in ('carrier', 'all'):
        C = build_carrier(a.smooth_mm)
        print('grid', C['P'].shape, 'valid', C['ok'].mean().round(3), 'A-weight>0.5', (C['wA'] > 0.5).mean().round(3), flush=True)
        X = crossings(C, a.T)
        np.savez_compressed(f'{a.out}/carrier.npz', u=C['u'], v=C['v'], P=C['P'], n=C['n'], ok=C['ok'], wA=C['wA'], D_in=C['D_in'],
                            U0=U0, V0=V0, CELL=CELL, smooth_mm=a.smooth_mm, **{f'x_{k}': v for k, v in X.items()})
        np.save(f'{a.out}/prof_ts.npy', np.arange(-a.T, a.T + 1, dtype=float))
    if a.stage in ('prof', 'all'):
        Cz = np.load(f'{a.out}/carrier.npz'); C = {k: Cz[k] for k in ('P', 'n', 'ok')}
        r0, r1 = (int(x) for x in a.rows.split(',')) if a.rows else (0, C['P'].shape[0])
        fn = f'{a.out}/prof_part_{r0:04d}_{r1:04d}.npy'
        if not os.path.exists(fn):
            Cs = {k: v[r0:r1] for k, v in C.items()}
            ts, prof = profiles(Cs, a.T)
            np.save(fn + '.tmp.npy', prof); os.replace(fn + '.tmp.npy', fn)
    if a.stage in ('merge', 'all'):
        parts = sorted(f for f in os.listdir(a.out) if f.startswith('prof_part_') and f.endswith('.npy') and '.tmp' not in f)
        NR = np.load(f'{a.out}/carrier.npz')['P'].shape[0]
        rng = [tuple(int(x) for x in f[10:-4].split('_')) for f in parts]
        cover = np.zeros(NR, bool)
        for r0, r1 in rng: cover[r0:r1] = True
        if cover.all():
            prof = np.concatenate([np.load(f'{a.out}/{f}') for f, _ in sorted(zip(parts, rng), key=lambda x: x[1][0])], 0)
            np.save(f'{a.out}/prof.npy', prof); print('merged', prof.shape)
        else:
            print('missing rows', np.nonzero(~cover)[0][[0, -1]])
    print('done')
