"""reflatten.py MESH_IN MESH_OUT [--region u0,u1,v0,v1] [--anchor u0,u1,v0,v1] [--iters 60] : ARAP re-flattening of a carrier-grid
mesh over a region (coordinator 15 Oct: correct the flattening where it distorts letters > 5 %; Q5: the right-end group is drawn
~14-16 % squatter across the line).  CT-free and ink-free: pure geometry of the adopted surface.

1. Take the mesh's nodes inside the region (carrier grid, 20 vx cells = 0.1728 mm) and triangulate the grid (2 triangles per cell).
2. As-rigid-as-possible parametrisation (Liu et al. 2008, local/global): each 3-D triangle is laid flat isometrically; the global
   step solves a cotangent-weighted Laplacian (factorised once) for the 2-D positions; the local step fits each triangle's best
   rotation by SVD (reflections excluded).  Initialised with the current (u, v) and iterated --iters times.
3. The result is aligned (rotation + translation, no scale) to the old (u, v) on the --anchor nodes (default: the line band
   u -10..+8, |v| <= 2.5, where the old grid is metric within ~1 %), so the claim frame is kept where it was already right.
4. Resampled on the SAME global carrier grid (origin u -36, v -21, 20 vx) inside the region by barycentric interpolation of the
   3-D positions over the parameter triangulation; nodes outside the region are invalid (-1).  So every P1 tool (render, ink,
   evalP1, views) works unchanged, and render px = claim-frame mm as before.
Writes MESH_OUT/{x,y,z}.tif + meta.json (+ reflatten.json with the distortion before / after)."""
import os, sys, json, argparse
import numpy as np, tifffile
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.interpolate import LinearNDInterpolator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p1geom as g                                                                   # noqa: E402
from metric import metric                                                           # noqa: E402
CELL = g.G / g.K; U0, V0 = -36.0, -21.0


def tris_of(valid):
    H, W = valid.shape; idx = -np.ones((H, W), int); ii, jj = np.nonzero(valid); idx[ii, jj] = np.arange(len(ii))
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    q = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
    T = np.concatenate([np.stack([a[q], b[q], d[q]], 1), np.stack([a[q], d[q], c[q]], 1)])
    return idx, T


def arap(X, T, uv0, iters=60, fix=0, tw=None):
    """tw: optional per-triangle weight (folds down-weighted so they absorb the distortion instead of spreading it)"""
    n = len(X); p0, p1, p2 = X[T[:, 0]], X[T[:, 1]], X[T[:, 2]]
    # isometric local 2-D frames per triangle
    e1 = p1 - p0; e2 = p2 - p0; l1 = np.linalg.norm(e1, axis=1); ex = e1 / l1[:, None]
    nrm = np.cross(e1, e2); ey = np.cross(nrm, e1); ey /= np.linalg.norm(ey, axis=1)[:, None]
    loc = np.zeros((len(T), 3, 2)); loc[:, 1, 0] = l1; loc[:, 2, 0] = (e2 * ex).sum(1); loc[:, 2, 1] = (e2 * ey).sum(1)
    # cotangent weights per (triangle, edge opposite vertex k)
    cot = np.zeros((len(T), 3))
    for k in range(3):
        a_, b_ = loc[:, (k + 1) % 3] - loc[:, k], loc[:, (k + 2) % 3] - loc[:, k]
        cot[:, k] = (a_ * b_).sum(1) / np.maximum(np.abs(a_[:, 0] * b_[:, 1] - a_[:, 1] * b_[:, 0]), 1e-9)
    cot = np.clip(cot, 1e-3, 1e3) * 0.5
    if tw is not None: cot = cot * tw[:, None]
    rows, cols, vals = [], [], []
    for k in range(3):
        i, j = T[:, (k + 1) % 3], T[:, (k + 2) % 3]; w = cot[:, k]
        rows += [i, j, i, j]; cols += [i, j, j, i]; vals += [w, w, -w, -w]
    L = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tolil()
    L[fix, :] = 0; L[fix, fix] = 1; L = L.tocsc(); lu = splu(L)
    uv = uv0.copy()
    for it in range(iters):
        # local: best rotation per triangle
        S = np.zeros((len(T), 2, 2))
        for k in range(3):
            i, j = T[:, (k + 1) % 3], T[:, (k + 2) % 3]
            du = uv[i] - uv[j]; dx = loc[:, (k + 1) % 3] - loc[:, (k + 2) % 3]
            S += cot[:, k, None, None] * du[:, :, None] * dx[:, None, :]
        Uq, _, Vt = np.linalg.svd(S); R = Uq @ Vt
        neg = np.linalg.det(R) < 0
        if neg.any():
            Uq[neg, :, -1] *= -1; R[neg] = Uq[neg] @ Vt[neg]
        # global
        b = np.zeros((n, 2))
        for k in range(3):
            i, j = T[:, (k + 1) % 3], T[:, (k + 2) % 3]
            dx = loc[:, (k + 1) % 3] - loc[:, (k + 2) % 3]; r = np.einsum('tab,tb->ta', R, dx) * cot[:, k, None]
            np.add.at(b, i, r); np.add.at(b, j, -r)
        b[fix] = uv[fix]
        uv = np.stack([lu.solve(b[:, 0]), lu.solve(b[:, 1])], 1)
    return uv


def procrustes(src, dst):
    ms, md = src.mean(0), dst.mean(0); A = (src - ms).T @ (dst - md); Uq, _, Vt = np.linalg.svd(A); R = Uq @ Vt
    if np.linalg.det(R) < 0: Uq[:, -1] *= -1; R = Uq @ Vt
    return lambda p: (p - ms) @ R + md


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('mesh'); ap.add_argument('out')
    ap.add_argument('--region', default='-16,27,-10,7'); ap.add_argument('--anchor', default='-10,8,-2.5,2.5'); ap.add_argument('--iters', type=int, default=60)
    ap.add_argument('--fold-weight', type=float, default=1.0, help='ARAP weight of fold triangles (3-D/param area ratio outside '
                    '[1/--fold-thr, --fold-thr] on the input grid); 1 = plain ARAP')
    ap.add_argument('--fold-thr', type=float, default=1.25)
    a = ap.parse_args()
    P = g.load(a.mesh); NR, NC = P.shape[:2]
    jj, ii = np.meshgrid(np.arange(NC), np.arange(NR)); U, V = U0 + jj * CELL, V0 + ii * CELL
    u0, u1, v0, v1 = (float(x) for x in a.region.split(','))
    reg = np.isfinite(P).all(-1) & (U >= u0) & (U <= u1) & (V >= v0) & (V <= v1)
    idx, T = tris_of(reg); X = P[reg] / 20.0                      # 3-D in grid units (20 vx = 1 cell)
    uv0 = np.stack([jj[reg], ii[reg]], 1).astype(float)
    tw = None
    if a.fold_weight < 1:
        q0, q1, q2 = X[T[:, 0]], X[T[:, 1]], X[T[:, 2]]; A3 = 0.5 * np.linalg.norm(np.cross(q1 - q0, q2 - q0), axis=1)
        r0, r1, r2 = uv0[T[:, 0]], uv0[T[:, 1]], uv0[T[:, 2]]; A2 = 0.5 * np.abs((r1 - r0)[:, 0] * (r2 - r0)[:, 1] - (r1 - r0)[:, 1] * (r2 - r0)[:, 0])
        ratio = A3 / np.maximum(A2, 1e-9); fold = (ratio > a.fold_thr) | (ratio < 1 / a.fold_thr)
        tw = np.where(fold, a.fold_weight, 1.0); print(f'fold triangles {fold.mean():.3f} (weight {a.fold_weight})')
    uv = arap(X, T, uv0, a.iters, fix=int(np.argmin(np.abs(U[reg]) + np.abs(V[reg]))), tw=tw)
    A0, A1, B0, B1 = (float(x) for x in a.anchor.split(','))
    anc = (U[reg] >= A0) & (U[reg] <= A1) & (V[reg] >= B0) & (V[reg] <= B1)
    f = procrustes(uv[anc], uv0[anc]); uvA = f(uv)
    # resample on the global grid inside the region
    interp = LinearNDInterpolator(uvA, P[reg])
    Q = interp(np.stack([jj[reg], ii[reg]], 1).astype(float))
    out = np.full(P.shape, -1.0); out[reg] = np.where(np.isfinite(Q), Q, -1.0)
    os.makedirs(a.out, exist_ok=True)
    for k, c in enumerate('xyz'): tifffile.imwrite(f'{a.out}/{c}.tif', out[..., k].astype(np.float32))
    meta = json.load(open(f'{a.mesh}/meta.json')); meta['uuid'] = meta['name'] = os.path.basename(os.path.normpath(a.out))
    ok = (out[..., 0] > 0); meta['bbox'] = [out[ok].min(0).tolist(), out[ok].max(0).tolist()]
    meta['p1_reflatten'] = dict(source=os.path.abspath(a.mesh), region_uv_mm=[u0, u1, v0, v1], anchor_uv_mm=[A0, A1, B0, B1], iters=a.iters, fold_weight=a.fold_weight, fold_thr=a.fold_thr,
                                method='ARAP (local/global, cotangent weights) on the grid triangulation, Procrustes-aligned to the old (u,v) on the anchor, '
                                       'barycentric resampling on the same carrier grid')
    json.dump(meta, open(f'{a.out}/meta.json', 'w'), indent=1)
    # distortion before / after on the region and the right group
    rep = {}
    for nm, PP in (('before', P), ('after', np.where(out[..., :1] > 0, out, np.nan))):
        su, sv, sh, ar = metric(PP); Uc, Vc = U[:-1, :-1] + CELL / 2, V[:-1, :-1] + CELL / 2
        for k, (x0, x1, y0, y1) in dict(region=(u0 + .5, u1 - .5, v0 + .5, v1 - .5), right_group=(15.5, 23.5, -2.7, 1.6),
                                         line_band=(-15, 12, -2.5, 2.5), rect=(-12.2, 23.54, -6.47, 2.53)).items():
            m = np.isfinite(su) & np.isfinite(sv) & (Uc >= x0) & (Uc < x1) & (Vc >= y0) & (Vc < y1)
            pct = lambda x: [round(float(np.percentile(x[m], q)), 3) for q in (5, 50, 95)]
            rep[f'{nm}:{k}'] = dict(stretch_u=pct(su), stretch_v=pct(sv), shear=pct(sh), area=pct(ar),
                                    out5=round(float(((np.abs(su[m] - 1) > .05) | (np.abs(sv[m] - 1) > .05)).mean()), 3))
    disp = np.linalg.norm(uvA - uv0, axis=1) * CELL
    rep['param_shift_mm'] = dict(anchor_p95=round(float(np.percentile(disp[anc], 95)), 3), right_group_median=round(float(np.median(
        disp[(U[reg] >= 15.5) & (U[reg] <= 23.5) & (V[reg] >= -2.7) & (V[reg] <= 1.6)])), 3), max=round(float(disp.max()), 3))
    json.dump(rep, open(f'{a.out}/reflatten.json', 'w'), indent=1)
    for k, x in rep.items(): print(k, x)


if __name__ == '__main__':
    main()
