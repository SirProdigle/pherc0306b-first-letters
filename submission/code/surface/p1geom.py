"""p1geom.py : shared geometry for lane P1 (refine the segmentation over lead A's claim box and column).

Frames
- claim frame (u, v) mm: the levelling of CANDIDATES/SUBMISSION_DRAFT_A_owner_only/region.json (= the submission's
  make_final_image levelling): u along A's line, v DOWN AS RENDERED (the submission image is flipped: image y = -v).
  Origin = the levelling centre (A render px 2727, 2825).  A's render px = A grid (col, row) * 20.
  v increases towards LOWER z (dz/dv ~ -113 vx/mm): rendered 'up' (v < 0) = higher z.
- other own-sheet surfaces: N7's affine grid -> (s, t) = (u, v) (x_refine/trials/N7_sheet_extension/frames/NAME.json),
  fitted on 3-D projections onto A (big rmse 0.16 mm, R1 0.22 mm; L1 rigid fallback, approximate).

Surfaces (Surf): P (H, W, 3) xyz with NaN invalid, n = normalize(dP/dcol x dP/drow) (the renderer's layer direction; layer L of a
41-layer render samples P + (L - 20) n), KD-tree, sub-cell projection.

Vol: cached reader of the 8.64 um masked CT (and the m7 surface prediction) streamed from the open-data bucket; reads S4's and
N7's chunk caches first (read-only), new chunks go to trials/P1_surface_refine/cache/.
"""
import os, sys, json, glob
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_k, '4')
import numpy as np, tifffile
from scipy.spatial import cKDTree

W = os.environ.get('FL_ROOT', '/home/liam/projects/vesuvius-work')                  # repo root (I2: portable copy of lane P1's file)
T = os.environ.get('SURF_WORK', f'{W}/x_refine/trials/P1_surface_refine')         # work dir (chunk caches, eval/ outputs)
N7T = f'{W}/x_refine/trials/N7_sheet_extension'
REGION = f'{W}/CANDIDATES/SUBMISSION_DRAFT_A_owner_only/region.json'
A_RENDER = f'{W}/x_focus306/renders/w074own.zarr'
UM = 8.64
G = 20
K = 1000.0 / UM                     # px (vx) per mm
LANE = f'{W}/x_vast_plan/results/PHerc0306B_z8400/results/flat'
LANEO = f'{W}/x_vast_plan/results/PHerc0306B_z8400o/results/flat'

SURF = {
    'A': f'{W}/fl/sqm/PHerc0306B_f306_w074own/auto_grown_20261012144243662',
    'big': f'{W}/fl/sqm/PHerc0306B_f306_w074big/auto_grown_20261014014410250',
    'R1': f'{N7T}/grow/R1/auto_grown_20261014015614971',
    'L1': f'{N7T}/grow/L1/auto_grown_20261014025938089',
}
for _w in ('w072', 'w073', 'w074'):
    SURF[f'z84_{_w}'] = f'{LANE}/{_w}/tifxyz/flatten.tifxyz'
for _w in ('w073', 'w074', 'w075', 'w076'):
    SURF[f'z84o_{_w}'] = f'{LANEO}/{_w}/tifxyz/flatten.tifxyz'


def load(d):
    P = np.stack([tifffile.imread(f'{d}/{c}.tif') for c in 'xyz'], -1).astype(np.float64)
    bad = (P[..., 0] == -1) | (P[..., 2] <= 0) | ~np.isfinite(P).all(-1)
    P[bad] = np.nan
    return P


def normals(P):
    """renderer convention (central differences, one-sided at the border/invalid neighbours handled as NaN)"""
    tu = np.full_like(P, np.nan); tv = np.full_like(P, np.nan)
    tu[:, 1:-1] = (P[:, 2:] - P[:, :-2]) / 2; tv[1:-1] = (P[2:] - P[:-2]) / 2
    # one-sided where a central difference is NaN
    f = np.full_like(P, np.nan); f[:, :-1] = P[:, 1:] - P[:, :-1]
    b = np.full_like(P, np.nan); b[:, 1:] = P[:, 1:] - P[:, :-1]
    tu = np.where(np.isfinite(tu), tu, np.where(np.isfinite(f), f, b))
    f = np.full_like(P, np.nan); f[:-1] = P[1:] - P[:-1]
    b = np.full_like(P, np.nan); b[1:] = P[1:] - P[:-1]
    tv = np.where(np.isfinite(tv), tv, np.where(np.isfinite(f), f, b))
    n = np.cross(tu, tv); n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    return n


def bilinear(A, rows, cols):
    """A (H, W, ...) sampled at float grid coords; NaN-propagating"""
    rows = np.asarray(rows, float); cols = np.asarray(cols, float)
    H, Wd = A.shape[:2]
    r0 = np.clip(np.floor(rows).astype(int), 0, H - 2); c0 = np.clip(np.floor(cols).astype(int), 0, Wd - 2)
    fr = rows - r0; fc = cols - c0
    out_of = (rows < 0) | (rows > H - 1) | (cols < 0) | (cols > Wd - 1)
    ex = (Ellipsis,) + (None,) * (A.ndim - 2)
    fr = fr[ex]; fc = fc[ex]
    o = A[r0, c0] * (1 - fr) * (1 - fc) + A[r0, c0 + 1] * (1 - fr) * fc + A[r0 + 1, c0] * fr * (1 - fc) + A[r0 + 1, c0 + 1] * fr * fc
    o = np.asarray(o, float); o[out_of] = np.nan
    return o


class Surf:
    def __init__(self, d_or_P, name='', up=1, box=None):
        """up: upsample factor for the KD-tree point cloud (crossings); box: (xyz_min, xyz_max) to restrict the tree"""
        self.name = name
        self.P = load(d_or_P) if isinstance(d_or_P, str) else d_or_P
        self.ok = np.isfinite(self.P).all(-1)
        self.n = normals(self.P)
        self.up = up
        if up > 1:
            H, Wd = self.P.shape[:2]
            rr = np.arange(0, H - 1 + 1e-9, 1.0 / up); cc = np.arange(0, Wd - 1 + 1e-9, 1.0 / up)
            R, C = np.meshgrid(rr, cc, indexing='ij')
            Q = bilinear(self.P, R, C)
            self.cloud_rc = np.stack([R, C], -1)
        else:
            Q = self.P; H, Wd = Q.shape[:2]
            R, C = np.mgrid[0:H, 0:Wd]; self.cloud_rc = np.stack([R, C], -1).astype(float)
        m = np.isfinite(Q).all(-1)
        if box is not None:
            lo, hi = np.asarray(box[0]), np.asarray(box[1])
            m &= np.all((Q >= lo) & (Q <= hi), -1) if True else m
        self.cloud = Q[m]; self.cloud_rc = self.cloud_rc[m]
        self.tree = cKDTree(self.cloud) if len(self.cloud) else None

    def cross(self, P0, n0, span=60, iters=5, miss_max=3.0):
        """where does this surface cross the lines P0 + t n0 (|t| <= span)?  -> t (NaN if no crossing), miss (vx)"""
        P0 = np.asarray(P0, float); n0 = np.asarray(n0, float)
        t = np.zeros(len(P0)); ok0 = np.isfinite(P0).all(-1) & np.isfinite(n0).all(-1)
        out_t = np.full(len(P0), np.nan); out_m = np.full(len(P0), np.nan)
        if self.tree is None or not ok0.any(): return out_t, out_m
        p = P0[ok0]; n = n0[ok0]; tt = np.zeros(len(p))
        d, j = self.tree.query(p, distance_upper_bound=span * 1.5)
        good = np.isfinite(d); jj = np.where(good, j, 0)
        tt = ((self.cloud[jj] - p) * n).sum(-1)
        for _ in range(iters):
            X = p + tt[:, None] * n
            d, j = self.tree.query(X); jj = j
            tt = tt + ((self.cloud[jj] - X) * n).sum(-1)
        X = p + tt[:, None] * n; d, j = self.tree.query(X)
        miss = d
        keep = good & (np.abs(tt) <= span) & (miss < miss_max)
        o_t = np.where(keep, tt, np.nan); o_m = np.where(keep, miss, np.nan)
        out_t[ok0] = o_t; out_m[ok0] = o_m
        return out_t, out_m

    def project(self, Q, max_d=60.0):
        """Q (N,3) -> (N,4): row, col (float grid), normal offset (vx, along this surface's +n), in-plane residual (vx)
        (KD-tree nearest vertex of the point cloud, then sub-cell least squares on the grid; N7 geom.Surf.project)"""
        Q = np.asarray(Q, np.float64).reshape(-1, 3)
        N = len(Q); out = np.full((N, 4), np.nan)
        if self.tree is None: return out
        d, i = self.tree.query(Q, distance_upper_bound=max_d)
        good = np.isfinite(d)
        P = self.P; H, Wd = P.shape[:2]
        for k in np.nonzero(good)[0]:
            r, c = np.floor(self.cloud_rc[i[k]]).astype(int); p = Q[k]
            for _ in range(5):
                r = int(np.clip(r, 0, H - 2)); c = int(np.clip(c, 0, Wd - 2))
                q00, q01, q10 = P[r, c], P[r, c + 1], P[r + 1, c]
                if not (np.isfinite(q00).all() and np.isfinite(q01).all() and np.isfinite(q10).all()): break
                tu = q01 - q00; tv = q10 - q00
                (du, dv), *_ = np.linalg.lstsq(np.stack([tu, tv], 1), p - q00, rcond=None)
                if -0.05 <= du <= 1.05 and -0.05 <= dv <= 1.05:
                    nn = np.cross(tu, tv); nn /= np.linalg.norm(nn) + 1e-12
                    res = p - q00 - du * tu - dv * tv; no = res @ nn
                    out[k] = (r + dv, c + du, no, np.linalg.norm(res - no * nn)); break
                r, c = r + int(np.floor(dv)), c + int(np.floor(du))
        return out


# ---------------------------------------------------------------- claim frame
def a_frame():
    R = json.load(open(REGION)); L = R['levelling']
    return np.array(L['centre_render_px']), np.array(L['line_dir']), np.array(L['down_dir'])


def a_grid_to_uv(rows, cols):
    c, d, n = a_frame(); rows = np.asarray(rows, float); cols = np.asarray(cols, float)
    px = np.stack([cols * G, rows * G], -1) - c
    return (px @ d) / K, (px @ n) / K


def a_uv_to_grid(u, v):
    c, d, n = a_frame(); u = np.asarray(u, float); v = np.asarray(v, float)
    px = c + (u * K)[..., None] * d + (v * K)[..., None] * n
    return px[..., 1] / G, px[..., 0] / G           # rows, cols


def a_uv_to_render(u, v):
    c, d, n = a_frame(); u = np.asarray(u, float); v = np.asarray(v, float)
    px = c + (u * K)[..., None] * d + (v * K)[..., None] * n
    return px[..., 0], px[..., 1]                     # x (col), y (row)


def n7_affine(name):
    return np.array(json.load(open(f'{N7T}/frames/{name}.json'))['grid_to_st'])


def grid_to_uv(name, rows, cols):
    if name == 'A': return a_grid_to_uv(rows, cols)
    M = n7_affine(name); r = np.asarray(rows, float); c = np.asarray(cols, float)
    return M[0, 0] * c + M[0, 1] * r + M[0, 2], M[1, 0] * c + M[1, 1] * r + M[1, 2]


def uv_to_grid(name, u, v):
    if name == 'A': return a_uv_to_grid(u, v)
    if name.startswith('/'):                       # a mesh on the P1 claim-frame carrier grid (u0 -36, v0 -21, 20 vx cells)
        return (np.asarray(v, float) + 21.0) / (G / K), (np.asarray(u, float) + 36.0) / (G / K)
    M = n7_affine(name); Ai = np.linalg.inv(M[:, :2]); u = np.asarray(u, float) - M[0, 2]; v = np.asarray(v, float) - M[1, 2]
    c = Ai[0, 0] * u + Ai[0, 1] * v; r = Ai[1, 0] * u + Ai[1, 1] * v
    return r, c


# ---------------------------------------------------------------- CT volume
class Vol:
    RO = [f'{T}/vcache/20250521133212-8.640um-1.2m-116keV-masked.zarr-dc1c906d14e0d1fe/0',              # own render cache
          f'{W}/x_focus306/vcache/20250521133212-8.640um-1.2m-116keV-masked.zarr-dc1c906d14e0d1fe/0',   # vc_render's mirror
          f'{W}/x_refine/trials/S4_skipped_sheets/chunks', f'{W}/x_refine/trials/N7_sheet_extension/chunks']
    RO_M7 = [f'{W}/x_focus306/vcache/20250521133212-surface-20260413222639-surface-m7-L0-th0.2.zarr-8b644deef31c10cc/0',
             f'{W}/x_refine/trials/S4_skipped_sheets/chunks_m7', f'{W}/x_refine/trials/N7_sheet_extension/chunks_m7']

    def __init__(self, which='volume_url', threads=16, scroll='PHerc0306B'):
        sys.path.insert(0, f'{W}/x_refine/S4_skipped_sheets')
        from vol import Vol as _V
        cache = f'{T}/cache/ct' if which == 'volume_url' else f'{T}/cache/m7'
        self.v = _V(scroll=scroll, cache=cache, threads=threads, which=which)
        if which != 'volume_url':
            self.v.cache = cache
            ro = list(self.RO_M7)
        else:
            ro = list(self.RO)
        self.ro = [p for p in ro if os.path.isdir(p)]
        orig_path = self.v._path

        def _path(k, _orig=orig_path, _ro=self.ro):
            for base in _ro:
                p = f'{base}/{k[0]}/{k[1]}/{k[2]}'
                if os.path.exists(p): return p
            return _orig(k)
        self.v._path = _path
        self.shape = self.v.shape

    def sample(self, P, order=1):
        return self.v.sample(P, order=order)

    def box(self, *a):
        return self.v.box(*a)
