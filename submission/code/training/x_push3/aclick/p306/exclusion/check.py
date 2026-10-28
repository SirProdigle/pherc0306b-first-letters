"""check.py : A-click exclusion check (lane P306).  Every lane uses this to assert that no click, KD view, label or statistic
comes from lead A's claim region (any surface shown in any claim image, versions 1-5) or from within 2.0 mm of it in 3-D.

Python:
    import sys; sys.path.insert(0, '<P3>/aclick/p306/exclusion'); import check
    ok = check.allowed('/path/render.zarr', ys, xs)                 # bool array, True = allowed
    d  = check.distance_mm('/path/render.zarr', ys, xs)             # mm to the zone surface (nan = no mesh there)
    m  = check.allowed_mask('/path/render.zarr', roi=(y0, y1, x0, x1), pad_mm=0.4)   # full-res bool mask of a render (or ROI)
    check.assert_allowed(zarr, ys, xs)                               # raises if any pixel is in the zone
`zarr_path_or_mesh` is a render zarr registered in renders.json (or a crop zarr with crop.json pointing at one), or a tifxyz
directory (x.tif/y.tif/z.tif/meta.json; render px = grid / scale, vc_render_tifxyz --scale 1 convention).  For an
unregistered render pass mesh='<tifxyz dir>' (and offset_yx if the zarr is a crop), or add it with `check.py register`.

Rule: a render pixel is mapped to 3-D through its tifxyz mesh (bilinear); it is allowed iff that point is >= radius_mm (2.0)
+ pad_mm from every zone point.  Pixels with no valid mesh are NOT allowed (no 3-D position).  pad_mm: add the half-extent of
whatever the pixel stands for, e.g. a v8-in 64 px input tile around a click -> pad_mm = 0.39 (half diagonal), so the whole
tile is >= 2.0 mm away.  The render slab (+-20 layers = +-0.17 mm along the normal) is covered by a fixed 0.2 mm in
SLAB_MM, added always.

CLI:
    check.py ZARR y,x [y,x ...]           print distance and verdict per point
    check.py ZARR --mask OUT.npz [--roi y0 y1 x0 x1] [--pad 0.4] [--step 4]
    check.py register ZARR MESH [--offset y x]
    check.py selftest
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VX_MM = 0.00864
RADIUS_MM = 2.0
SLAB_MM = 0.2          # render slab half-depth (41 layers, +-20 vx = 0.17 mm) + densify spacing (<= 0.03 mm), rounded up
_TREE = None
_MESH = {}


def _tree():
    global _TREE
    if _TREE is None:
        from scipy.spatial import cKDTree
        _TREE = cKDTree(np.load(f'{HERE}/zone.npz')['pts'].astype(np.float64))
    return _TREE


def _registry():
    p = f'{HERE}/renders.json'
    return json.load(open(p)) if os.path.exists(p) else {}


def resolve(zarr_path_or_mesh, mesh=None, offset_yx=(0, 0)):
    """-> (mesh dir, (oy, ox) offset added to render px before the mesh lookup)."""
    p = os.path.realpath(str(zarr_path_or_mesh).rstrip('/'))
    if mesh: return os.path.realpath(mesh), tuple(offset_yx)
    if os.path.exists(f'{p}/x.tif'): return p, (0, 0)
    reg = _registry()
    if p in reg: return reg[p]['mesh'], tuple(reg[p].get('offset_yx', (0, 0)))
    cj = f'{p}/crop.json'
    if os.path.exists(cj):
        c = json.load(open(cj)); m, (oy, ox) = resolve(c['render']); b = c['box']   # box = y0, y1, x0, x1
        return m, (oy + b[0], ox + b[2])
    raise KeyError(f'{p}: render not registered in {HERE}/renders.json; pass mesh= or run "check.py register ZARR MESH"')


def _load_mesh(mesh):
    if mesh not in _MESH:
        import tifffile
        P = np.stack([tifffile.imread(f'{mesh}/{a}.tif').astype(np.float64) for a in 'xyz'], -1)
        s = float(json.load(open(f'{mesh}/meta.json'))['scale'][0])
        _MESH[mesh] = (P, np.all(P > 0, -1), s)
    return _MESH[mesh]


def px_to_xyz(zarr_path_or_mesh, ys, xs, mesh=None, offset_yx=(0, 0)):
    """render pixels -> (N, 3) volume voxels (x, y, z); nan where the mesh has no valid quad."""
    m, (oy, ox) = resolve(zarr_path_or_mesh, mesh, offset_yx)
    P, valid, s = _load_mesh(m)
    gy = (np.asarray(ys, np.float64).ravel() + oy) * s; gx = (np.asarray(xs, np.float64).ravel() + ox) * s
    H, Wd = valid.shape
    i = np.clip(np.floor(gy).astype(int), 0, H - 2); j = np.clip(np.floor(gx).astype(int), 0, Wd - 2)
    fa = (gy - i)[:, None]; fb = (gx - j)[:, None]
    ok = (gy >= 0) & (gx >= 0) & (gy <= H - 1) & (gx <= Wd - 1) & valid[i, j] & valid[i + 1, j] & valid[i, j + 1] & valid[i + 1, j + 1]
    p = (1 - fa) * (1 - fb) * P[i, j] + fa * (1 - fb) * P[i + 1, j] + (1 - fa) * fb * P[i, j + 1] + fa * fb * P[i + 1, j + 1]
    p[~ok] = np.nan
    return p


def distance_mm(zarr_path_or_mesh, ys, xs, mesh=None, offset_yx=(0, 0), cap_mm=50.0):
    """mm from each pixel's 3-D point to the zone surface (capped at cap_mm); nan where no mesh."""
    p = px_to_xyz(zarr_path_or_mesh, ys, xs, mesh, offset_yx)
    d = np.full(len(p), np.nan)
    ok = np.isfinite(p).all(1)
    if ok.any():
        dd, _ = _tree().query(p[ok], k=1, distance_upper_bound=cap_mm / VX_MM, workers=4)
        d[ok] = np.minimum(dd * VX_MM, cap_mm)
    return d.reshape(np.shape(ys))


def allowed(zarr_path_or_mesh, ys, xs, radius_mm=RADIUS_MM, pad_mm=0.0, mesh=None, offset_yx=(0, 0)):
    """bool array (shape of ys): True iff the pixel's 3-D point is >= radius_mm + SLAB_MM + pad_mm from the zone surface."""
    thr = radius_mm + SLAB_MM + pad_mm
    d = distance_mm(zarr_path_or_mesh, ys, xs, mesh, offset_yx, cap_mm=thr + 0.01)   # bounded query: fast
    return np.nan_to_num(d, nan=-1.0) >= thr


def assert_allowed(zarr_path_or_mesh, ys, xs, **kw):
    ok = allowed(zarr_path_or_mesh, ys, xs, **kw)
    if not np.all(ok):
        raise AssertionError(f'{int((~ok).sum())} of {ok.size} pixels are inside the A-click exclusion zone (or have no mesh)')
    return True


def allowed_mask(zarr_path_or_mesh, roi=None, step=4, radius_mm=RADIUS_MM, pad_mm=0.0, mesh=None, offset_yx=(0, 0), shape=None):
    """full-resolution bool mask over a render (or roi y0, y1, x0, x1).  Distances are computed every `step` px and the
    verdict uses an extra step*0.71 px tolerance, so a pixel between samples can never be wrongly allowed."""
    if roi is None:
        if shape is None:
            import zarr
            shape = zarr.open(os.path.realpath(zarr_path_or_mesh), mode='r')['0'].shape[-2:]
        roi = (0, shape[0], 0, shape[1])
    y0, y1, x0, x1 = roi
    gy = np.arange(y0, y1 + step, step); gx = np.arange(x0, x1 + step, step)
    YY, XX = np.meshgrid(gy, gx, indexing='ij')
    thr = radius_mm + SLAB_MM + pad_mm + step * 0.71 * VX_MM
    d = distance_mm(zarr_path_or_mesh, YY, XX, mesh, offset_yx, cap_mm=thr + 0.01)
    ok = np.nan_to_num(d, nan=-1.0) >= thr
    # a full-res pixel is allowed iff all 4 surrounding samples are allowed
    q = ok[:-1, :-1] & ok[1:, :-1] & ok[:-1, 1:] & ok[1:, 1:]
    full = np.repeat(np.repeat(q, step, 0), step, 1)
    return full[:y1 - y0, :x1 - x0]


def register(zarr_path, mesh, offset_yx=(0, 0), note=''):
    reg = _registry(); p = os.path.realpath(zarr_path)
    reg[p] = dict(mesh=os.path.realpath(mesh), offset_yx=list(offset_yx), note=note)
    json.dump(reg, open(f'{HERE}/renders.json', 'w'), indent=1)


def selftest():
    """known points: lead A pixels -> rejected; pick 2 -> allowed fraction (prints)."""
    W = '/home/liam/projects/vesuvius-work'
    K = 115.741
    V5 = f'{W}/x_refine/trials/P1_surface_refine/renders/P1V5Rc.zarr'
    rng = np.random.default_rng(0)
    res = {}
    # 1. V5Rc box corners and centre + random pixels inside the context window -> all rejected (where mesh valid)
    u = rng.uniform(-30, 30, 4000); v = rng.uniform(-20, 15, 4000)
    xs, ys = (u + 36) * K, (v + 21) * K
    d = distance_mm(V5, ys, xs); ok = allowed(V5, ys, xs)
    res['V5Rc_context_random'] = dict(n=4000, with_mesh=int(np.isfinite(d).sum()), allowed=int(ok.sum()))
    assert ok.sum() == 0
    # 2. the same 3-D places seen through other renders: lane w074 render and w074own (v1 mesh) -> the line region rejected
    for z in ('x_focus306/renders/w074own.zarr', 'x_focus306/renders/w074.zarr'):
        m = allowed_mask(f'{W}/{z}', step=8)
        res[z] = dict(allowed_frac=float(m.mean()))
    # 3. pick 2 (z8400 w050 row render)
    m = allowed_mask(f'{W}/x_focus306/renders/w050.zarr', step=4)
    res['w050_allowed_frac_all_px'] = float(m.mean())
    print(json.dumps(res, indent=1))
    return res


if __name__ == '__main__':
    a = sys.argv[1:]
    if not a: print(__doc__); sys.exit(0)
    if a[0] == 'selftest': selftest(); sys.exit(0)
    if a[0] == 'register':
        off = (int(a[a.index('--offset') + 1]), int(a[a.index('--offset') + 2])) if '--offset' in a else (0, 0)
        register(a[1], a[2], off); print('registered', a[1]); sys.exit(0)
    z = a[0]
    if '--mask' in a:
        out = a[a.index('--mask') + 1]
        roi = tuple(int(t) for t in a[a.index('--roi') + 1:a.index('--roi') + 5]) if '--roi' in a else None
        pad = float(a[a.index('--pad') + 1]) if '--pad' in a else 0.0
        step = int(a[a.index('--step') + 1]) if '--step' in a else 4
        m = allowed_mask(z, roi=roi, pad_mm=pad, step=step)
        np.savez_compressed(out, allowed=m, roi=np.array(roi if roi else (0, m.shape[0], 0, m.shape[1])), zarr=z, pad_mm=pad)
        print(f'{out}: allowed {m.mean():.4f} of {m.size} px'); sys.exit(0)
    pts = [tuple(int(t) for t in s.split(',')) for s in a[1:]]
    ys = np.array([p[0] for p in pts]); xs = np.array([p[1] for p in pts])
    d = distance_mm(z, ys, xs); ok = allowed(z, ys, xs)
    for (y, x), di, o in zip(pts, d, ok): print(f'{y},{x}  {di:.3f} mm  {"ALLOWED" if o else "REJECTED"}')
    sys.exit(0 if ok.all() else 1)
