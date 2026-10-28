"""stroke_feats.py (fl/scripts copy, used by tri_rank.py): per-window stroke features on the fused ink z-map (own implementation from a written description of
skuzbucket1's stroke-score idea; none of their code was fetched or run).

hot = z > 2 on-sheet; marks = 8-connected components of hot >= MIN_PX pixels; each mark's length/width/orientation from its
second moments (equivalent-ellipse full axes 4*sqrt(eigenvalue)).
  shape = share of hot area in marks >= 1 mm long and >= 2x as long as wide          (strokes vs round blobs)
  horiz = share of those long marks (count) oriented within 30 deg of render x     (crack-edge streaks run along the winding)
  recur = max over w-2, w+2 of the correlation between the window's along-winding hot-density profile and the profile at the
          same 3-D spot on that winding (20 px tifxyz cells, nearest 3-D grid point within MAXD voxels). w+-2, not w+-1:
          real text bleeds onto w+-1 (0343 positive control).  Folds/cracks repeat turn to turn; text should not.
"""
import os, glob, numpy as np, cv2, tifffile
import os as _os; cv2.setNumThreads(int(_os.environ.get('OMP_NUM_THREADS', '4')))  # OpenCV ignores OMP caps; uncapped it used all 32 cores
from scipy.spatial import cKDTree
WIN, B, MIN_PX, MAXD = 2600, 20, 12, 60.0


def marks(z, valid, um):
    hot = ((np.nan_to_num(z) > 2) & valid).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(hot, connectivity=8)
    out = []
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] < MIN_PX: continue
        x, y, w, h = st[k, :4]
        yy, xx = np.nonzero(lab[y:y + h, x:x + w] == k)
        if len(xx) < 3: continue
        c = np.cov(np.stack([xx, yy]).astype(np.float64)) + np.eye(2) * 1 / 12
        ev, evec = np.linalg.eigh(c)
        L, Wd = 4 * np.sqrt(max(ev[1], 1e-9)), 4 * np.sqrt(max(ev[0], 1e-9))
        ang = np.degrees(np.arctan2(evec[1, 1], evec[0, 1])) % 180
        out.append(dict(area=len(xx), len_mm=L * um / 1000, aspect=L / max(Wd, 1e-6), ang=ang))
    return out


def shape_horiz(z, valid, um):
    M = marks(z, valid, um)
    tot = sum(m['area'] for m in M)
    longm = [m for m in M if m['len_mm'] >= 1.0 and m['aspect'] >= 2.0]
    shape = sum(m['area'] for m in longm) / tot if tot else None
    horiz = (sum(1 for m in longm if min(m['ang'], 180 - m['ang']) <= 30) / len(longm)) if len(longm) >= 3 else None
    return shape, horiz, len(M), len(longm)


_cache = {}
def wind(maps_root, tag, w, flat):
    """(hot-fraction grid [gh,gw], valid, xyz) for winding w; hot = max over both faces of z>2, 20 px cells"""
    key = (maps_root, tag, w)
    if key in _cache: return _cache[key]
    d = os.path.join(maps_root, f'{tag}_{w}')
    t = glob.glob(os.path.join(flat, w, 'tifxyz', '*.tifxyz')) or glob.glob(os.path.join(flat, w, '*.tifxyz'))
    zs = glob.glob(os.path.join(d, '*_d0_*_z.npy'))
    if not t or not zs: _cache[key] = None; return None
    xyz = np.stack([tifffile.imread(os.path.join(t[0], c + '.tif')) for c in 'xyz'], -1); gh, gw = xyz.shape[:2]
    hot = np.zeros((gh, gw), np.float32)
    for f in zs:
        z = np.load(f, mmap_mode='r'); H, Wd = z.shape; h, wd = min(H, gh * B) // B * B, min(Wd, gw * B) // B * B
        blk = (np.nan_to_num(np.asarray(z[:h, :wd], np.float32)) > 2).reshape(h // B, B, wd // B, B).mean((1, 3))
        hot[:h // B, :wd // B] = np.maximum(hot[:h // B, :wd // B], blk)
    _cache[key] = (hot, xyz[..., 0] > 0, xyz); return _cache[key]


def recur(maps_root, tag, w, x0, flat):
    me = wind(maps_root, tag, w, flat)
    if me is None: return None, {}
    hot, valid, xyz = me; g0, g1 = x0 // B, min((x0 + WIN) // B, hot.shape[1])
    sub_v = valid[:, g0:g1]; own = np.where(sub_v, hot[:, g0:g1], np.nan)
    k = int(w[1:]); res = {}
    for dn in (-2, 2):
        nb = wind(maps_root, tag, f'w{k + dn:03d}', flat)
        if nb is None: continue
        nh, nv, nx = nb
        tree = cKDTree(nx[nv]); pts = xyz[:, g0:g1][sub_v]
        dist, idx = tree.query(pts, distance_upper_bound=MAXD); ok = np.isfinite(dist)
        if ok.mean() < 0.3: res[dn] = None; continue
        other = np.full(own.shape, np.nan); vals = np.full(len(pts), np.nan); vals[ok] = nh[nv][idx[ok]]
        other[sub_v] = vals
        both = ~np.isnan(own) & ~np.isnan(other)
        cnt = both.sum(0); keep = cnt >= 5
        if keep.sum() < 20: res[dn] = None; continue
        po = np.where(both, own, 0).sum(0)[keep] / cnt[keep]; pn = np.where(both, other, 0).sum(0)[keep] / cnt[keep]
        po = np.convolve(po, np.ones(3) / 3, 'same'); pn = np.convolve(pn, np.ones(3) / 3, 'same')
        res[dn] = round(float(np.corrcoef(po, pn)[0, 1]), 3) if po.std() > 0 and pn.std() > 0 else None
    vals = [v for v in res.values() if v is not None]
    return (max(vals) if vals else None), res


def texture_energy(ct, valid, um):
    """papyrus presence: std of fibre-scale band-pass CT / local mean, on-sheet. Smooth mounting material ~0.13, papyrus >= 0.23"""
    px = 1000 / um; x = ct.astype(np.float32)
    lo = cv2.GaussianBlur(x, (0, 0), 0.8 * px); hi = x - lo; bp = hi - cv2.GaussianBlur(hi, (0, 0), 0.075 * px)
    vv = cv2.erode(valid.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    return float(np.std(bp[vv]) / (np.mean(lo[vv]) + 1e-6)) if vv.sum() > 1000 else None
