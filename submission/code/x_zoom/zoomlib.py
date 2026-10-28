"""zoomlib.py: max-quality ink re-read of a small region (ROI) of a surface render, for vetting single leads.

Everything runs on an in-memory crop of the render (ROI + margin), so a 20x20 mm read costs minutes, not hours:
  villa_map   Reader v2 / TRACE (VillaFast, villa's exact robust normalisation and Hann blending), any overlap, 17-layer
              window starting at c + d - 8, optional 8-fold dihedral TTA (rot90 x mirror4).
  hecate_map  Hecate 9.6 um (16 layers resampled from the native grid, centred at c - 0.5 + d like hecate_prep), any
              stride, optional 8-fold dihedral TTA (rot 0..3 x flip), resized back to the crop grid.
  zfuse       plain z-score mean over valid pixels (the lane's fusion).
No super-resolution or generative step: every output pixel is a model probability on measured CT.
"""
import os, sys
import numpy as np, cv2, torch, zarr
cv2.setNumThreads(4)
W = os.environ.get('FL_ROOT', '/home/liam/projects/vesuvius-work')
sys.path.insert(0, f'{W}/x_research2/speed')
import fastink as FI                                                     # noqa: E402
CK = {'rv2': f'{W}/models/community/reader-v2/reader-v2-step040000.pth',
      'trace': f'{W}/models/community/trace/v3B_best.pth'}
_M = {}


def model(name):
    if name not in _M:
        _M[name] = FI.HecateFast(precision='bf16', compile=None) if name == 'hecate' else FI.VillaFast(CK[name], compile=None)
    return _M[name]


def crop(render, y0, y1, x0, x1):
    a = zarr.open(str(render), mode='r')['0']
    D, H, Wd = a.shape
    y0, x0 = max(0, y0), max(0, x0); y1, x1 = min(H, y1), min(Wd, x1)
    return np.asarray(a[:, y0:y1, x0:x1]), (y0, y1, x0, x1)


def _dihedral(k):
    """k in 0..7 -> (rot90 count, flip-x)"""
    return k % 4, k >= 4


def villa_map(name, vol, center, d, face, overlap=0.5, tta8=False, batch=8):
    """vol (D,h,w) uint8 crop; face 'forward' | 'reverse'; returns float32 probability map (h,w)."""
    m = model(name); s = center + d - 8
    idx = np.arange(s, s + m.cfg.input_depth)
    if s < 0 or idx[-1] >= vol.shape[0]:
        raise ValueError(f'layers {s}..{idx[-1]} outside 0..{vol.shape[0] - 1}')
    if face == 'reverse':
        idx = idx[::-1]
    win = vol[idx]
    ks = (0, 1) if tta8 else (0,)                                        # rot90 x mirror4 = 8 dihedral elements
    acc = None
    for r in ks:
        w = np.ascontiguousarray(np.rot90(win, r, axes=(1, 2)))
        a, _ = m.predict(None, 0, window=(w, w), overlap=overlap, batch=batch, tta='mirror4' if tta8 else 'none')
        a = np.rot90(a.astype(np.float32) / 255., -r)
        acc = a if acc is None else acc + a
    return acc / len(ks)


def hecate_prep_np(vol, native_um, z_center, nz=16, target_um=9.6):
    """fastink.hecate_prep on an in-memory crop (same maths)"""
    D, H, Wd = vol.shape; step = target_um / native_um
    pos = z_center + (np.arange(nz) - (nz - 1) / 2.) * step
    if pos.min() < 0 or pos.max() > D - 1:
        raise ValueError(f'hecate z window {pos.min():.2f}..{pos.max():.2f} outside 0..{D - 1}')
    lo = np.floor(pos).astype(int); frac = (pos - lo).astype(np.float32); hi = np.minimum(lo + 1, D - 1)
    s = native_um / target_um; nh, nw = max(1, int(round(H * s))), max(1, int(round(Wd * s)))
    interp = cv2.INTER_AREA if s < 0.95 else cv2.INTER_LINEAR
    out = np.empty((nz, nh, nw), np.uint8)
    for i in range(nz):
        zi = vol[lo[i]].astype(np.float32) * (1 - frac[i]) + vol[hi[i]].astype(np.float32) * frac[i]
        out[i] = np.clip(cv2.resize(zi, (nw, nh), interpolation=interp), 0, 255).astype(np.uint8)
    return out


def hecate_map(vol, um, center, d, face, stride=32, tta8=False, batch=8):
    """center = render read layer c (Hecate is centred at c - 0.5 + d, as hecate_prep's 32.5 for a 66-layer c=33)"""
    m = model('hecate'); prep = hecate_prep_np(vol, um, center - 0.5 + d)
    acc = None; n = 0
    for k in (range(8) if tta8 else (0,)):
        r, fl = _dihedral(k)
        v = prep[:, :, ::-1] if fl else prep
        a, _ = m.predict(np.ascontiguousarray(v), reverse=(face == 'reverse'), stride=stride, batch=batch, rot=r, return_float=True)
        a = a[:, ::-1] if fl else a
        acc = a if acc is None else acc + a; n += 1
    a = acc / n
    return cv2.resize(a.astype(np.float32), (vol.shape[2], vol.shape[1]), interpolation=cv2.INTER_LINEAR)


def zfuse(maps, valid):
    zs = []
    for a in maps:
        v = a[valid]; zs.append((a - v.mean()) / (v.std() + 1e-6))
    return np.mean(zs, 0) * valid
