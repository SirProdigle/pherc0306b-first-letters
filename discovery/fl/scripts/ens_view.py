"""Ensemble ink maps (per-model z-score over valid pixels, mean), write preview PNGs.
usage: ens_view.py OUT_PREFIX SURFACE_ZARR LAYER  map1.tif [map2 ...] [--crop y0,y1,x0,x1] [--down N] [--rot180]
"""
import sys, numpy as np, tifffile, cv2, zarr
args = sys.argv[1:]
crop = None; down = 3; rot = False
if '--crop' in args:
    i = args.index('--crop'); crop = [int(v) for v in args[i+1].split(',')]; del args[i:i+2]
if '--down' in args:
    i = args.index('--down'); down = int(args[i+1]); del args[i:i+2]
if '--rot180' in args:
    args.remove('--rot180'); rot = True
out, surf, layer, maps = args[0], args[1], int(args[2]), args[3:]
ct = np.asarray(zarr.open(surf, mode='r')['0'][layer]).astype(np.float32)
valid = ct > 0
import os as _os
_qf = _os.path.join(surf, 'quality_tiles.npy')
stat = valid
if _os.path.exists(_qf):                 # z-score over on-sheet tiles only (off-sheet pixels would inflate on-sheet z)
    _q = np.load(_qf); _q = np.nan_to_num(_q, nan=-1) >= 0.10
    _m = np.kron(_q, np.ones((128, 128), bool))
    _mm = np.zeros_like(valid); h_, w_ = min(_m.shape[0], valid.shape[0]), min(_m.shape[1], valid.shape[1])
    _mm[:h_, :w_] = _m[:h_, :w_]
    if (_mm & valid).sum() > 0.2 * valid.sum():
        stat = _mm & valid
zs = []
for m in maps:
    a = cv2.imread(m, cv2.IMREAD_UNCHANGED) if m.endswith('.png') else tifffile.imread(m)
    a = a.astype(np.float32)
    if a.shape != ct.shape:
        a = cv2.resize(a, (ct.shape[1], ct.shape[0]), interpolation=cv2.INTER_LINEAR)
    v = a[stat]
    zs.append((a - v.mean()) / (v.std() + 1e-6))
z = np.mean(zs, 0); z[~valid] = 0
B = 48                                   # models fire along the render's outer border: blank it
z[:B] = 0; z[-B:] = 0; z[:, :B] = 0; z[:, -B:] = 0
np.save(out + '_z.npy', z.astype(np.float16))
def prep(img):
    if crop: img = img[crop[0]:crop[1], crop[2]:crop[3]]
    if rot: img = img[::-1, ::-1]
    if down > 1: img = cv2.resize(img, (img.shape[1]//down, img.shape[0]//down), interpolation=cv2.INTER_AREA)
    return img
c = prep(ct); zz = prep(z); vv = prep(valid.astype(np.float32)) > 0.5
lo, hi = np.percentile(c[vv], [1, 99.5]) if vv.any() else (0, 255)
g = np.clip((c - lo) / (hi - lo + 1e-6), 0, 1) * 0.45
alpha = np.clip((zz - 0.9) / (2.7 - 0.9), 0, 1)
rgb = np.stack([g*(1-alpha) + alpha*0.35, g*(1-alpha) + alpha*0.85, g*(1-alpha) + alpha*0.95], -1)  # BGR: yellow
cv2.imwrite(out + '_overlay.png', (rgb*255).astype(np.uint8))
zg = np.clip((zz + 1.0) / 4.0, 0, 1); zg[~vv] = 0
cv2.imwrite(out + '_z.png', (zg*255).astype(np.uint8))
print('wrote', out, 'z stats', float(z[valid].mean()), float(z[valid].std()), 'frac>2', float((z[valid] > 2).mean()))
