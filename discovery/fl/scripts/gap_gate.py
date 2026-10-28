"""gap_gate.py: is the ink signal sitting on cracks / delamination gaps?

gap map  = black-top-hat of the min-projection of the CT over layers c+d-K..c+d+K (dark, thin, in-plane gaps within
           the sheet's own layers), smoothed; crack pixels = top-hat above the on-sheet 90th percentile.
enrich   = P(crack-adjacent | ink z > T) / P(crack-adjacent | on-sheet)    (1 = ink ignores cracks, >1 = ink hugs cracks)
usage: python gap_gate.py RENDER.zarr INKMAP [--d 0] [--centre C] [--um 8.64]   (INKMAP: png/tif/npy on the render grid)
"""
import sys, argparse, json
import numpy as np, cv2, zarr, tifffile
import os as _os; cv2.setNumThreads(int(_os.environ.get('OMP_NUM_THREADS', '4')))  # OpenCV ignores OMP caps; uncapped it used all 32 cores
GAP_T = 0.3                                            # at least 30 % darker than the local (2 mm) mean


def load(f, shape):
    a = cv2.imread(f, cv2.IMREAD_UNCHANGED) if f.endswith('.png') else (np.load(f) if f.endswith('.npy') else tifffile.imread(f))
    a = np.nan_to_num(a.astype(np.float32))
    return cv2.resize(a, (shape[1], shape[0])) if a.shape != shape else a


def gap_stats(render, ink, d=0, centre=None, um=8.64, K=3, T=2.0, mask=None, dump=None, near_mm=0.3):
    v = zarr.open(str(render), mode='r')['0']
    c = v.shape[0] // 2 if centre is None else centre
    st = np.asarray(v[c + d - K:c + d + K + 1]).astype(np.float32)
    valid = (st > 0).all(0)
    valid = cv2.erode(valid.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    if mask is not None:
        valid &= mask
    mn = st.min(0)
    px = 1000.0 / um                                   # px per mm
    ms = cv2.GaussianBlur(mn, (0, 0), 0.15 * px)
    loc = cv2.GaussianBlur(np.where(valid, mn, 0), (0, 0), 2 * px) / (cv2.GaussianBlur(valid.astype(np.float32), (0, 0), 2 * px) + 1e-6)
    g = 1 - ms / (loc + 1e-3)                          # relative darkness of the thin-slab min-projection
    crack = (g > max(GAP_T, np.percentile(g[valid], 95))) & valid
    n, lab, stats, _ = cv2.connectedComponentsWithStats(crack.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    for i in range(1, n):                              # keep long structures only (>= 1.5 mm extent)
        keep[i] = max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) >= 1.5 * px
    crack = keep[lab]
    near = cv2.dilate(crack.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(near_mm * px) | 1,) * 2)) > 0
    if dump:
        cv2.imwrite(dump, np.dstack([crack * 255, np.clip(ink * 0, 0, 0), ((ink - ink[valid].mean()) / (ink[valid].std() + 1e-6) > T) * 255]).astype(np.uint8))
    z = ink.copy(); z = (z - z[valid].mean()) / (z[valid].std() + 1e-6)
    hot = (z > T) & valid
    p_all = near[valid].mean(); p_hot = near[hot].mean() if hot.any() else float('nan')
    gz = cv2.GaussianBlur(g, (0, 0), 0.3 * px); iz = cv2.GaussianBlur(z, (0, 0), 0.3 * px)
    corr = float(np.corrcoef(gz[valid], iz[valid])[0, 1])
    return dict(enrich=round(float(p_hot / p_all), 3), p_near_hot=round(float(p_hot), 3), p_near_all=round(float(p_all), 3),
                gap_corr=round(corr, 3), hot_frac=round(float(hot[valid].mean()), 4))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('render'); ap.add_argument('ink'); ap.add_argument('--d', type=int, default=0)
    ap.add_argument('--centre', type=int); ap.add_argument('--um', type=float, default=8.64); ap.add_argument('--K', type=int, default=3); ap.add_argument('--dump')
    a = ap.parse_args()
    shape = zarr.open(a.render, mode='r')['0'].shape[1:]
    print(json.dumps(gap_stats(a.render, load(a.ink, shape), a.d, a.centre, a.um, a.K, dump=a.dump)))
