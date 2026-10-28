"""Spectral text-row detector for ink z-maps (8.64 um px).

Rows of letters make the ink map periodic across the rows (line pitch ~3.5-7 mm). Take the 2D power
spectrum of the (windowed, on-surface) map, express it in polar coordinates, and for each direction
measure the power in the pitch band relative to the median power of all directions at the same radius
(direction-whitened). Score = max over directions of that ratio, compared with the same statistic for
spatially shuffled controls of the map (tile shuffle keeps blob statistics, destroys rows).
usage: spectral_rows.py z.npy [...]   -> json per map
"""
import sys, json, numpy as np, cv2
import os as _os; cv2.setNumThreads(int(_os.environ.get('OMP_NUM_THREADS', '4')))  # OpenCV ignores OMP caps; uncapped it used all 32 cores

PX_MM = 1000 / 8.64
DS = 4                                # analyse at 34.6 um/px


def prep(z, qmask=None):
    z = z.astype(np.float32)
    v = z != 0
    if qmask is not None:
        v &= qmask[:v.shape[0], :v.shape[1]] if qmask.shape == v.shape else cv2.resize(qmask.astype(np.uint8), (v.shape[1], v.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    z = cv2.resize(z, (z.shape[1] // DS, z.shape[0] // DS), interpolation=cv2.INTER_AREA)
    v = cv2.resize(v.astype(np.float32), (z.shape[1], z.shape[0]), interpolation=cv2.INTER_AREA) > 0.99
    z = np.clip(z, -1, 4)
    z[~v] = 0
    z[v] -= z[v].mean()
    taper = cv2.GaussianBlur(cv2.erode(v.astype(np.float32), np.ones((9, 9), np.uint8)), (0, 0), 6)
    return z * taper, v


def polar_band_stat(z, v, pmin=3.5, pmax=7.0, nang=36):
    h, w = z.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    F = np.abs(np.fft.fftshift(np.fft.fft2(z * win * v))) ** 2
    yy, xx = np.mgrid[:h, :w]
    fy = (yy - h // 2) / h; fx = (xx - w // 2) / w            # cycles per px
    r = np.hypot(fy, fx) * (PX_MM / DS)                         # cycles per mm
    ang = (np.degrees(np.arctan2(fy, fx)) + 180) % 180
    band = (r >= 1 / pmax) & (r <= 1 / pmin)
    ab = (ang // (180 / nang)).astype(int)
    pw = np.array([F[band & (ab == k)].mean() if (band & (ab == k)).any() else 0 for k in range(nang)])
    return pw / (np.median(pw) + 1e-9)


MIN_TILES = 8


def eligible_tiles(v, t, frac=0.6):
    return [(i, j) for i in range(0, v.shape[0] - t + 1, t) for j in range(0, v.shape[1] - t + 1, t) if v[i:i + t, j:j + t].mean() >= frac]


def shuffle_tiles(z, v, t, rng, pos):
    """permute ink content among mostly-on-sheet tiles; the mask (and its edges) stays fixed"""
    zz = z.copy()
    src = [z[i:i + t, j:j + t].copy() for i, j in pos]
    for (i, j), k in zip(pos, rng.permutation(len(pos))):
        zz[i:i + t, j:j + t] = src[k]
    return zz * v, v


def score(zmap, n_null=12, seed=0, qmask=None):
    z, v = prep(zmap, qmask)
    if v.mean() < 0.15:
        return dict(score=0.0, ratio=0.0, null=0.0, angle=None)
    ratio = polar_band_stat(z, v)
    obs = float(ratio.max())
    rng = np.random.default_rng(seed)
    t = int(2.5 * PX_MM / DS)                                   # 2.5 mm tiles: about a letter, breaks rows
    pos = eligible_tiles(v, t)
    if len(pos) < MIN_TILES:
        return dict(score=None, ratio=obs, null=None, angle=float(np.argmax(ratio) * 5), tiles=len(pos))
    null = [float(polar_band_stat(*shuffle_tiles(z, v, t, rng, pos)).max()) for _ in range(n_null)]
    return dict(score=float((obs - np.mean(null)) / (np.std(null) + 1e-6)), ratio=obs, null=float(np.mean(null)),
                angle=float(np.argmax(ratio) * 5), tiles=len(pos))


def qmask_for(f):
    import os
    q = os.path.join(os.path.dirname(f), 'quality_tiles.npy')
    if not os.path.exists(q):
        return None
    t = np.nan_to_num(np.load(q), nan=-1) >= float(os.environ.get('FL_QUALITY_THR', '0.10'))   # same threshold as the ink --mask
    return np.kron(t, np.ones((128, 128), bool))


if __name__ == '__main__':
    for f in sys.argv[1:]:
        r = score(np.load(f), qmask=qmask_for(f))
        print(f.split('/')[-1][:46], json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)


def count_rows(zmap, angle_deg, qmask=None, min_mm=3.0, max_mm=8.0):
    """project positive ink z perpendicular to the detected rows; count separated peaks and how evenly they are spaced"""
    from scipy.signal import find_peaks
    from scipy.ndimage import gaussian_filter1d
    z, v = prep(zmap, qmask)
    zp = np.clip(z, 0, None)
    h, w = zp.shape
    # angle is the frequency direction; rotate so it becomes vertical -> rows horizontal
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg - 90, 1.0)
    zr = cv2.warpAffine(zp, M, (w, h)); vr = cv2.warpAffine(v.astype(np.float32), M, (w, h))
    prof = zr.sum(1) / np.maximum(vr.sum(1), 1)
    prof[vr.sum(1) < 0.25 * w] = 0
    px_mm = PX_MM / DS
    prof = gaussian_filter1d(prof, 0.35 * px_mm)
    if prof.max() <= 0:
        return dict(n_rows=0)
    pk, pr = find_peaks(prof, distance=int(min_mm * px_mm), prominence=0.35 * prof.max())
    gaps = np.diff(pk) / px_mm
    ok = gaps[(gaps >= min_mm) & (gaps <= max_mm)]
    return dict(n_rows=int(len(pk)), n_gaps_ok=int(len(ok)), gaps_mm=[round(float(g), 1) for g in gaps])
