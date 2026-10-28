"""Shared helpers for the glyph-likeness detector (x_glyph).

Everything is resampled to a common physical grid: UM = 50 µm per pixel. Windows are WIN x WIN px (6.4 mm).
A "map item" is a dict: name, scroll, um, model, face, z (float32, z-scored over on-sheet px at 50 µm),
mask (bool on-sheet at 50 µm), optional lab (float32 fraction of labelled px, at 50 µm).
"""
import os, json, glob
import numpy as np, cv2, tifffile
cv2.setNumThreads(2)

W = '/home/liam/projects/vesuvius-work'
G = f'{W}/x_glyph'
UM = 50.0
WIN = 128
SCROLL_UM = {k: v['voxel_um'] for k, v in json.load(open(f'{W}/fl/scrolls.json')).items()}
SCROLL_UM['PHerc1447'] = 7.91   # HF w058 surfaces (v8-in "8 um" model); exact value matters little at 50 um


def scroll_of(path):
    for part in path.split('/'):
        p = part.split('@')[0].split('__')[0]
        if p in SCROLL_UM:
            return p
        if part.startswith('PHerc') and '_' in part:
            q = part.split('_')[0]
            if q in SCROLL_UM:
                return q
        if part[:4].isdigit() and part[:4] in ('0125', '0175', '0306', '0343', '0191', '0813', '0800'):
            for k in SCROLL_UM:
                if k[5:].startswith(part.split('_')[0]):
                    return k
    return None


def imread_any(m):
    if m.endswith('.npy'):
        return np.load(m).astype(np.float32)
    if m.endswith('.tif') or m.endswith('.tiff'):
        return tifffile.imread(m).astype(np.float32)
    return cv2.imread(m, cv2.IMREAD_UNCHANGED).astype(np.float32)


def resample(a, um, interp=cv2.INTER_AREA):
    f = um / UM
    h, w = a.shape
    return cv2.resize(a.astype(np.float32), (max(1, int(round(w * f))), max(1, int(round(h * f)))), interpolation=interp)


def tiles_mask(qt, shape, thr=0.10, tile=128):
    q = np.nan_to_num(qt, nan=-1) >= thr
    m = np.kron(q, np.ones((tile, tile), bool))
    out = np.zeros(shape, bool)
    h, w = min(m.shape[0], shape[0]), min(m.shape[1], shape[1])
    out[:h, :w] = m[:h, :w]
    return out


def zscore(a, m):
    v = a[m]
    if v.size < 100:
        return np.zeros_like(a)
    return ((a - v.mean()) / (v.std() + 1e-6)).astype(np.float32)


def fuse_z(maps, weights, stat):
    """fuse_view.py logic: weighted mean of per-map z (over stat px), re-z-scored over stat."""
    acc = 0.0
    for a, w in zip(maps, weights):
        v = a[stat]; acc = acc + w * (a - v.mean()) / (v.std() + 1e-6)
    z = acc / sum(weights)
    v = z[stat]
    return ((z - v.mean()) / (v.std() + 1e-6)).astype(np.float32)


def a_from_z(z, um, valid, down=3):
    """Overlay representation used in review galleries: z (native res, 0 off-valid) -> 1/down res (AREA) ->
    a = clip((z-0.9)/1.8, 0, 1) -> resampled to UM. Returns (a50, mask50)."""
    z = np.where(valid, z, 0).astype(np.float32)
    h, w = z.shape
    z3 = cv2.resize(z, (w // down, h // down), interpolation=cv2.INTER_AREA)
    v3 = cv2.resize(valid.astype(np.float32), (w // down, h // down), interpolation=cv2.INTER_AREA)
    return a_from_a3(np.clip((z3 - 0.9) / 1.8, 0, 1) * (v3 > 0.5), v3, um * down)


def a_from_a3(a3, v3, um3):
    a = resample(a3.astype(np.float32), um3)
    m = resample(v3.astype(np.float32), um3) > 0.6
    a[~m] = 0
    return a.astype(np.float32), m


def make_item(name, z, valid, um, lab=None, **meta):
    """z: native-res z map (already z-scored over on-sheet px), valid: native on-sheet mask, lab: native labels."""
    a, m = a_from_z(z, um, valid)
    it = dict(name=name, um=um, a=a, mask=m, **meta)
    if lab is not None:
        it['lab'] = resample(lab.astype(np.float32), um)[:a.shape[0], :a.shape[1]]
        if it['lab'].shape != a.shape:
            it['lab'] = cv2.resize(it['lab'], (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
    return it


def save_item(it, path):
    np.savez_compressed(path, **{k: (v if isinstance(v, np.ndarray) else np.array(v)) for k, v in it.items()})


def load_item(path):
    d = np.load(path, allow_pickle=False)
    it = {k: d[k] for k in d.files}
    for k in list(it):
        if it[k].ndim == 0:
            it[k] = it[k].item()
    return it


def windows(shape, win=WIN, step=32):
    H, Wd = shape
    ys = list(range(0, max(1, H - win + 1), step))
    xs = list(range(0, max(1, Wd - win + 1), step))
    if ys and ys[-1] != H - win and H > win:
        ys.append(H - win)
    if xs and xs[-1] != Wd - win and Wd > win:
        xs.append(Wd - win)
    return [(y, x) for y in ys for x in xs]


def onsheet_windows(it, win=WIN, step=32, min_frac=0.7):
    m = it['mask']
    ii = cv2.integral(m.astype(np.uint8))
    out = []
    for y, x in windows(m.shape, win, step):
        s = ii[y + win, x + win] - ii[y, x + win] - ii[y + win, x] + ii[y, x]
        if s >= min_frac * win * win:
            out.append((y, x))
    return out




def auc(pos, neg):
    pos = np.asarray(pos, np.float64); neg = np.asarray(neg, np.float64)
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind='mergesort')
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks); ranks = sums[inv] / cnt[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
