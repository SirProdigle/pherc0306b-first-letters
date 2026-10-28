"""Window sampling + labels for the held-out known-text items, and simple baseline features.

Protocol (all on the 50 um a-map grid, WIN=128 px = 6.4 mm context windows, stride 16 px):
  usable   : >= 70 % of the window on-sheet (and inside 'evalmask' when the item has one)
  positive : labelled fraction of the central 64x64 (3.2 mm) on-sheet px >= 0.10   ("window centred on text")
  negative : no labelled px anywhere in the window dilated by 0.5 mm                  ("window without text")
  others are ignored.
"""
import numpy as np, cv2
from common import *

STRIDE = 16


def label_windows(it, stride=STRIDE, win=WIN):
    m = it['mask'].copy()
    if 'evalmask' in it:
        m &= it['evalmask']
    lab = it['lab'] > 0.3
    labd = cv2.dilate(lab.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0
    im = cv2.integral(m.astype(np.uint8)); il = cv2.integral((lab & m).astype(np.uint8)); ild = cv2.integral(labd.astype(np.uint8))
    S = lambda ii, y, x, h: ii[y + h, x + h] - ii[y, x + h] - ii[y + h, x] + ii[y, x]
    out = []
    c = win // 4
    for y, x in windows(m.shape, win, stride):
        if S(im, y, x, win) < 0.7 * win * win:
            continue
        on_c = S(im, y + c, x + c, win // 2)
        fc = S(il, y + c, x + c, win // 2) / max(on_c, 1)
        if fc >= 0.10 and on_c > 0.5 * (win // 2) ** 2:
            out.append((y, x, 1))
        elif S(ild, y, x, win) == 0:
            out.append((y, x, 0))
    return out


def blob_feats(a, m):
    """Simple unsupervised baselines on one window (a in [0,1], m on-sheet)."""
    v = a[m]
    f = dict(mean_a=float(v.mean()), frac50=float((v > 0.5).mean()), p95=float(np.percentile(v, 95)))
    b = ((a > 0.5) & m).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    # letter-sized components: area 0.5-16 mm^2 (200-6400 px at 50 um), bbox <= 5 mm
    k = 0; ka = 0
    for i in range(1, n):
        ar = st[i, cv2.CC_STAT_AREA]; w = st[i, cv2.CC_STAT_WIDTH]; h = st[i, cv2.CC_STAT_HEIGHT]
        if 200 <= ar <= 6400 and max(w, h) <= 100:
            k += 1; ka += ar
    f['n_letterblobs'] = k
    f['letterblob_frac'] = ka / max(m.sum(), 1)
    # stroke-likeness: skeleton length per area of the >0.5 mask (thin strokes have high ratio)
    if b.sum() > 30:
        sk = cv2.ximgproc.thinning(b * 255) if hasattr(cv2, 'ximgproc') else None
        f['skel_ratio'] = float((sk > 0).sum() / b.sum()) if sk is not None else 0.0
    else:
        f['skel_ratio'] = 0.0
    return f


if __name__ == '__main__':
    import glob, json, sys
    res = {}
    for p in sorted(glob.glob(f'{G}/cache/heldout/*.npz')):
        it = load_item(p)
        ws = label_windows(it)
        y = np.array([w[2] for w in ws])
        F = [blob_feats(it['a'][a:a + WIN, b:b + WIN], it['mask'][a:a + WIN, b:b + WIN]) for a, b, _ in ws]
        r = dict(npos=int(y.sum()), nneg=int((y == 0).sum()))
        for k in F[0] if F else []:
            s = np.array([f[k] for f in F])
            r[k] = round(auc(s[y == 1], s[y == 0]), 3)
        res[it['name']] = r
        print(it['name'], json.dumps(r), flush=True)
    json.dump(res, open(f'{G}/baseline_heldout.json', 'w'), indent=1)
