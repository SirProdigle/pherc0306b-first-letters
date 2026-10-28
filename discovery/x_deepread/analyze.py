"""analyze.py : local analysis of the cloud deep-read maps (x_deepread/results/b*/out/<name>/).

For each lead surface (rows given as a polygon in that surface's render px):
  * line profile perpendicular to the row: hot share (fused z > 2) in 0.5 mm bands at offset t from the row, over the
    row's own span and over the full surface width, as a multiple of the surface's base hot rate.  Yardstick: 0343 text
    through the same profile shows its second row at -6.04 mm, 6.25x base (x_focus/out/A_own/own_patch.json ref_0343).
  * row continuation: 2 mm along-row bins at >= 2x base, contiguous extent from the row centre.
  * full-page ink-as-page views (fixed contrast z 1..3, 1 cm bar), neighbour strips, native-resolution close-ups
    (row horizontal, line +-8 mm, ~116 px/mm) with CT layer 20 below.
Outputs: CANDIDATES/1_closest_calls/deepread/ (images) and x_deepread/analysis.json.
"""
import os, sys, glob, json
import numpy as np, cv2
cv2.setNumThreads(4)
W = '/home/liam/projects/vesuvius-work'; D = f'{W}/x_deepread'; OUT = f'{W}/CANDIDATES/1_closest_calls/deepread'
os.makedirs(OUT, exist_ok=True)
UM = 8.64; PXMM = 1000 / UM
YARD = dict(second_row_t_mm=-6.04, x_base=6.25)


def find(name):
    for n in (name, name + '_f'):                         # b1's reordered fwd-only jobs carry an _f suffix
        f = glob.glob(f'{D}/results/b*/out/{n}/fwd_fused.npy')
        if f: return os.path.dirname(f[0])
    return None


def load(name, face='fwd'):
    d = find(name)
    if d is None or not os.path.exists(f'{d}/{face}_fused.npy'): return None, None
    try:
        z = np.load(f'{d}/{face}_fused.npy').astype(np.float32)
    except Exception:                                     # still being pulled
        return None, None
    ct = cv2.imread(f'{d}/ct20.png', cv2.IMREAD_GRAYSCALE)
    return z, ct


def page(z, valid):
    a = np.clip((z - 1) / 2, 0, 1); a[~valid] = 0
    p = np.where(valid, 1 - 0.9 * a, 0.15)
    return (np.stack([p * 0.80, p * 0.88, p * 0.92], -1) * 255).astype(np.uint8)


def bar(img, pxmm, label='1 cm'):
    h = img.shape[0]; L = int(10 * pxmm)
    cv2.line(img, (12, h - 14), (12 + L, h - 14), (0, 0, 255), 4); cv2.putText(img, label, (12, h - 22), 0, 0.7, (0, 0, 255), 2)
    return img


def head(img, text):
    hd = np.full((34, img.shape[1], 3), 25, np.uint8); cv2.putText(hd, text, (8, 24), 0, 0.72, (255, 255, 255), 2)
    return np.concatenate([hd, img], 0)


def row_line(z, poly):
    """fit the row from z>2 pixels inside poly: centre c, unit direction u (row), unit normal n"""
    m = np.zeros(z.shape, np.uint8); cv2.fillPoly(m, [np.array(poly, np.int32)], 1)
    ys, xs = np.nonzero((m > 0) & (z > 2))
    P = np.stack([xs, ys], 1).astype(np.float64); c = P.mean(0)
    u = np.linalg.svd(P - c, full_matrices=False)[2][0]
    if u[0] < 0: u = -u
    n = np.array([u[1], -u[0]])                                   # n points "up" (towards smaller y) when u ~ +x
    s = (P - c) @ u
    return c, u, n, (np.percentile(s, 3) / PXMM, np.percentile(s, 97) / PXMM)


def profile(z, valid, c, u, n, span):
    ys, xs = np.nonzero(valid); P = np.stack([xs, ys], 1).astype(np.float64) - c
    t = (P @ n) / PXMM; s = (P @ u) / PXMM; hot = z[ys, xs] > 2; base = hot.mean()
    bins = np.arange(-15, 15.01, 0.5); res = {}
    for key, sel in (('row_span', (s >= span[0]) & (s <= span[1])), ('full_width', np.ones_like(s, bool))):
        prof = []
        for b0 in bins[:-1]:
            k = sel & (t >= b0) & (t < b0 + 0.5)
            prof.append(round(float(hot[k].mean() / base), 2) if k.sum() > 200 else None)
        res[key] = prof
    # continuation along the row band (|t| <= 1.5 mm), 2 mm bins
    band = np.abs(t) <= 1.5; sb = np.arange(np.floor(s.min()), np.ceil(s.max()) + 0.01, 2.0); along = []
    for b0 in sb[:-1]:
        k = band & (s >= b0) & (s < b0 + 2)
        along.append((round(float(b0), 1), round(float(hot[k].mean() / base), 2) if k.sum() > 100 else None))
    return dict(base_rate=round(float(base), 4), t_bins_mm=[round(float(b), 1) for b in bins[:-1]], **res, along=along)


def peaks(prof, tb, thr=2.4):
    out = []
    for i in range(1, len(prof) - 1):
        v = prof[i]
        if v is None or abs(tb[i] + 0.25) < 2.0: continue            # skip the row itself (|t| < 2 mm)
        if v >= thr and all(prof[j] is None or v >= prof[j] for j in (i - 1, i + 1)):
            out.append(dict(t_mm=tb[i] + 0.25, x_base=v))
    return out


def extent(along):
    """contiguous run of >= 2x base bins through the row centre (s=0) -> (s_lo, s_hi) mm"""
    ok = {b: (v is not None and v >= 2) for b, v in along}
    keys = sorted(ok); c = min(keys, key=lambda b: abs(b + 1))
    if not ok[c]: return None
    i = keys.index(c); lo = hi = i
    while lo > 0 and ok[keys[lo - 1]]: lo -= 1
    while hi < len(keys) - 1 and ok[keys[hi + 1]]: hi += 1
    return [keys[lo], keys[hi] + 2]


def closeup(name, z, ct, c, u, span, pad_mm=8, tag=''):
    ang = np.degrees(np.arctan2(u[1], u[0]))
    M = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), ang, 1.0)
    zr = cv2.warpAffine(z, M, (z.shape[1], z.shape[0]), flags=cv2.INTER_LINEAR)
    cr = cv2.warpAffine(ct, M, (ct.shape[1], ct.shape[0]), flags=cv2.INTER_LINEAR)
    x0 = int(c[0] + (span[0] - 3) * PXMM); x1 = int(c[0] + (span[1] + 3) * PXMM)
    y0 = int(c[1] - pad_mm * PXMM); y1 = int(c[1] + pad_mm * PXMM)
    x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(z.shape[1], x1), min(z.shape[0], y1)
    zz, cc = zr[y0:y1, x0:x1], cr[y0:y1, x0:x1]; v = cc > 0
    pg = page(zz, v)
    lo, hi = np.percentile(cc[v], [1, 99.5]) if v.any() else (0, 255)
    g = np.clip((cc.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1); g[~v] = 0
    g = cv2.cvtColor(cv2.createCLAHE(2.0, (8, 8)).apply((g * 255).astype(np.uint8)), cv2.COLOR_GRAY2BGR)
    for im in (pg, g):                                            # ticks every 1 mm along the left edge, row line marks
        for k in range(0, im.shape[0], int(PXMM)): cv2.line(im, (0, k), (10, k), (0, 0, 255), 1)
        bar(im, PXMM, '1 cm')
    a = head(pg, f'{name}{tag}: max-quality fused ink as a page (dark = ink), row horizontal, native {UM} um/px (~{PXMM:.0f} px/mm)')
    b = head(g, f'{name}{tag}: bare CT layer 20 (read layer), same crop')
    out = f'{OUT}/{name}{tag}_CLOSEUP_native.jpg'
    cv2.imwrite(out, np.concatenate([a, np.full((10, a.shape[1], 3), 60, np.uint8), b], 0), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out


def full_page(name, z, ct, poly=None, face='fwd', scale=1 / 3):
    v = (ct > 0) if ct is not None else (z != 0)
    img = page(z, v)
    if poly is not None: cv2.polylines(img, [np.array(poly, np.int32)], True, (255, 150, 0), 6)
    img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    img = head(bar(img, PXMM * scale), f'{name} {face} face, max-quality fused (Hecate s16 + Reader v2 + TRACE, 8x TTA, d-1..+1), whole surface')
    out = f'{OUT}/{name}_{face}_FULL_PAGE.jpg'; cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 88]); return out


def strip(names, xr, title, out_name, scale=1 / 3):
    rows = []
    for n in names:
        z, ct = load(n)
        if z is None: continue
        zz = z[:, xr[0]:xr[1]]; v = (ct[:, xr[0]:xr[1]] > 0) if ct is not None else zz != 0
        ys = np.nonzero(v.any(1))[0]
        if len(ys): zz, v = zz[ys.min():ys.max() + 1], v[ys.min():ys.max() + 1]
        img = page(zz, v); img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        rows.append(head(img, f'{n} fwd, x {xr[0]}-{xr[1]}'))
    if not rows: return None
    w = max(r.shape[1] for r in rows); rows = [np.pad(r, ((0, 6), (0, w - r.shape[1]), (0, 0)), constant_values=40) for r in rows]
    img = head(bar(np.concatenate(rows, 0), PXMM * scale), title)
    out = f'{OUT}/{out_name}.jpg'; cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 86]); return out


if __name__ == '__main__':
    ONLY = set(sys.argv[1:])
    j = lambda p: json.load(open(p))
    leads = {
        'w074': j(f'{W}/x_focus306/out/w074_row/result.json')['poly'],
        'w050': j(f'{W}/x_focus306/out/w050_row/result.json')['poly'],
        'patchW074': j(f'{W}/x_focus306/loops/own_row.json')['poly_render'],
    }
    A = j(f'{W}/x_focus/out/A_own/own_patch.json')['row_fit']               # user's row on its own-sheet patch: a fitted line
    c, u = np.array(A['centre_px']), np.array(A['u']); s0, s1 = A['s_range_mm']; n = np.array([u[1], -u[0]])
    corners = [c + u * s0 * PXMM + n * 1.5 * PXMM, c + u * s1 * PXMM + n * 1.5 * PXMM, c + u * s1 * PXMM - n * 1.5 * PXMM, c + u * s0 * PXMM - n * 1.5 * PXMM]
    leads['patchA'] = [[float(p[0]), float(p[1])] for p in corners]
    res = {}
    for name, poly in leads.items():
        if ONLY and name not in ONLY: continue
        z, ct = load(name)
        if z is None: res[name] = 'not available yet'; continue
        valid = (ct > 0) if ct is not None else (z != 0)
        c, u, n, span = row_line(z, poly)
        pr = profile(z, valid, c, u, n, span)
        r = dict(row_centre_px=[round(float(x), 1) for x in c], row_dir=[round(float(x), 3) for x in u], row_span_mm=[round(span[0], 1), round(span[1], 1)],
                 row_length_mm=round(span[1] - span[0], 1), base_rate=pr['base_rate'],
                 peaks_row_span=peaks(pr['row_span'], pr['t_bins_mm']), peaks_full_width=peaks(pr['full_width'], pr['t_bins_mm']),
                 continuation_mm=extent(pr['along']), yardstick_0343=YARD, profile=pr)
        r['full_page'] = full_page(name, z, ct, poly)
        if ct is not None: r['closeup'] = closeup(name, z, ct, c, u, span)
        zr, _ = load(name, 'rev')
        if zr is not None: r['full_page_rev'] = full_page(name, zr, ct, poly, face='rev')
        res[name] = r
        print(name, {k: r[k] for k in ('row_length_mm', 'continuation_mm', 'peaks_row_span', 'peaks_full_width')}, flush=True)
    for n_ in ([] if ONLY else ['w044', 'w045', 'w046', 'w047', 'w048', 'w049', 'w051', 'w052', 'w053', 'w054', 'w055', 'w056', 'w068', 'w069', 'w070', 'w071', 'w072', 'w073']):
        z, ct = load(n_)
        if z is not None: full_page(n_, z, ct)
    if not ONLY: res['strip_w050'] = strip(['w048', 'w049', 'w050', 'w051', 'w052'], (2300, 7300), 'user row (w050) vs w048-w052, same x range, fwd = reading face', 'NEIGHBOURS_w048-w052')
    if not ONLY: res['strip_w074'] = strip(['w072', 'w073', 'w074'], (5100, 10400), 'w074 line vs w072-w073 (the lane fit has no w075), fwd = reading face', 'NEIGHBOURS_w072-w074')
    json.dump(res, open(f'{D}/analysis{"_" + "_".join(sorted(ONLY)) if ONLY else ""}.json', 'w'), indent=1)
