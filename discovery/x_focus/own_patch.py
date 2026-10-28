"""own_patch.py : are there more lines around a lead row?  Works on an own-sheet GrowPatch patch grown on the row.

Inputs: the patch render + fused z map (reading face), its tifxyz, and the row's points mapped into the patch render
(rowmap json: row_render_px [[x, y], ...], from the bilinear point-to-surface check).
1. Fits the row line (PCA) and resamples the z map and CT in row coordinates: s along the row, t across it, with +t pointing
   to higher scroll z (shown at the top).
2. Across-row profile: share of on-sheet pixels with z > 2 per 0.25 mm of t, over the row's own s-range and over the whole
   patch width; patch base rate; peaks at 2.5-20 mm from the row with >= 2x base = candidate parallel lines.
3. Along-row profile in 1 mm bins (|t| <= 1.2 mm) over the whole patch: does the row continue left/right?
4. The same profile on PHerc0343 known text (two lines 5.9 mm apart) at the same scale, as the yardstick.
Card: rotated ink-as-page and CT of the whole patch with guides at 0, +-5.5, +-11 mm, the profile, and the 0343 reference.
"""
import argparse, json, os, sys
import numpy as np, cv2, zarr, tifffile
cv2.setNumThreads(int(os.environ.get('OMP_NUM_THREADS', '4')))
W = '/home/liam/projects/vesuvius-work'
sys.path.insert(0, f'{W}/x_focus')
from make_card import ink_page, ct_img, hstack, vstack, text_block      # noqa: E402
from focus_region import load_tifxyz, xyz_at                             # noqa: E402
DISP = 30.0                 # um per card pixel (lead and 0343 alike)
STEP = 2                    # render px per analysis sample


def resample(z, v, ct, m, u, vv, um, S, T):
    """row-coordinate images: rows = t from +T (top) to -T, cols = s from -S to +S (mm); STEP render px per sample"""
    px = 1000 / um
    s = np.arange(-S * px, S * px, STEP, dtype=np.float32); t = np.arange(T * px, -T * px, -STEP, dtype=np.float32)
    Sg, Tg = np.meshgrid(s, t)
    X = (m[0] + Sg * u[0] + Tg * vv[0]).astype(np.float32); Y = (m[1] + Sg * u[1] + Tg * vv[1]).astype(np.float32)
    zr = cv2.remap(z, X, Y, cv2.INTER_LINEAR, borderValue=0)
    vr = cv2.remap(v.astype(np.uint8), X, Y, cv2.INTER_NEAREST, borderValue=0) > 0
    cr = cv2.remap(ct, X, Y, cv2.INTER_LINEAR, borderValue=0)
    return zr, vr, cr, s / px, t / px


def profiles(zr, vr, s_mm, t_mm, s_rng):
    hot = (zr > 2) & vr; base = float(hot.sum() / max(vr.sum(), 1))
    cols = (s_mm >= s_rng[0]) & (s_mm <= s_rng[1])
    def tprof(cm):
        h = hot[:, cm].sum(1).astype(np.float32); n = vr[:, cm].sum(1).astype(np.float32)
        k = max(3, int(0.25 / (t_mm[0] - t_mm[1])) | 1)
        h = cv2.GaussianBlur(h.reshape(-1, 1), (1, k), 0).ravel(); n = cv2.GaussianBlur(n.reshape(-1, 1), (1, k), 0).ravel()
        return np.where(n > 0.3 * cm.sum(), h / np.maximum(n, 1), np.nan), n / cm.sum()
    occ_row, cov_row = tprof(cols); occ_all, cov_all = tprof(np.ones_like(cols))
    band = np.abs(t_mm) <= 1.2
    bins = np.arange(np.floor(s_mm[0]), np.ceil(s_mm[-1]) + 1, 1.0); along = []
    for b0 in bins[:-1]:
        cm = (s_mm >= b0) & (s_mm < b0 + 1); n = vr[band][:, cm].sum(); h = hot[band][:, cm].sum()
        along.append(dict(s_mm=float(b0 + 0.5), occ=round(float(h / n), 4) if n > 0.3 * band.sum() * cm.sum() else None,
                          cover=round(float(n / max(band.sum() * cm.sum(), 1)), 2)))
    return base, occ_row, cov_row, occ_all, cov_all, along


def peaks(occ, t_mm, base, lo=2.5, hi=20.0):
    out = []
    o = np.nan_to_num(occ, nan=0)
    for i in range(1, len(o) - 1):
        if lo <= abs(t_mm[i]) <= hi and o[i] >= o[i - 1] and o[i] >= o[i + 1] and o[i] >= 2 * base:
            out.append(dict(t_mm=round(float(t_mm[i]), 2), occ=round(float(o[i]), 4), x_base=round(float(o[i] / max(base, 1e-6)), 2)))
    out.sort(key=lambda p: -p['occ']); keep = []
    for p in out:                                                       # one peak per 1.5 mm
        if all(abs(p['t_mm'] - q['t_mm']) > 1.5 for q in keep): keep.append(p)
    return keep[:8]


def plot_profile(occ_row, occ_all, t_mm, base, row_occ, h_px, w=260, xmax=None):
    """vertical plot aligned with the page panel: y = t (top = +T), x = occupancy"""
    img = np.full((h_px, w, 3), 245, np.uint8)
    xmax = xmax or max(0.05, np.nanmax(np.nan_to_num(occ_row)) * 1.1)
    ys = np.linspace(0, h_px - 1, len(t_mm)).astype(int)
    def X(o): return int(10 + (w - 20) * min(o, xmax) / xmax)
    for o, col in ((base, (150, 150, 150)), (2 * base, (0, 140, 255))):
        cv2.line(img, (X(o), 0), (X(o), h_px - 1), col, 1)
    for occ, col, th in ((occ_all, (200, 160, 0), 1), (occ_row, (0, 0, 200), 2)):
        p = [(X(o), y) for o, y in zip(occ, ys) if np.isfinite(o)]
        for a_, b_ in zip(p[:-1], p[1:]):
            if abs(a_[1] - b_[1]) <= 3: cv2.line(img, a_, b_, col, th)
    cv2.putText(img, 'z>2 share per t', (8, 20), 0, 0.5, (0, 0, 0), 1)
    cv2.putText(img, 'red: row span', (8, 40), 0, 0.45, (0, 0, 200), 1); cv2.putText(img, 'blue: full width', (8, 58), 0, 0.45, (200, 160, 0), 1)
    cv2.putText(img, 'orange: 2x base', (8, 76), 0, 0.45, (0, 140, 255), 1)
    cv2.putText(img, f'0..{xmax:.2f}', (8, h_px - 10), 0, 0.45, (0, 0, 0), 1)
    return img


def page_panel(zr, vr, cr, s_mm, t_mm, s_rng, guides, title, ct=False):
    img = ct_img(cr, vr) if ct else ink_page(zr, vr)
    sc = (abs(s_mm[1] - s_mm[0]) * 1000) / DISP
    img = cv2.resize(img, (int(img.shape[1] * sc), int(img.shape[0] * sc)), interpolation=cv2.INTER_AREA)
    H, Wd = img.shape[:2]; T = t_mm[0]; S0 = s_mm[0]
    ty = lambda t: int((T - t) * 1000 / DISP); sx = lambda s: int((s - S0) * 1000 / DISP)
    for g, col in guides:
        y = ty(g)
        for x in range(0, Wd, 24): cv2.line(img, (x, y), (min(x + 12, Wd - 1), y), col, 1)
        cv2.putText(img, f'{g:+.1f} mm', (Wd - 90, y - 4), 0, 0.45, col, 1)
    if s_rng is not None:
        cv2.rectangle(img, (sx(s_rng[0]), ty(1.2)), (sx(s_rng[1]), ty(-1.2)), (255, 200, 0), 2)
    L = int(10000 / DISP); cv2.rectangle(img, (12, H - 22), (12 + L, H - 12), (0, 200, 0), -1); cv2.putText(img, '1 cm', (12, H - 28), 0, 0.6, (0, 200, 0), 2)
    cv2.putText(img, title, (10, 26), 0, 0.65, (0, 0, 0), 4); cv2.putText(img, title, (10, 26), 0, 0.65, (0, 255, 255), 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--key', required=True); ap.add_argument('--render', required=True); ap.add_argument('--zmap', required=True)
    ap.add_argument('--tifxyz', required=True); ap.add_argument('--rowmap', required=True); ap.add_argument('--um', type=float, default=8.64)
    ap.add_argument('--S', type=float, default=30.0); ap.add_argument('--T', type=float, default=25.0); ap.add_argument('--title', default='')
    a = ap.parse_args()
    out = f'{W}/x_focus/out/{a.key}'; os.makedirs(out, exist_ok=True)
    z = np.nan_to_num(np.load(a.zmap).astype(np.float32)); R = zarr.open(a.render, mode='r')['0']; c = R.shape[0] // 2
    ct = np.asarray(R[c]).astype(np.float32); v = ct > 0
    rm = json.load(open(a.rowmap)); pts = np.asarray(rm['row_render_px'], np.float64)
    m = pts.mean(0); _, _, Vt = np.linalg.svd(pts - m, full_matrices=False); u = Vt[0] / np.linalg.norm(Vt[0])
    if u[0] < 0: u = -u
    vv = np.array([-u[1], u[0]]); Q = load_tifxyz(a.tifxyz); px = 1000 / a.um
    zz = lambda p: float(xyz_at(Q, np.array([p[0]]), np.array([p[1]]))[0][2])
    if zz(m + vv * px) < zz(m - vv * px): vv = -vv                       # +t = towards higher scroll z
    proj = (pts - m) @ u / px; s_rng = (float(np.percentile(proj, 1)), float(np.percentile(proj, 99)))
    zr, vr, cr, s_mm, t_mm = resample(z, v, ct, m, u, vv, a.um, a.S, a.T)
    base, occ_row, cov_row, occ_all, cov_all, along = profiles(zr, vr, s_mm, t_mm, s_rng)
    band = np.abs(t_mm) <= 1.2; cols = (s_mm >= s_rng[0]) & (s_mm <= s_rng[1])
    row_occ = float(((zr > 2) & vr)[band][:, cols].sum() / max(vr[band][:, cols].sum(), 1))
    pk_row = peaks(occ_row, t_mm, base); pk_all = peaks(occ_all, t_mm, base)
    cover_t = [float(t_mm[np.nonzero(cov_row > 0.5)[0][[0, -1]]][i]) if (cov_row > 0.5).any() else None for i in (0, 1)]
    # 0343 yardstick: same profile across its letter rows (rows horizontal in its render)
    def zf(p):
        q = np.nan_to_num(tifffile.imread(p).astype(np.float32)); vq = q != 0
        return (q - q[vq].mean()) / (q[vq].std() + 1e-6) * vq
    rz = (zf(f'{W}/x_px343/maps/hecate_rev_d+0.tif') + zf(f'{W}/x_px343/maps/rv2_rev_d+0.tif')) / 2
    R3 = zarr.open(f'{W}/fl/0343/concat_66.zarr', mode='r')['0']; ct3 = np.asarray(R3[33]).astype(np.float32)
    # x_zoom's 0343 box (450-2350, 2400-5300) holds two letter rows at y 5.51 / 11.45 mm (5.94 mm apart): centre on the first
    m3 = np.array([3850.0, 450 + 5.51 * 1000 / 8.64])
    z3, v3, c3, s3, t3 = resample(rz, ct3 > 0, ct3, m3, np.array([1.0, 0.0]), np.array([0.0, -1.0]), 8.64, 13.0, 10.0)
    r3 = (-12.5, 12.5)
    b3, o3_row, _, o3_all, _, al3 = profiles(z3, v3, s3, t3, r3)
    pk3 = peaks(o3_row, t3, b3)
    res = dict(key=a.key, row_fit=dict(centre_px=m.tolist(), u=u.tolist(), up=vv.tolist(), s_range_mm=s_rng, length_mm=round(s_rng[1] - s_rng[0], 2)),
               base_rate=round(base, 4), row_occ=round(row_occ, 4), row_x_base=round(row_occ / max(base, 1e-6), 2),
               coverage_t_mm=cover_t, peaks_row_span=pk_row, peaks_full_width=pk_all, along=along,
               ref_0343=dict(base_rate=round(b3, 4), peaks_row_span=pk3, row_centre=m3.tolist()),
               profile=dict(t_mm=t_mm[::4].round(3).tolist(), occ_row=np.round(occ_row[::4], 4).tolist(), occ_all=np.round(occ_all[::4], 4).tolist()))
    json.dump(res, open(f'{out}/own_patch.json', 'w'), indent=1, default=lambda o: None if o != o else o)
    # card
    guides = [(0.0, (255, 200, 0)), (5.5, (200, 0, 200)), (-5.5, (200, 0, 200)), (11.0, (200, 0, 200)), (-11.0, (200, 0, 200))]
    pg = page_panel(zr, vr, cr, s_mm, t_mm, s_rng, guides, 'LEAD own-sheet patch: ink as a page (dark = ink), row boxed; guides +-5.5 / +-11 mm')
    cp = page_panel(zr, vr, cr, s_mm, t_mm, None, guides, 'LEAD own-sheet patch: bare CT d0, same view', ct=True)
    xmax = max(0.05, np.nanmax(np.nan_to_num(occ_row)) * 1.1, np.nanmax(np.nan_to_num(o3_row)) * 1.1)
    pp = plot_profile(occ_row, occ_all, t_mm, base, row_occ, pg.shape[0], xmax=xmax)
    p3 = page_panel(z3, v3, c3, s3, t3, r3, [(0.0, (255, 200, 0)), (5.5, (200, 0, 200)), (-5.5, (200, 0, 200))], 'PHerc0343 KNOWN TEXT (2 rows, 5.94 mm apart), same scale, same profile')
    pp3 = plot_profile(o3_row, o3_all, t3, b3, 0, p3.shape[0], xmax=xmax)
    fmtp = lambda pk: ', '.join(f"{p['t_mm']:+.1f} mm ({p['x_base']}x)" for p in pk) or 'none'
    al_occ = [q for q in along if q['occ'] is not None]
    outside = [q['occ'] for q in al_occ if not (s_rng[0] - 1 <= q['s_mm'] <= s_rng[1] + 1)]
    lines = [f"{a.title}",
             f"row: {res['row_fit']['length_mm']} mm long, z>2 share in the row band {row_occ:.3f} = {res['row_x_base']}x the patch base rate {base:.4f}; patch covers t {cover_t} mm around the row (on-sheet > 50 %)",
             f"parallel-line candidates (peaks 2.5-20 mm off the row, >= 2x base) over the row's span: {fmtp(pk_row)};  over the full patch width: {fmtp(pk_all)}",
             f"PHerc0343 letters, same profile: base {b3:.4f}; peaks {fmtp(pk3)}  (its second row is 5.94 mm below the first)",
             f"along the row line outside the row (1 mm bins, |t| <= 1.2 mm): median share {np.median(outside) if outside else float('nan'):.3f}, "
             f"bins >= 2x base {sum(o >= 2 * base for o in outside)}/{len(outside)}"]
    body = vstack([hstack([pg, pp]), hstack([cp]), hstack([p3, pp3])])
    head = np.full((44, body.shape[1], 3), 18, np.uint8); cv2.putText(head, f'{a.key}: more lines around the row? (x_focus/own_patch.py)', (10, 30), 0, 0.9, (255, 255, 255), 2)
    card = vstack([head, body, text_block(lines, body.shape[1])])
    f = f'{out}/{a.key}_OWN_PATCH_CARD.jpg'; cv2.imwrite(f, card, [cv2.IMWRITE_JPEG_QUALITY, 90]); print(f, card.shape)
    print(json.dumps({k: res[k] for k in ('row_fit', 'base_rate', 'row_occ', 'row_x_base', 'coverage_t_mm', 'peaks_row_span', 'peaks_full_width')}, default=str))
    print('0343', json.dumps(res['ref_0343']))
    print('along', [(q['s_mm'], q['occ']) for q in along if q['occ'] is not None])


if __name__ == '__main__':
    main()
