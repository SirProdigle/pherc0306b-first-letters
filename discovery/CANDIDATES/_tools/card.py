"""card.py: one review card per candidate.
Layout (2 x 2): [ink overlay, full view | bare CT, full view]
                [ink overlay, close-up  | bare CT, close-up ]   (close-up = the ringed area, enlarged)
The same ring (ellipse) is drawn on all four panels; a 1 cm scale bar is on the full views.
make_card(ct, ink_z, roi, um, title, notes, out, angle=None, valid=None)
  ct    : 2-D CT layer (any dtype, 0 = off-sheet)       ink_z : 2-D ink z-score map on the same grid
  roi   : (cy, cx, ry, rx) ellipse centre and radii in px of the (rotated) image
  angle : optional rotation (degrees, zoom.py convention) applied to both before drawing"""
import numpy as np, cv2

def _rot(img, ang):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang - 90, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1]); nw, nh = int(h * s + w * c), int(h * c + w * s)
    M[0, 2] += nw / 2 - w / 2; M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR)

def make_card(ct, ink_z, roi, um, title, notes, out, angle=None, valid=None, pad_frac=0.6, panel=700):
    ct = ct.astype(np.float32); z = np.nan_to_num(ink_z.astype(np.float32))
    if z.shape != ct.shape:
        z = cv2.resize(z, (ct.shape[1], ct.shape[0]))
    v = (ct > 0) if valid is None else valid
    lo, hi = np.percentile(ct[v], 1), np.percentile(ct[v], 99.5)
    g = np.clip((ct - lo) / (hi - lo + 1e-6), 0, 1); g[~v] = 0
    if angle is not None:
        g = _rot(g, angle); z = _rot(z, angle); v = _rot(v.astype(np.uint8), angle) > 0
    al = np.clip((z - 1.0) / 2.0, 0, 1); al[~v] = 0
    gg = g * 0.6
    ov = np.stack([gg * (1 - al) + al * 0.15, gg * (1 - al) + al * 0.85, gg * (1 - al) + al * 1.0], -1)
    ov = (ov * 255).astype(np.uint8)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    bare = cv2.cvtColor(cl.apply((g * 255).astype(np.uint8)), cv2.COLOR_GRAY2BGR)
    cy, cx, ry, rx = [int(t) for t in roi]
    H, W = g.shape
    # full views: crop to the on-sheet bounding box (with margin) so the panels are not mostly black
    ys, xs = np.nonzero(v); m = 40
    fy0, fy1, fx0, fx1 = max(0, ys.min() - m), min(H, ys.max() + m), max(0, xs.min() - m), min(W, xs.max() + m)
    # close-up: the ellipse plus a margin
    py, px = int(ry * (1 + pad_frac)), int(rx * (1 + pad_frac))
    zy0, zy1, zx0, zx1 = max(0, cy - py), min(H, cy + py), max(0, cx - px), min(W, cx + px)
    def ring(img, y0, x0, scale, thick):
        c = (int((cx - x0) * scale), int((cy - y0) * scale)); ax = (int(rx * scale), int(ry * scale))
        cv2.ellipse(img, c, ax, 0, 0, 360, (0, 0, 0), thick + 4); cv2.ellipse(img, c, ax, 0, 0, 360, (60, 60, 255), thick)
        return img
    def fit(img, y0, y1, x0, x1):
        crop = img[y0:y1, x0:x1]; s = panel / max(crop.shape[:2])
        return cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC), s
    pz = np.clip((z - 0.8) / 2.4, 0, 1); pz[~v] = 0
    page = np.stack([0.78 * (1 - pz) + 0.10 * pz, 0.88 * (1 - pz) + 0.10 * pz, 0.92 * (1 - pz) + 0.12 * pz], -1)
    page[~v] = 0.12; page = (page * 255).astype(np.uint8)
    def padto(t, h, w):
        o = np.full((h, w, 3), 25, np.uint8); o[:t.shape[0], :t.shape[1]] = t; return o
    def label(t, lab, bar_um, bar_txt, scale, col=(0, 255, 255)):
        L = int(bar_um / um * scale); cv2.rectangle(t, (15, t.shape[0] - 25), (15 + L, t.shape[0] - 15), (255, 255, 255), -1)
        cv2.rectangle(t, (15, t.shape[0] - 25), (15 + L, t.shape[0] - 15), (0, 0, 0), 1)
        cv2.putText(t, bar_txt, (15, t.shape[0] - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(t, lab, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4); cv2.putText(t, lab, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2)
        return t
    # full views (2 panels)
    wide = (fx1 - fx0) > 1.6 * (fy1 - fy0)
    fw = 2 * panel + 8 if wide else panel
    full = []
    for img, lab in ((ov, 'ink models (yellow = ink signal)'), (page, 'ink only, as a page (dark = ink)'), (bare, 'bare CT, same view')):
        crop = img[fy0:fy1, fx0:fx1]; s_ = fw / crop.shape[1] if wide else panel / max(crop.shape[:2])
        t = cv2.resize(crop, None, fx=s_, fy=s_, interpolation=cv2.INTER_AREA if s_ < 1 else cv2.INTER_CUBIC); ring(t, fy0, fx0, s_, 3)
        full.append(label(t, lab, 1e4, '1 cm', s_))
    if wide:
        top = np.vstack([full[0], np.full((8, fw, 3), 90, np.uint8), full[1], np.full((8, fw, 3), 90, np.uint8), full[2]])
    else:
        h = max(t.shape[0] for t in full); full = [padto(t, h, panel) for t in full]
        top = np.hstack([full[0], np.full((h, 8, 3), 90, np.uint8), full[1], np.full((h, 8, 3), 90, np.uint8), full[2]])
    # close-ups (3 panels): ink as a page | ink overlay | bare CT
    cp = int((top.shape[1] - 16) / 3)
    close = []
    for img, lab in ((page, 'close-up: ink only, as a page (dark = ink)'), (ov, 'close-up: ink overlay'), (bare, 'close-up: bare CT')):
        crop = img[zy0:zy1, zx0:zx1]; s_ = cp / max(crop.shape[:2])
        t = cv2.resize(crop, None, fx=s_, fy=s_, interpolation=cv2.INTER_AREA if s_ < 1 else cv2.INTER_CUBIC); ring(t, zy0, zx0, s_, 2)
        close.append(label(t, lab, 1e3, '1 mm', s_))
    h = max(t.shape[0] for t in close); close = [padto(t, h, cp) for t in close]
    bot = np.hstack([close[0], np.full((h, 8, 3), 90, np.uint8), close[1], np.full((h, 8, 3), 90, np.uint8), close[2]])
    Wd = max(top.shape[1], bot.shape[1])
    grid = np.vstack([padto(top, top.shape[0], Wd), np.full((8, Wd, 3), 90, np.uint8), padto(bot, bot.shape[0], Wd)])
    head = np.full((70 + 26 * len(notes), grid.shape[1], 3), 15, np.uint8)
    cv2.putText(head, title, (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    for i, n in enumerate(notes):
        cv2.putText(head, n, (12, 72 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)
    cv2.imwrite(out, np.vstack([head, grid]), [cv2.IMWRITE_JPEG_QUALITY, 90])


def auto_roi(ct, ink_z, um, r_mm=4.0, valid=None, angle=None):
    """ellipse around the densest ink area (z smoothed over ~1 mm), radius r_mm; coords in the (rotated) frame"""
    z = np.nan_to_num(ink_z.astype(np.float32))
    if z.shape != ct.shape: z = cv2.resize(z, (ct.shape[1], ct.shape[0]))
    v = (ct > 0) if valid is None else valid
    if angle is not None:
        z = _rot(z, angle); v = _rot(v.astype(np.uint8), angle) > 0
    d = cv2.GaussianBlur(np.clip(z, 0, None) * v, (0, 0), 1000 / um)
    d = d * (cv2.erode(v.astype(np.uint8), np.ones((int(2000 / um),) * 2, np.uint8)) > 0)
    cy, cx = np.unravel_index(np.argmax(d), d.shape); r = int(r_mm * 1000 / um)
    return (cy, cx, r, r)
