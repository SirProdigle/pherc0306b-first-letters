"""compare.py : lead A depth sweep (c17..c23 = -3..+3 vx off the own-sheet surface, each a d-1..+1 max-quality map) as images
the reader can compare, plus per-depth numbers for the line and for the group of marks above its left half.

Same rendering as the reader pack (x_reader/make_pack_v2.py): per-model z-score, mean fused, dark = ink, 10 um/px, levelled on the
line, 16 x 40 mm window.  Writes into x_followup/A_depth/:
  depth_stack.jpg        the wide window at every depth, stacked (-3 at top)
  depth_stack_faint.jpg  the same at the faint threshold (z 0.3..2.0)
  depth_group.jpg        the line + group box (10 x 26 mm) at every depth, 2x
  depth_metrics.json     per depth: line band and group box ink (mean z, fraction z>2, blob count), and a blank-band reference
Boxes (levelled window coordinates, as rendered, +y down): line band = rows -2.5..+2.5 mm, x -13..+13 mm; group box = rows -7..-3
mm, x -13..0 mm (left half); blank reference = rows +4..+8 mm and -7..-3 mm on the right half (to compare).
"""
import os, json
import numpy as np, cv2
cv2.setNumThreads(4)
W = '/home/liam/projects/vesuvius-work'; T = f'{W}/x_followup/A_depth'
src = open(f'{W}/x_reader/make_pack.py').read(); ns = {}
exec(compile(src[:src.index('def level(')], 'make_pack', 'exec'), ns)
zs, page, rescale, bar = ns['zs'], ns['page'], ns['rescale'], ns['bar']
UM = 8.64; Y0, X0 = 1089, 991                                                      # depth crop origin in the render
P = np.array(json.load(open(f'{W}/x_focus306/loops/own_row.json'))['poly_render'], float)
P[:, 0] -= X0; P[:, 1] -= Y0
cx, cy = P.mean(0); ang = float(np.degrees(np.arctan2(P[1, 1] - P[0, 1], P[1, 0] - P[0, 0])))
DEPTHS = [17, 18, 19, 20, 21, 22, 23]


def levelled(c):
    d = f'{T}/c{c}'
    zz = [zs(np.load(f'{d}/fwd_{m}.npy').astype(np.float32)) for m in ('hec', 'rv2', 'tr')]
    ct = cv2.imread(f'{d}/ct{c}.png', 0).astype(np.float32)
    valid = np.logical_and.reduce([v for _, v in zz]) & (ct > 0)
    fused = np.mean([z for z, _ in zz], 0); fused[~valid] = 0
    r = int(30 * 1000 / UM)
    y0, x0 = max(0, int(cy) - r), max(0, int(cx) - r)
    crop = lambda a: a[y0:int(cy) + r, x0:int(cx) + r]
    zr = rescale(crop(fused), UM); vr = rescale(crop(valid).astype(np.uint8), UM) > 0
    c0 = np.array([cx - x0, cy - y0]) * UM / 10.0
    M = cv2.getRotationMatrix2D((float(c0[0]), float(c0[1])), ang, 1.0)
    zr = cv2.warpAffine(zr, M, (zr.shape[1], zr.shape[0]), flags=cv2.INTER_LINEAR, borderValue=0)
    vr = cv2.warpAffine(vr.astype(np.uint8), M, (vr.shape[1], vr.shape[0]), flags=cv2.INTER_NEAREST, borderValue=0) > 0
    return zr, vr, int(c0[1]), int(c0[0])


def win(img, cyy, cxx, h_mm, w_mm, fill, dy_mm=0.0):
    h, w = int(h_mm * 100), int(w_mm * 100)
    out = np.full((h, w) + img.shape[2:], fill, img.dtype)
    ya, xa = cyy + int(dy_mm * 100) - h // 2, cxx - w // 2
    sy0, sx0 = max(0, ya), max(0, xa); sy1, sx1 = min(img.shape[0], ya + h), min(img.shape[1], xa + w)
    out[sy0 - ya:sy1 - ya, sx0 - xa:sx1 - xa] = img[sy0:sy1, sx0:sx1]
    return out


def box_stats(z, v, cyy, cxx, r0, r1, x0, x1):
    b = z[cyy + int(r0 * 100):cyy + int(r1 * 100), cxx + int(x0 * 100):cxx + int(x1 * 100)]
    bv = v[cyy + int(r0 * 100):cyy + int(r1 * 100), cxx + int(x0 * 100):cxx + int(x1 * 100)]
    if bv.sum() < 100: return None
    m = (b > 2) & bv
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    blobs = int(((st[1:, cv2.CC_STAT_AREA] >= 25) & (st[1:, cv2.CC_STAT_AREA] <= 2500)).sum())   # 0.25..25 mm2 marks
    return dict(mean_z=round(float(b[bv].mean()), 3), frac_gt2=round(float(m.sum() / bv.sum()), 4), marks=blobs,
                valid=round(float(bv.mean()), 3))


def faint(z, v):
    a = np.clip((z - 0.3) / 1.7, 0, 1); a[~v] = 0; g = np.where(v, 1 - 0.92 * a, 0.12)
    return (np.stack([g * 0.80, g * 0.88, g * 0.93], -1) * 255).astype(np.uint8)


if __name__ == '__main__':
    rows, rows_f, rows_g, met = [], [], [], {}
    for c in DEPTHS:
        z, v, cyy, cxx = levelled(c)
        off = c - 20; lab = f'depth {off:+d} vx ({off * UM:+.0f} um) off the fitted surface'
        P_ = page(z, v)
        rows.append(bar(win(P_, cyy, cxx, 16, 40, 31), lab))
        rows_f.append(bar(win(faint(z, v), cyy, cxx, 16, 40, 31), lab + ', faint marks'))
        g = win(P_, cyy, cxx, 10, 26, 31, dy_mm=-2.0); g = cv2.resize(g, (g.shape[1] * 2, g.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
        Hh = g.shape[0]
        cv2.line(g, (20, Hh - 22), (1020, Hh - 22), (40, 40, 200), 6); cv2.putText(g, '5 mm', (20, Hh - 34), 0, 1.0, (40, 40, 200), 2)
        cv2.putText(g, lab, (20, 40), 0, 1.1, (40, 40, 200), 2)
        rows_g.append(g)
        met[off] = dict(line=box_stats(z, v, cyy, cxx, -2.5, 2.5, -13, 13), group_above_left=box_stats(z, v, cyy, cxx, -7, -3, -13, 0),
                        above_right=box_stats(z, v, cyy, cxx, -7, -3, 0, 13), below=box_stats(z, v, cyy, cxx, 4, 8, -13, 13))
        print(off, json.dumps(met[off]), flush=True)
    sep = lambda L: np.concatenate([np.pad(p, ((0, 12), (0, 0), (0, 0)), constant_values=40) for p in L], 0)
    cv2.imwrite(f'{T}/depth_stack.jpg', sep(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    cv2.imwrite(f'{T}/depth_stack_faint.jpg', sep(rows_f), [cv2.IMWRITE_JPEG_QUALITY, 88])
    cv2.imwrite(f'{T}/depth_group.jpg', sep(rows_g), [cv2.IMWRITE_JPEG_QUALITY, 88])
    json.dump(dict(angle_deg=ang, boxes_mm=dict(line='rows -2.5..2.5, x -13..13', group_above_left='rows -7..-3, x -13..0',
                   above_right='rows -7..-3, x 0..13', below='rows 4..8, x -13..13'), per_depth_vx=met),
              open(f'{T}/depth_metrics.json', 'w'), indent=1)
    print('wrote', T)
