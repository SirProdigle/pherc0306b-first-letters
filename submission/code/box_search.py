"""box_search.py [MAPS] [OUT] : the box search of version 4, exactly as methodology/BOXES_PREREG.md (20 Oct 2026).

Scores every 3.95 cm2 rectangle (L 25..45 mm x 395 / L, 0.25 mm grid, inside the context) by the number of confirmed letter-sized
marks (half tone in the V3 + TRACE main view, >= 20 % of their px also half tone in the Rv2 + v8Q check view) whose whole bbox lies
inside, ties by fewer stray marks.  MODEL IMAGES ONLY: no tracing, no model trained on PHerc0306B.
Writes box_search/{boxes.json, marks.json, top*_main.jpg, top*_check.jpg, v2box_*.jpg, overview_boxes.jpg}."""
import os, sys, json
import numpy as np, cv2

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))                  # the package
M = sys.argv[1] if len(sys.argv) > 1 else '/home/liam/projects/vesuvius-work/x_req/20261020/A_V3/maps'   # fwd_v3 / fwd_tr / fwd_rv2 / fwd_v8q.npy + ct20.png
OUT = sys.argv[2] if len(sys.argv) > 2 else 'box_search'; MESH = f'{PKG}/mesh/PHerc0306B_20250521133212_w074own_touchup_V5Rc'
sys.path.insert(0, f'{PKG}/code')
K = 1000 / 8.64; U0, V0 = -36.0, -21.0; CTX = dict(u0=-30.0, u1=30.0, v0=-20.0, v1=15.0); V2BOX = dict(u0=-12.1, u1=23.5, v0=-8.55, v1=2.5)
PAGE = np.array([243, 236, 222], np.float32); INK = np.array([22, 20, 18], np.float32); OFFQ = np.array([150, 160, 175], np.float32)
STEP = 0.25; AREA = 395.0; LENS = list(range(25, 46))


def px(u, v): return int(round((u - U0) * K)), int(round((v - V0) * K))


def zs(a, m):
    x = a[m].astype(np.float64); return ((a - x.mean()) / (x.std() + 1e-9)).astype(np.float32)


def view(maps, sl, valid):
    f = np.mean([zs(np.nan_to_num(np.load(f'{M}/fwd_{k}.npy', mmap_mode='r')[sl].astype(np.float32)), valid) * valid for k in maps], 0)
    lo, hi = float(np.percentile(f[valid], 50)), float(np.percentile(f[valid], 99.9))
    return np.where(valid, np.clip((f - lo) / (hi - lo), 0, 1), 0).astype(np.float32), (lo, hi)


def rgb(d, valid):
    o = PAGE * (1 - d[..., None]) + INK * d[..., None]; o[~valid] = OFFQ; return o.astype(np.uint8)[..., ::-1]


def main():
    os.makedirs(OUT, exist_ok=True)
    x0, y0 = px(CTX['u0'], CTX['v0']); x1, y1 = px(CTX['u1'], CTX['v1']); sl = (slice(y0, y1), slice(x0, x1))
    ct = cv2.imread(f'{M}/ct20.png', -1)[sl]
    v3 = np.load(f'{M}/fwd_v3.npy', mmap_mode='r')[sl].astype(np.float32)
    tr = np.load(f'{M}/fwd_tr.npy', mmap_mode='r')[sl].astype(np.float32)
    valid = (ct > 0) & np.isfinite(v3) & np.isfinite(tr) & (v3 != 0)
    d, lv = view(('v3', 'tr'), sl, valid)
    assert abs(lv[0] + 0.300) < 0.01 and abs(lv[1] - 6.872) < 0.01, lv                       # = the v3 main image's levels
    vc = valid & np.isfinite(np.load(f'{M}/fwd_rv2.npy', mmap_mode='r')[sl].astype(np.float32)) & np.isfinite(np.load(f'{M}/fwd_v8q.npy', mmap_mode='r')[sl].astype(np.float32))
    dc, lvc = view(('rv2', 'v8q'), sl, vc)
    ink = ((d >= 0.5) & valid).astype(np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) & valid.astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(ink, connectivity=8)
    conf_px = np.bincount(lab.ravel(), weights=((dc >= 0.5) & vc).ravel().astype(np.float64), minlength=n)
    marks = []
    for i in range(1, n):
        bx, by, bw, bh, ar = (int(v) for v in st[i]); wmm, hmm, amm = bw / K, bh / K, ar / K ** 2
        letter = 0.3 <= wmm <= 4.5 and 0.3 <= hmm <= 4.5 and max(wmm, hmm) >= 1.0 and amm >= 0.15
        cls = 'letter' if letter else ('blob' if max(wmm, hmm) > 4.5 else 'speck')
        marks.append(dict(i=i, cls=cls, confirmed=bool(letter and conf_px[i] / ar >= 0.2), conf_share=round(float(conf_px[i] / ar), 3),
                          u0=CTX['u0'] + bx / K, u1=CTX['u0'] + (bx + bw) / K, v0=CTX['v0'] + by / K, v1=CTX['v0'] + (by + bh) / K,
                          cu=CTX['u0'] + cen[i][0] / K, cv=CTX['v0'] + cen[i][1] / K, w_mm=round(wmm, 2), h_mm=round(hmm, 2), area_mm2=round(amm, 3)))
    print('marks', len(marks), {c: sum(m['cls'] == c for m in marks) for c in ('letter', 'blob', 'speck')}, 'confirmed', sum(m['confirmed'] for m in marks), flush=True)

    def count_box(b, sel, by_bbox=True):
        return sum(1 for m in marks if sel(m) and ((b['u0'] <= m['u0'] and m['u1'] <= b['u1'] and b['v0'] <= m['v0'] and m['v1'] <= b['v1']) if by_bbox
                                                  else (b['u0'] <= m['cu'] <= b['u1'] and b['v0'] <= m['cv'] <= b['v1'])))
    conf = lambda m: m['confirmed']; unconf = lambda m: m['cls'] == 'letter' and not m['confirmed']; stray = lambda m: m['cls'] != 'letter'  # noqa: E731
    score = lambda b: dict(S=count_box(b, conf), stray=count_box(b, stray, False), unconf=count_box(b, unconf))   # noqa: E731
    v2s = score(V2BOX); print('v2 box', v2s, flush=True)
    cands = []
    for Lm in LENS:
        H = AREA / Lm; na = int(np.floor((CTX['u1'] - CTX['u0'] - Lm) / STEP + 1e-9)) + 1; nb = int(np.floor((CTX['v1'] - CTX['v0'] - H) / STEP + 1e-9)) + 1
        acc = {k: np.zeros((nb + 1, na + 1), np.int32) for k in ('S', 'stray', 'unconf')}
        for m in marks:
            if m['cls'] == 'letter':
                lo_a, hi_a, lo_b, hi_b = m['u1'] - Lm, m['u0'], m['v1'] - H, m['v0']; key = 'S' if m['confirmed'] else 'unconf'
            else:
                lo_a, hi_a, lo_b, hi_b = m['cu'] - Lm, m['cu'], m['cv'] - H, m['cv']; key = 'stray'
            ia0 = max(0, int(np.ceil((lo_a - CTX['u0']) / STEP - 1e-9))); ia1 = min(na - 1, int(np.floor((hi_a - CTX['u0']) / STEP + 1e-9)))
            ib0 = max(0, int(np.ceil((lo_b - CTX['v0']) / STEP - 1e-9))); ib1 = min(nb - 1, int(np.floor((hi_b - CTX['v0']) / STEP + 1e-9)))
            if ia0 > ia1 or ib0 > ib1: continue
            a = acc[key]; a[ib0, ia0] += 1; a[ib0, ia1 + 1] -= 1; a[ib1 + 1, ia0] -= 1; a[ib1 + 1, ia1 + 1] += 1
        g = {k: np.cumsum(np.cumsum(v, 0), 1)[:nb, :na] for k, v in acc.items()}
        key = g['S'].astype(np.int64) * 10 ** 6 - g['stray'] * 10 ** 3 + g['unconf']
        for idx in np.argsort(key.ravel())[::-1][:400]:
            ib, ia = divmod(int(idx), na); u0 = CTX['u0'] + ia * STEP; v0 = CTX['v0'] + ib * STEP
            cands.append(dict(u0=u0, u1=u0 + Lm, v0=v0, v1=v0 + H, L=Lm, H=round(H, 3), S=int(g['S'][ib, ia]), stray=int(g['stray'][ib, ia]), unconf=int(g['unconf'][ib, ia])))
    cands.sort(key=lambda c: (-c['S'], c['stray'], -c['unconf']))

    def iou(a, b):
        iw = max(0, min(a['u1'], b['u1']) - max(a['u0'], b['u0'])); ih = max(0, min(a['v1'], b['v1']) - max(a['v0'], b['v0']))
        i = iw * ih; return i / ((a['u1'] - a['u0']) * (a['v1'] - a['v0']) + (b['u1'] - b['u0']) * (b['v1'] - b['v0']) - i)
    top = []
    for c in cands:
        if all(iou(c, t) < 0.3 for t in top): top.append(c)
        if len(top) == 3: break
    import make_claim_image_v2 as C
    v2rec = dict(V2BOX, **v2s)
    for t in top + [v2rec]:
        t['mesh_cm2'] = round(C.mesh_box_area(MESH, t), 4); t['flat_cm2'] = round((t['u1'] - t['u0']) * (t['v1'] - t['v0']) / 100, 4)
        inside = lambda m, b=t: b['u0'] <= m['u0'] and m['u1'] <= b['u1'] and b['v0'] <= m['v0'] and m['v1'] <= b['v1']   # noqa: E731
        t['confirmed_marks'] = [m['i'] for m in marks if m['confirmed'] and inside(m)]
    v2conf = [m['i'] for m in marks if m['confirmed'] and V2BOX['u0'] <= m['u0'] and m['u1'] <= V2BOX['u1'] and V2BOX['v0'] <= m['v0'] and m['v1'] <= V2BOX['v1']]
    for t in top:
        keeps = set(v2conf) <= set(t['confirmed_marks'])
        t['offer'] = bool((t['S'] >= v2s['S'] + 2 and keeps) or (keeps and len(set(t['confirmed_marks']) - set(v2conf)) >= 1))
        t['keeps_all_v2_confirmed'] = keeps; t['new_confirmed_vs_v2'] = sorted(set(t['confirmed_marks']) - set(v2conf))
    res = dict(prereg='BOXES_PREREG.md', levels_main=lv, levels_check=lvc, n_marks=len(marks), v2_box=v2rec,
               top3=top, best_per_length={str(Lm): next(c for c in cands if c['L'] == Lm) for Lm in LENS})
    json.dump(res, open(f'{OUT}/boxes.json', 'w'), indent=1); json.dump(marks, open(f'{OUT}/marks.json', 'w'))
    # review images: main and check views, reading orientation (flip rows), box + 1.5 mm, thin box outlines (candidate blue, v2 orange)
    R, Rc = rgb(d, valid), rgb(dc, vc)
    def crop(img, b, pad=1.5):
        ua, ub, va, vb = min(b['u0'], V2BOX['u0']) - pad, max(b['u1'], V2BOX['u1']) + pad, min(b['v0'], V2BOX['v0']) - pad, max(b['v1'], V2BOX['v1']) + pad
        X0, Y0 = px(ua, va); X1, Y1 = px(ub, vb); X0 -= x0; X1 -= x0; Y0 -= y0; Y1 -= y0
        X0, Y0 = max(X0, 0), max(Y0, 0); c = img[Y0:Y1, X0:X1].copy()
        for bb, col in ((V2BOX, (0, 140, 255)), (b, (255, 90, 0))):
            p0 = px(bb['u0'], bb['v0']); p1 = px(bb['u1'], bb['v1'])
            cv2.rectangle(c, (p0[0] - x0 - X0, p0[1] - y0 - Y0), (p1[0] - x0 - X0, p1[1] - y0 - Y0), col, 5)
        return np.flipud(c)
    for k, t in enumerate(top, 1):
        for nm, img in (('main', R), ('check', Rc)):
            c = crop(img, t); s = 1600 / c.shape[1]
            cv2.imwrite(f'{OUT}/top{k}_{nm}.jpg', cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 90])
    ov = R.copy()
    for t, col in zip(top + [V2BOX], ((255, 90, 0), (200, 0, 200), (0, 160, 0), (0, 140, 255))):
        p0 = px(t['u0'], t['v0']); p1 = px(t['u1'], t['v1']); cv2.rectangle(ov, (p0[0] - x0, p0[1] - y0), (p1[0] - x0, p1[1] - y0), col, 12)
    for m in marks:
        if m['confirmed']:
            p0 = px(m['u0'], m['v0']); p1 = px(m['u1'], m['v1']); cv2.rectangle(ov, (p0[0] - x0, p0[1] - y0), (p1[0] - x0, p1[1] - y0), (0, 0, 220), 3)
    ov = np.flipud(ov); cv2.imwrite(f'{OUT}/overview_boxes.jpg', cv2.resize(ov, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(json.dumps(dict(v2={k: v for k, v in v2rec.items() if k != 'confirmed_marks'}, top3=[{k: t[k] for k in ('u0', 'u1', 'v0', 'v1', 'S', 'stray', 'unconf', 'mesh_cm2', 'offer', 'new_confirmed_vs_v2')} for t in top]), indent=1))


if __name__ == '__main__':
    main()
