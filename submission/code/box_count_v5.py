"""box_count_v5.py : the mark count of the claim box (evidence/box_count_<tag>.json), version 5.

Marks exactly as the version 4 box search (methodology/BOXES_PREREG.md, code/box_search.py): the main view = the lead pair, fused as
the main image (per-model z over the context's on-sheet px, mean) under the grey rule (median .. p99.9 of the context), taken on
the render raster's integer pixels of the context (as box_search.py); ink = d >= 0.5, closed with a 15 px ellipse, 8-connected
components; letter-sized = both sides 0.3..4.5 mm, longer side >= 1 mm, area >= 0.15 mm2; confirmed = >= 20 % of its px also at
half tone in the check view (a model pair sharing no model with the main view; version 4: Reader v2 + v8Q).
Each named box is scored: S = confirmed letter-sized marks whose whole bbox lies inside; stray = specks + blobs whose centre lies
inside; unconf = unconfirmed letter-sized marks inside; letter_cut = letter-sized marks whose bbox crosses an edge.
With the version 4c config the json is version 4c's evidence/box_count_v4c.json byte for byte.  A mark is not a letter."""
import os, sys, json
import numpy as np, cv2

K = 1000 / 8.64; U0, V0 = -36.0, -21.0; CTX = dict(u0=-30.0, u1=30.0, v0=-20.0, v1=15.0)
RULE = ('BOXES_PREREG.md marks (code/box_search.py, box_search/marks.json): confirmed letter-sized marks whose whole bbox lies inside; '
        'strays = specks + blobs whose centre lies inside; letter_cut = letter-sized marks whose bbox crosses an edge')


def px(u, v): return int(round((u - U0) * K)), int(round((v - V0) * K))


def zs(a, m):
    x = a[m].astype(np.float64); return ((a - x.mean()) / (x.std() + 1e-9)).astype(np.float32)


def load(path, sl, crop=None):
    """a map's context slice on the render raster (a cropped map is pasted at its crop box: (y0, y1, x0, x1) in fwd_<k>.json)"""
    a = np.load(path, mmap_mode='r'); j = path[:-4] + '.json'
    if a.shape != (4900, 7780):
        y0, y1, x0, x1 = json.load(open(j))['box']; full = np.full((4900, 7780), np.nan, np.float32); full[y0:y1, x0:x1] = a[:y1 - y0, :x1 - x0]; a = full
    return np.asarray(a[sl]).astype(np.float32)


def view(maps_dir, keys, sl, valid):                                                # box_search.view
    f = np.mean([zs(np.nan_to_num(load(f'{maps_dir}/fwd_{k}.npy', sl)), valid) * valid for k in keys], 0)
    lo, hi = float(np.percentile(f[valid], 50)), float(np.percentile(f[valid], 99.9))
    return np.where(valid, np.clip((f - lo) / (hi - lo), 0, 1), 0).astype(np.float32), (lo, hi)


def marks_of(maps_dir, main, check):
    x0, y0 = px(CTX['u0'], CTX['v0']); x1, y1 = px(CTX['u1'], CTX['v1']); sl = (slice(y0, y1), slice(x0, x1))
    ct = cv2.imread(f'{maps_dir}/ct20.png', -1)[sl]
    mm = {k: load(f'{maps_dir}/fwd_{k}.npy', sl) for k in main}
    valid = (ct > 0) & np.logical_and.reduce([np.isfinite(a) for a in mm.values()]) & (mm[main[0]] != 0)   # box_search: v3 finite, tr finite, v3 != 0
    d, lv = view(maps_dir, main, sl, valid)
    vc = valid & np.logical_and.reduce([np.isfinite(load(f'{maps_dir}/fwd_{k}.npy', sl)) for k in check])
    dc, lvc = view(maps_dir, check, sl, vc)
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
    return marks, dict(main=lv, check=lvc)


def score(marks, b):
    ins = lambda m: b['u0'] <= m['u0'] and m['u1'] <= b['u1'] and b['v0'] <= m['v0'] and m['v1'] <= b['v1']        # noqa: E731
    cin = lambda m: b['u0'] <= m['cu'] <= b['u1'] and b['v0'] <= m['cv'] <= b['v1']                               # noqa: E731
    cut = lambda m: (m['u0'] < b['u1'] and m['u1'] > b['u0'] and m['v0'] < b['v1'] and m['v1'] > b['v0']) and not ins(m)  # noqa: E731
    return dict(box_mm=b, S=sum(1 for m in marks if m['confirmed'] and ins(m)), stray=sum(1 for m in marks if m['cls'] != 'letter' and cin(m)),
                unconf=sum(1 for m in marks if m['cls'] == 'letter' and not m['confirmed'] and ins(m)),
                letter_cut=[m['i'] for m in marks if m['cls'] == 'letter' and cut(m)],
                confirmed_marks=[m['i'] for m in marks if m['confirmed'] and ins(m)])


def box_count(maps_dir, main, check, boxes, lost_vs, out_json, marks_json=''):
    """boxes: ordered {name: box}; the first is the claim box; lost_vs: name of the box whose confirmed marks the first is compared to"""
    marks, lv = marks_of(maps_dir, list(main), list(check))
    res = dict(rule=RULE)
    for name, b in boxes.items(): res[name] = score(marks, {k: float(b[k]) for k in ('u0', 'u1', 'v0', 'v1')})
    first = next(iter(boxes))
    if lost_vs and lost_vs in res:
        lost = sorted(set(res[lost_vs]['confirmed_marks']) - set(res[first]['confirmed_marks']))
        res[f'lost_vs_{lost_vs}'] = [{k: m[k] for k in ('i', 'u0', 'u1', 'v0', 'v1', 'w_mm', 'h_mm', 'area_mm2')} for m in marks if m['i'] in lost]
    res['marks'] = [{k: m[k] for k in ('i', 'cls', 'confirmed', 'conf_share', 'u0', 'u1', 'v0', 'v1', 'w_mm', 'h_mm', 'area_mm2')}
                    for m in marks if m['cls'] == 'letter']
    json.dump(res, open(out_json, 'w'), indent=1)
    if marks_json: json.dump(dict(levels=lv, main=list(main), check=list(check), marks=marks), open(marks_json, 'w'))
    return res, lv
