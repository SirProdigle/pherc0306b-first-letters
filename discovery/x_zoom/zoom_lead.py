"""zoom_lead.py --name N --render R.zarr --c C --um UM --face forward|reverse --roi Y0 Y1 X0 X1 [--title T] [--verdict V]

Max-quality re-read of ONE lead region, compared side by side with the same recipe on known PHerc0343 text.

Recipe (validated on the 0343 stroke labels, x_zoom/README.md; each part kept only if it raised AP):
  Hecate stride 16 (lane: 32), d -1/0/+1 averaged; Reader v2 and TRACE overlap 0.5 with 8-fold dihedral TTA,
  d -1/0/+1 averaged; plain z-score fusion.  0343 crop AP .693 / d' 3.44 vs the lane's Hecate+Rv2 .669 / 3.22.
Outputs (x_zoom/<N>/): maps.npz, metrics.json, <N>_ZOOM_CARD.jpg with
  (a) per-model ink-as-page panels at one contrast, (b) agreement map (pixels with z>2 in >=2 of 3 models),
  (c) fused max-quality page, (d) depth profile d -4..+4 (Reader v2 + TRACE, row score and hot fraction;
  broad = text-like, narrow peak = crack, x_depth), (e) mark geometry vs 0343 (letters ~2.5 mm, pitch ~3.8 mm,
  lines 5.3-6.0 mm), (f) glyph score (x_glyph gA, on the lane-recipe map as calibrated, and on the max-quality map),
  (g) the same panels for 0343 letters (val343 cache) at the same mm scale, with a 1 cm bar.
No super-resolution or generative step anywhere."""
import os, sys, json, argparse, time
import numpy as np, cv2, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zoomlib as Z                                                        # noqa: E402
W = Z.W
sys.path.insert(0, f'{W}/fl/scripts'); sys.path.insert(0, f'{W}/x_glyph')
from spectral_rows import score as row_score                              # noqa: E402
DISP_UM = 25.0          # display scale (um per card pixel) for lead and reference alike
MARGIN = 192            # crop margin around the ROI (px), so the models see context; panels show the ROI only
REF = dict(dir=f'{W}/x_zoom/val343', um=8.64, render=f'{W}/fl/0343/concat_66.zarr', box=(450, 2350, 2400, 5300), c=33)


def cache(path, fn):
    if os.path.exists(path): return np.load(path)
    a = fn(); np.save(path, a.astype(np.float32)); return a


def mq_maps(vol, c, um, face, out, depths=(-1, 0, 1)):
    """per-model max-quality maps (crop grid) + lane-recipe maps for the glyph calibration"""
    m = {}
    m['hec'] = np.mean([cache(f'{out}/hec_s16_d{d}.npy', lambda d=d: Z.hecate_map(vol, um, c, d, face, stride=16)) for d in depths], 0)
    m['rv2'] = np.mean([cache(f'{out}/rv2_tta_d{d}.npy', lambda d=d: Z.villa_map('rv2', vol, c, d, face, tta8=True)) for d in depths], 0)
    m['tr'] = np.mean([cache(f'{out}/tr_tta_d{d}.npy', lambda d=d: Z.villa_map('trace', vol, c, d, face, tta8=True)) for d in depths], 0)
    m['lane_hec'] = cache(f'{out}/hec_s32_d0.npy', lambda: Z.hecate_map(vol, um, c, 0, face, stride=32))
    m['lane_rv2'] = cache(f'{out}/rv2_o50_d0.npy', lambda: Z.villa_map('rv2', vol, c, 0, face))
    return m


def ref_maps():
    d = REF['dir']; L = lambda n: np.load(f'{d}/{n}.npy')
    return {'hec': np.mean([L(f'hec_s16_d{k}') for k in (-1, 0, 1)], 0), 'rv2': np.mean([L(f'rv2_o50_tta_d{k}') for k in (-1, 0, 1)], 0),
            'tr': np.mean([L(f'tr_o50_tta_d{k}') for k in (-1, 0, 1)], 0), 'lane_hec': L('hec_s32_d0'), 'lane_rv2': L('rv2_o50_d0')}


def ref_blank(rm, valid, rr):
    """0343 reference normalised the way a lead usually is: per-model mean/SD from background papyrus only (inside the
    tracer's mask, > 0.35 mm from any traced stroke). The val343 crop is text-dense (~13 % ink), which inflates the SD,
    so its marks read smaller and fewer than the same letters would in a mostly blank lead crop or winding
    (x_refine/reports/R1_ink_ml.md, Test C: hot ink +43-107 %, marks +38-101 % at 1 % ink)."""
    Y0, Y1, X0, X1 = REF['box']
    lab = np.load(f'{W}/x_px343/label_render.npy')[Y0:Y1, X0:X1]; msk = np.load(f'{W}/x_px343/mask_render.npy')[Y0:Y1, X0:X1]
    near = cv2.dilate(lab.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81))) > 0
    bg = valid & msk & ~near
    f = np.mean([(rm[k] - rm[k][bg].mean()) / (rm[k][bg].std() + 1e-6) for k in ('hec', 'rv2', 'tr')], 0) * valid
    return dict(geometry=geometry(f, rr, REF['um']),
                row_mq=round(float(row_score(np.where(rr, f, 0).astype(np.float32), qmask=rr).get('score') or 0), 2))


def zs(a, v):
    x = a[v]; return (a - x.mean()) / (x.std() + 1e-6) * v


def products(m, valid):
    z = {k: zs(m[k], valid) for k in ('hec', 'rv2', 'tr')}
    fused = Z.zfuse([m['hec'], m['rv2'], m['tr']], valid)
    agree = sum((z[k] > 2).astype(np.uint8) for k in z)
    lane = Z.zfuse([m['lane_hec'], m['lane_rv2']], valid)
    return z, fused, agree, lane


def geometry(fused, roi_valid, um, thr=2.0):
    """marks = connected components of fused z > thr (>= 0.15 mm^2); sizes, spacing along rows, row pitch"""
    px = um / 1000.0; hot = ((fused > thr) & roi_valid).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(hot, connectivity=8)
    keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] * px * px >= 0.15]
    H = [st[i, cv2.CC_STAT_HEIGHT] * px for i in keep]; Wm = [st[i, cv2.CC_STAT_WIDTH] * px for i in keep]
    C = np.array([cen[i] for i in keep]) if keep else np.zeros((0, 2))
    gaps = []
    for j, (x, y) in enumerate(C):                       # nearest neighbour to the right within +-1 mm vertically
        same = [(cx - x) for k, (cx, cy) in enumerate(C) if k != j and cx > x and abs(cy - y) * px <= 1.0]
        if same: gaps.append(min(same) * px)
    prof = (hot.sum(1) / np.maximum(roi_valid.sum(1), 1)).astype(np.float32)          # hot fraction per row of pixels
    prof = cv2.GaussianBlur(prof[:, None], (0, 0), 0.4 / px)[:, 0]
    rows = [i for i in range(1, len(prof) - 1) if prof[i] >= prof[i - 1] and prof[i] >= prof[i + 1] and prof[i] > max(0.04, 0.35 * prof.max())]
    rows_merged = []
    for r in rows:                                        # merge peaks closer than 1.5 mm
        if rows_merged and (r - rows_merged[-1]) * px < 1.5:
            if prof[r] > prof[rows_merged[-1]]: rows_merged[-1] = r
        else: rows_merged.append(r)
    pitch = [round((b - a) * px, 2) for a, b in zip(rows_merged, rows_merged[1:])]
    med = lambda v: round(float(np.median(v)), 2) if len(v) else None
    return dict(n_marks=len(keep), mark_h_mm=med(H), mark_w_mm=med(Wm), mark_h_p90=round(float(np.percentile(H, 90)), 2) if H else None,
                gap_mm=med(gaps), rows=len(rows_merged), row_y_mm=[round(r * px, 2) for r in rows_merged], row_pitch_mm=pitch,
                hot_frac=round(float(hot.sum() / max(roi_valid.sum(), 1)), 4))


def glyph(z, um, valid):
    from common import a_from_z, onsheet_windows, WIN as GWIN
    from evaluate import score_windows
    from train import load_model
    torch.cuda.set_per_process_memory_fraction(0.06)
    net = glyph.net = getattr(glyph, 'net', None) or load_model(f'{W}/x_glyph/models/gA.pt', 'cuda')
    A, M = a_from_z(z.astype(np.float32), um, valid)
    ws = onsheet_windows(dict(a=A, mask=M), GWIN, 16, 0.7)
    return round(float(score_windows(net, dict(a=A, mask=M), ws).max()), 3) if ws else None


def depth_profile(vol, c, face, roi, out, ds=range(-4, 5)):
    """Reader v2 + TRACE (no TTA) per depth, scored inside the ROI: masked row score + hot fraction"""
    y0, y1, x0, x1 = roi; res = []
    for d in ds:
        try:
            a = cache(f'{out}/dp_rv2_d{d}.npy', lambda d=d: Z.villa_map('rv2', vol, c, d, face))
            b = cache(f'{out}/dp_tr_d{d}.npy', lambda d=d: Z.villa_map('trace', vol, c, d, face))
        except ValueError:
            continue
        v = vol[c] > 0; f = Z.zfuse([a, b], v)[y0:y1, x0:x1]; vv = v[y0:y1, x0:x1]
        s = row_score(np.where(vv, f, 0).astype(np.float32), qmask=vv)
        res.append(dict(d=d, row=round(float(s.get('score') or 0), 2), hot=round(float(((f > 2) & vv).sum() / max(vv.sum(), 1)), 4)))
    sc = [r['row'] for r in res]; pk = int(np.argmax(sc)) if sc else 0
    at3 = [r['row'] for r in res if abs(r['d'] - res[pk]['d']) == 3] if res else []
    broad = bool(at3) and min(at3) >= 0.5 * max(sc[pk], 1e-6) if sc and sc[pk] > 0 else None
    return dict(profile=res, peak_d=res[pk]['d'] if res else None, broad=broad)


def page(z, valid, um, roi=None):
    """ink as a page at one contrast (z 1..3 -> light..dark), resampled to DISP_UM"""
    if roi: y0, y1, x0, x1 = roi; z, valid = z[y0:y1, x0:x1], valid[y0:y1, x0:x1]
    a = np.clip((z - 1) / 2, 0, 1); g = np.where(valid, 1 - 0.92 * a, 0.12)
    rgb = np.stack([g * 0.80, g * 0.88, g * 0.93], -1) * 255
    s = um / DISP_UM
    return cv2.resize(rgb.astype(np.uint8), (max(1, int(z.shape[1] * s)), max(1, int(z.shape[0] * s))), interpolation=cv2.INTER_AREA)


def agree_img(agree, valid, um, roi=None):
    if roi: y0, y1, x0, x1 = roi; agree, valid = agree[y0:y1, x0:x1], valid[y0:y1, x0:x1]
    g = np.where(valid, 1.0, 0.12) - 0.45 * (agree == 1) - 0.9 * (agree >= 2) * valid
    rgb = np.stack([g, g, g], -1); rgb[..., 2] = np.where(agree >= 2, 0.75, rgb[..., 2])   # >=2 models: dark red
    s = um / DISP_UM
    return cv2.resize((np.clip(rgb, 0, 1) * 255).astype(np.uint8), (max(1, int(agree.shape[1] * s)), max(1, int(agree.shape[0] * s))), interpolation=cv2.INTER_AREA)


def label(img, text, sub=''):
    head = np.full((44 if sub else 28, img.shape[1], 3), 25, np.uint8)
    cv2.putText(head, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 220, 255), 1, cv2.LINE_AA)
    if sub: cv2.putText(head, sub, (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
    out = np.concatenate([head, img], 0); bar = int(10000 / DISP_UM)
    cv2.line(out, (8, out.shape[0] - 10), (8 + bar, out.shape[0] - 10), (40, 40, 220), 4)
    cv2.putText(out, '1 cm', (8, out.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 220), 1, cv2.LINE_AA)
    return out


def hstack(imgs, gap=10):
    h = max(i.shape[0] for i in imgs); pad = lambda i: np.pad(i, ((0, h - i.shape[0]), (0, 0), (0, 0)), constant_values=40)
    out = []
    for i in imgs: out += [pad(i), np.full((h, gap, 3), 40, np.uint8)]
    return np.concatenate(out[:-1], 1)


def vstack(imgs, gap=12):
    w = max(i.shape[1] for i in imgs); pad = lambda i: np.pad(i, ((0, 0), (0, w - i.shape[1]), (0, 0)), constant_values=40)
    out = []
    for i in imgs: out += [pad(i), np.full((gap, w, 3), 40, np.uint8)]
    return np.concatenate(out[:-1], 0)


def text_block(lines, width):
    img = np.full((24 * len(lines) + 12, width, 3), 20, np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (8, 22 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)
    return img


def wrap(t, n=150):
    out, cur = [], ''
    for w in t.split():
        if len(cur) + len(w) + 1 > n: out.append(cur); cur = w
        else: cur = (cur + ' ' + w).strip()
    return out + ([cur] if cur else [])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True); ap.add_argument('--render', required=True); ap.add_argument('--c', type=int, required=True)
    ap.add_argument('--um', type=float, required=True); ap.add_argument('--face', required=True, choices=['forward', 'reverse'])
    ap.add_argument('--roi', type=int, nargs=4, required=True); ap.add_argument('--title', default=''); ap.add_argument('--verdict', default='')
    ap.add_argument('--card-dir', default=f'{W}/CANDIDATES/1_closest_calls/zoom')
    a = ap.parse_args()
    out = f'{W}/x_zoom/{a.name}'; os.makedirs(out, exist_ok=True); t0 = time.time()
    Y0, Y1, X0, X1 = a.roi
    vol, box = Z.crop(a.render, Y0 - MARGIN, Y1 + MARGIN, X0 - MARGIN, X1 + MARGIN)
    roi = (Y0 - box[0], Y1 - box[0], X0 - box[2], X1 - box[2]); valid = vol[a.c] > 0
    rv = np.zeros_like(valid); rv[roi[0]:roi[1], roi[2]:roi[3]] = True; rv &= valid
    m = mq_maps(vol, a.c, a.um, a.face, out)
    z, fused, agree, lane = products(m, valid)
    met = dict(name=a.name, render=a.render, c=a.c, um=a.um, face=a.face, roi=a.roi, crop=box)
    met['geometry'] = geometry(fused, rv, a.um)
    met['glyph_lane'] = glyph(np.where(rv, lane, 0), a.um, rv); met['glyph_mq'] = glyph(np.where(rv, fused, 0), a.um, rv)
    met['row_mq'] = round(float(row_score(np.where(rv, fused, 0)[roi[0]:roi[1], roi[2]:roi[3]].astype(np.float32), qmask=rv[roi[0]:roi[1], roi[2]:roi[3]]).get('score') or 0), 2)
    met['depth'] = depth_profile(vol, a.c, a.face, roi, out)
    # reference: the same recipe on 0343 letters (val343 cache, reverse face)
    rvol_valid = Z.crop(REF['render'], *REF['box'])[0][REF['c']] > 0
    rm = ref_maps(); rz, rfused, ragree, rlane = products(rm, rvol_valid)
    rr = np.zeros_like(rvol_valid); rr[160:-160, 160:-160] = True; rr &= rvol_valid
    ref = dict(geometry=geometry(rfused, rr, REF['um']), glyph_lane=glyph(np.where(rr, rlane, 0), REF['um'], rr),
               glyph_mq=glyph(np.where(rr, rfused, 0), REF['um'], rr),
               row_mq=round(float(row_score(np.where(rr, rfused, 0).astype(np.float32), qmask=rr).get('score') or 0), 2))
    ref['blank'] = ref_blank(rm, rvol_valid, rr)                     # R1: same letters, normalised as a blank lead crop
    met['ref_0343'] = ref
    np.savez_compressed(f'{out}/maps.npz', fused=fused.astype(np.float16), agree=agree, hec=m['hec'].astype(np.float16),
                        rv2=m['rv2'].astype(np.float16), tr=m['tr'].astype(np.float16), valid=valid, roi=np.array(roi))
    json.dump(met, open(f'{out}/metrics.json', 'w'), indent=1)
    # card
    g, rg = met['geometry'], ref['geometry']; dp = met['depth']
    rows = [label(hstack([page(fused, valid, a.um, roi), agree_img(agree, valid, a.um, roi)]),
                  f'LEAD {a.name}: fused max-quality page  |  agreement (dark red = hot in >=2 of 3 models)', a.title),
            label(hstack([page(z['hec'], valid, a.um, roi), page(z['rv2'], valid, a.um, roi), page(z['tr'], valid, a.um, roi)]),
                  'LEAD per model, same contrast: Hecate s16 d-1..+1  |  Reader v2 8xTTA d-1..+1  |  TRACE 8xTTA d-1..+1'),
            label(hstack([page(rfused, rvol_valid, REF['um'], (160, -160, 160, -160)), agree_img(ragree, rvol_valid, REF['um'], (160, -160, 160, -160))]),
                  'REFERENCE PHerc0343 known text (Erwin letters 5-18), SAME recipe and SAME mm scale', 'fl/0343/concat_66.zarr reverse face')]
    lines = [f"geometry   lead: {g['n_marks']} marks, height {g['mark_h_mm']} mm (p90 {g['mark_h_p90']}), width {g['mark_w_mm']} mm, gap {g['mark_gap'] if 'mark_gap' in g else g['gap_mm']} mm, "
             f"rows {g['rows']} pitch {g['row_pitch_mm']} mm, hot {g['hot_frac']}",
             f"           0343: {rg['n_marks']} marks, height {rg['mark_h_mm']} mm (p90 {rg['mark_h_p90']}), width {rg['mark_w_mm']} mm, gap {rg['gap_mm']} mm, "
             f"rows {rg['rows']} pitch {rg['row_pitch_mm']} mm, hot {rg['hot_frac']}",
             f"   0343 blank-normalised (as a mostly blank lead crop): {ref['blank']['geometry']['n_marks']} marks, height {ref['blank']['geometry']['mark_h_mm']} mm "
             f"(p90 {ref['blank']['geometry']['mark_h_p90']}), gap {ref['blank']['geometry']['gap_mm']} mm, hot {ref['blank']['geometry']['hot_frac']}, row {ref['blank']['row_mq']}",
             f"glyph (lane map / max-quality map)   lead {met['glyph_lane']} / {met['glyph_mq']}    0343 {ref['glyph_lane']} / {ref['glyph_mq']}    (top 1% of negatives: 9.06)",
             f"masked row score (max-quality)   lead {met['row_mq']}    0343 {ref['row_mq']}",
             f"depth profile (Rv2+TRACE row score d{dp['profile'][0]['d'] if dp['profile'] else ''}..): " + ' '.join(f"{r['d']:+d}:{r['row']}" for r in dp['profile'])
             + f"   peak d{dp['peak_d']}, {'BROAD (text-like)' if dp['broad'] else 'NARROW (crack-like)' if dp['broad'] is False else 'n/a'}"]
    if a.verdict: lines += [''] + wrap('VERDICT: ' + a.verdict)
    card = vstack(rows + [text_block(lines, max(r.shape[1] for r in rows))])
    os.makedirs(a.card_dir, exist_ok=True)
    cf = f'{a.card_dir}/{a.name}_ZOOM_CARD.jpg'; cv2.imwrite(cf, card, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(json.dumps({k: met[k] for k in ('geometry', 'glyph_lane', 'glyph_mq', 'row_mq')}), flush=True)
    print('ref', json.dumps(ref), flush=True); print('depth', json.dumps(dp), flush=True)
    print(f'card {cf}  ({time.time() - t0:.0f}s)', flush=True)
