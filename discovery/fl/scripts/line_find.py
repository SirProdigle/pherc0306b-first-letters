"""line_find.py : find single LINES of separate letter-sized marks running along the fibres (what the owner's eye found
on PHerc0306B w050 / w074), score them, and write per-winding candidates with polygons (JSON).

Why (x_refine S1, 2026-10-13): the triage scores windows, not lines.  The spectral row score needs >= 2 periodic rows, the
glyph CNN takes a max over 6.4 mm sub-windows, and the hard gates were tuned on a few cases.  A single line of separate
2-3 mm marks is what a reader sees first, so look for it directly.

Detector (fused ink z map, any recipe; all sizes in mm, so any voxel size works):
  1. marks   work grid ~35 um (area mean of z over on-sheet px).  Components of z >= Z_HI (2.0) (hysteresis growth to a
             lower Z_LO merged neighbouring letters into oversized blobs, so Z_LO = Z_HI).
             Fragments closer than LINK_MM (0.15) are one mark.  Kept if area >= MIN_AREA_MM2 (0.06) and largest extent >= 0.4 mm.
  2. chains  for each line angle in ANGLES (default +-15 deg around the winding = map x) and band centre (0.25 mm steps):
             marks whose centroid lies within +-BAND_MM (1.0) of the band centre line, whose extent across the line is
             <= MAX_H_MM (4.0) and along it <= MAX_L_MM (8.0, joined letters) -> sorted along the line -> marks closer than
             SLOT_MM (1.2, centroid gap) or overlapping are one letter slot -> runs split where the slot gap exceeds
             PITCH_MAX (6.5 mm) -> a chain is a run of >= MIN_SLOTS (4) slots, LEN_MIN..LEN_MAX (8..60 mm) long.
             Duplicates (same marks from neighbouring angles / offsets) are removed greedily by mark overlap.
  3. features per chain (the line is refit by area-weighted least squares):
             n_slots, len_mm, pitch_med, pitch_cv, fill, h_med, h_p90, resid_mm, angle,
             contrast = hot share in the +-1 mm band / hot share in the flanks 1.75-3.5 mm either side (text sits between
             blank inter-line strips; blob fields and lattices do not), flank_max (the busier flank), crowd (letter-sized
             marks in the flanks per slot), onsheet (valid share of band + flanks), zmean, stroke (share of mark area in
             elongated marks), and optionally
             glyph_mean / glyph_max: x_glyph CNN logits on 6.4 mm windows centred every 1.6 mm along the chain, and
             glyph_p: (glyph_mean - median) / MAD-sd of the POOLED recipe-level null (GLYPH_REF: every on-sheet window of
             the reviewed-null dev lanes 0306B z9900 + 0268 z11500 for lane maps; the 0306B deep-read maps minus the lead
             windows for max-quality maps).  glyph_z (against the same map's own windows, as R2 proposed) is kept but NOT
             scored: on a map that is mostly text the map's own null is text (0343 lane windings: null median 2-3.6,
             MAD-sd 4-5, against 0.4 / 1.2 on null lanes), which erased every 0343 line (S1 report sec 3.2-3.3),
             crack: enrichment of chain mark pixels within 0.3 mm of long dark CT gaps over the on-sheet base rate
             (fl/scripts/gap_gate.py logic on one CT layer; R2: crack >= 2 = 0/23 text bands) -> crack_veto.
  4. score   see score_chain() (pre-registered in x_refine/reports/S1_line_detector.md; calibration there).

usage:
  line_find.py --lane PHerc0306B_z8400 [--faces reading|both] [--glyph gA|none] [--device cuda|cpu] [--out F.json]
  line_find.py --map Z.npy --um 8.64 [--ct CT.png] [--out F.json]
  line_find.py --deepread [--out F.json]                 (x_deepread/results/b*/out/w0NN*/{fwd,rev}_fused.npy)
GPU (glyph) runs must go through `flock /home/liam/projects/vesuvius-work/x_discord/gpu.lock`; --device cpu works too (slower).
"""
import os, sys, glob, json, argparse, time
import numpy as np, cv2
cv2.setNumThreads(int(os.environ.get('OMP_NUM_THREADS', '4')))
W = '/home/liam/projects/vesuvius-work'

UM_WORK = 35.0
Z_HI, Z_LO = 2.0, 2.0
LINK_MM, MIN_AREA_MM2, MIN_EXT_MM = 0.15, 0.06, 0.4
BAND_MM, MAX_H_MM, MIN_H_MM, MAX_L_MM = 1.0, 4.0, 0.6, 8.0
SLOT_MM, PITCH_MAX, MIN_SLOTS = 1.2, 6.5, 4
LEN_MIN, LEN_MAX = 8.0, 60.0
FLANK = (1.75, 3.5)
ANGLES = tuple(np.arange(-15.0, 15.01, 2.5))
GWIN_MM, GSTEP_MM = 6.4, 1.6
# pooled glyph null per recipe and model (x_refine/S1_line_detector/pooled_null.py, 2026-10-13): (median, MAD-sd) of all
# on-sheet 6.4 mm windows; 'lane' = Hecate+Rv2 cloud-lane maps (28 488 windows of 0306B z9900 + 0268 z11500; per-lane medians
# 0.43-0.88 for gA), 'mq' = max-quality deep-read maps (5 189 windows of 0306B w044-w074, lead windows excluded)
GLYPH_REF = {'lane': {'gA': (0.586, 1.406), 'L_e343': (0.401, 1.13), 'L_w1447': (0.393, 0.929)},
             'mq': {'gA': (0.283, 2.889), 'L_e343': (0.078, 2.862)}}


# ------------------------------------------------------------------ map loading
def skip_mask(z, tile=128):
    """tiles the lane skipped (fastink --mask): >= 90 % of the tile's on-sheet px sit at the map's minimum (glyph_win.py)"""
    valid = z != 0
    if not valid.any(): return np.zeros_like(valid)
    at_min = valid & (z == z[valid].min())
    H, Wd = z.shape; gh, gw = -(-H // tile), -(-Wd // tile)
    pad = lambda a: np.pad(a, ((0, gh * tile - H), (0, gw * tile - Wd)))
    nv = pad(valid).reshape(gh, tile, gw, tile).sum((1, 3)); nm = pad(at_min).reshape(gh, tile, gw, tile).sum((1, 3))
    sk = (nv > 0) & (nm >= 0.9 * nv)
    return np.kron(sk, np.ones((tile, tile), bool))[:H, :Wd] & valid


def load_map(path):
    z = np.nan_to_num(np.load(path).astype(np.float32))
    valid = (z != 0) & ~skip_mask(z)
    return z, valid


def downsample(z, valid, um):
    f = max(1, int(round(UM_WORK / um)))
    H, Wd = z.shape; h, w = H // f, Wd // f
    zs = cv2.resize(np.where(valid, z, 0)[:h * f, :w * f], (w, h), interpolation=cv2.INTER_AREA)
    vs = cv2.resize(valid[:h * f, :w * f].astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    zs = np.where(vs > 0, zs / np.maximum(vs, 1e-6), 0)
    return zs.astype(np.float32), vs > 0.5, um * f, f


# ------------------------------------------------------------------ marks
def extract_marks(zs, vs, umw):
    px = 1000.0 / umw
    lo = ((zs >= Z_LO) & vs).astype(np.uint8)
    n, lab = cv2.connectedComponents(lo, connectivity=8)
    seed = np.zeros(n, bool); seed[np.unique(lab[(zs >= Z_HI) & vs])] = True; seed[0] = False
    hot = seed[lab]
    r = max(1, int(round(LINK_MM * px / 2)))
    grp = cv2.dilate(hot.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    n2, lab2 = cv2.connectedComponents(grp, connectivity=8)
    lab2 = np.where(hot, lab2, 0)
    ys, xs = np.nonzero(lab2); ids = lab2[ys, xs]
    order = np.argsort(ids, kind='stable'); ys, xs, ids = ys[order], xs[order], ids[order]
    cuts = np.flatnonzero(np.diff(ids)) + 1
    marks = []
    for yy, xx in zip(np.split(ys, cuts), np.split(xs, cuts)):
        if len(xx) == 0: continue
        area = len(xx) / px ** 2
        ext = max(xx.max() - xx.min() + 1, yy.max() - yy.min() + 1) / px
        if area < MIN_AREA_MM2 or ext < MIN_EXT_MM: continue
        w_ = np.clip(zs[yy, xx] - Z_LO, 0.05, None)
        cx, cy = np.average(xx, weights=w_), np.average(yy, weights=w_)
        c = np.cov(np.vstack([xx, yy]).astype(np.float64)) + np.eye(2) / 12 if len(xx) > 2 else np.eye(2)
        ev, evec = np.linalg.eigh(c)
        L, Wm = 4 * np.sqrt(max(ev[1], 1e-9)) / px, 4 * np.sqrt(max(ev[0], 1e-9)) / px
        marks.append(dict(cx=cx / px, cy=cy / px, area=area, zmax=float(zs[yy, xx].max()), zmean=float(zs[yy, xx].mean()),
                          elong=L / max(Wm, 1e-6), len_mm=L, xs=(xx / px).astype(np.float32), ys=(yy / px).astype(np.float32)))
    hotmask = lab2 > 0
    return marks, hotmask


def mark_extents(marks, th):
    """per mark: (u_min, u_max, v_min, v_max) in mm along / across a line at angle th (radians, image coords)"""
    c, s = np.cos(th), np.sin(th)
    out = np.zeros((len(marks), 4), np.float32)
    for i, m in enumerate(marks):
        u = m['xs'] * c + m['ys'] * s; v = -m['xs'] * s + m['ys'] * c
        out[i] = (np.percentile(u, 2), np.percentile(u, 98), np.percentile(v, 3), np.percentile(v, 97))
    return out


# ------------------------------------------------------------------ chains
def _slots(idx, uc, ext):
    """group marks (sorted by uc) into letter slots: centroid gap < SLOT_MM or overlapping along the line"""
    slots = [[idx[0]]]
    for a, b in zip(idx[:-1], idx[1:]):
        prev = slots[-1]
        if uc[b] - uc[prev[-1]] < SLOT_MM or ext[b, 0] <= max(ext[k, 1] for k in prev): prev.append(b)
        else: slots.append([b])
    return slots


def find_chains(marks, angles=ANGLES, band=BAND_MM, min_slots=MIN_SLOTS, len_min=LEN_MIN, len_max=LEN_MAX, max_cands=4000):
    if len(marks) < min_slots: return []
    cx = np.array([m['cx'] for m in marks]); cy = np.array([m['cy'] for m in marks]); area = np.array([m['area'] for m in marks])
    found = {}
    for a in angles:
        th = np.deg2rad(a); c, s = np.cos(th), np.sin(th)
        ext = mark_extents(marks, th)
        uc = cx * c + cy * s; vc = -cx * s + cy * c
        h = ext[:, 3] - ext[:, 2]; l = ext[:, 1] - ext[:, 0]
        ok = (h <= MAX_H_MM) & (l <= MAX_L_MM) & (np.maximum(h, l) >= MIN_H_MM)
        cand = np.flatnonzero(ok)
        if len(cand) < min_slots: continue
        for v0 in np.arange(vc[cand].min(), vc[cand].max() + 0.25, 0.25):
            sel = cand[np.abs(vc[cand] - v0) <= band]
            if len(sel) < min_slots: continue
            sel = sel[np.argsort(uc[sel])]
            slots = _slots(list(sel), uc, ext)
            sc = np.array([np.average(uc[sl], weights=area[sl]) for sl in slots])
            run = [0]
            for k in range(1, len(slots) + 1):
                if k == len(slots) or sc[k] - sc[k - 1] > PITCH_MAX:
                    if len(run) >= min_slots:
                        ms = [i for j in run for i in slots[j]]
                        L = ext[ms, 1].max() - ext[ms, 0].min()
                        if L >= len_min:
                            key = frozenset(ms)
                            if key not in found or found[key][0] < len(run):
                                found[key] = (len(run), a, v0)
                    run = [k]
                else:
                    run.append(k)
    # greedy de-duplication: most slots first, then most mark area; drop chains sharing >= 50 % of their marks
    items = sorted(found.items(), key=lambda kv: (-kv[1][0], -area[list(kv[0])].sum()))
    kept, used = [], []
    for key, (ns, a, v0) in items:
        if any(len(key & u) >= 0.5 * min(len(key), len(u)) for u in used): continue
        kept.append(dict(marks=sorted(key), angle0=a, v0=v0)); used.append(key)
        if len(kept) >= max_cands: break
    return kept


def refine(ch, marks):
    """area-weighted line fit through the chain's mark centroids; returns angle (deg) and the line (point, unit dir)"""
    ms = ch['marks']; x = np.array([marks[i]['cx'] for i in ms]); y = np.array([marks[i]['cy'] for i in ms])
    w = np.array([marks[i]['area'] for i in ms])
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    C = np.cov(np.vstack([x - mx, y - my]), aweights=w)
    ev, evec = np.linalg.eigh(C); u = evec[:, 1]
    if u[0] < 0: u = -u
    ang = float(np.degrees(np.arctan2(u[1], u[0])))
    if abs(ang - ch['angle0']) > 3.0: ang, u = ch['angle0'], np.array([np.cos(np.deg2rad(ch['angle0'])), np.sin(np.deg2rad(ch['angle0']))])
    return ang, (mx, my), u


def chain_features(ch, marks, hot, vs, zs, umw):
    px = 1000.0 / umw
    ang, (mx, my), u = refine(ch, marks); vdir = np.array([-u[1], u[0]])
    ms = ch['marks']; th = np.deg2rad(ang)
    ext = mark_extents([marks[i] for i in ms], th)
    uc = np.array([(marks[i]['cx'] - mx) * u[0] + (marks[i]['cy'] - my) * u[1] for i in ms])
    vc = np.array([(marks[i]['cx'] - mx) * vdir[0] + (marks[i]['cy'] - my) * vdir[1] for i in ms])
    area = np.array([marks[i]['area'] for i in ms])
    order = np.argsort(uc); idx = list(order)
    off_u = mx * u[0] + my * u[1]
    rel = np.c_[ext[:, 0] - off_u, ext[:, 1] - off_u]
    slots = _slots(idx, uc, rel)
    sc = np.array([np.average(uc[sl], weights=area[sl]) for sl in slots])
    sv = np.array([np.average(vc[sl], weights=area[sl]) for sl in slots])
    gaps = np.diff(sc)
    u0, u1 = float(rel[:, 0].min()), float(rel[:, 1].max())
    L = u1 - u0
    cov = np.zeros(max(1, int(L * 10) + 1), bool)                 # along-line coverage by slots (0.1 mm bins)
    for sl in slots:
        a_, b_ = rel[sl, 0].min() - u0, rel[sl, 1].max() - u0
        cov[int(max(0, a_) * 10):int(max(0, b_) * 10) + 1] = True
    hs = ext[:, 3] - ext[:, 2]
    # band / flank hot shares on the work grid
    H, Wd = hot.shape
    corners = np.array([[mx + u[0] * uu + vdir[0] * t, my + u[1] * uu + vdir[1] * t] for uu in (u0, u1) for t in (-FLANK[1], FLANK[1])]) * px
    x0, x1 = int(max(0, corners[:, 0].min() - 2)), int(min(Wd, corners[:, 0].max() + 3))
    y0, y1 = int(max(0, corners[:, 1].min() - 2)), int(min(H, corners[:, 1].max() + 3))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    pu = (xx / px - mx) * u[0] + (yy / px - my) * u[1]; pv = (xx / px - mx) * vdir[0] + (yy / px - my) * vdir[1]
    inl = (pu >= u0) & (pu <= u1)
    vv = vs[y0:y1, x0:x1]; hh = hot[y0:y1, x0:x1]
    bandm = inl & (np.abs(pv) <= BAND_MM) & vv
    fl_up = inl & (pv <= -FLANK[0]) & (pv >= -FLANK[1]) & vv
    fl_dn = inl & (pv >= FLANK[0]) & (pv <= FLANK[1]) & vv
    allm = inl & (np.abs(pv) <= FLANK[1])
    hb = hh[bandm].mean() if bandm.any() else 0.0
    hu = hh[fl_up].mean() if fl_up.sum() > 20 else np.nan; hd = hh[fl_dn].mean() if fl_dn.sum() > 20 else np.nan
    hf = np.nanmean([hu, hd]) if not (np.isnan(hu) and np.isnan(hd)) else 0.0
    eps = 0.01
    # letter-sized marks in the flanks (crowding)
    crowd = 0
    for m in marks:
        du = (m['cx'] - mx) * u[0] + (m['cy'] - my) * u[1]; dv = (m['cx'] - mx) * vdir[0] + (m['cy'] - my) * vdir[1]
        if u0 <= du <= u1 and FLANK[0] <= abs(dv) <= FLANK[1] and m['area'] >= 0.3: crowd += 1
    zb = zs[y0:y1, x0:x1][bandm & hh]
    stroke_a = sum(marks[i]['area'] for i in ms if marks[i]['len_mm'] >= 1.0 and marks[i]['elong'] >= 2.0)
    f = dict(n_slots=len(slots), n_marks=len(ms), len_mm=round(L, 2), angle=round(ang, 1),
             pitch_med=round(float(np.median(gaps)), 3) if len(gaps) else None,
             pitch_cv=round(float(np.std(gaps) / max(np.mean(gaps), 1e-6)), 3) if len(gaps) > 1 else None,
             fill=round(float(cov.mean()), 3), h_med=round(float(np.median(hs)), 3), h_p90=round(float(np.percentile(hs, 90)), 3),
             resid_mm=round(float(np.sqrt(np.average(sv ** 2))), 3),
             band_hot=round(float(hb), 4), flank_hot=round(float(hf), 4),
             flank_max=round(float(np.nanmax([hu, hd])) if not (np.isnan(hu) and np.isnan(hd)) else 0.0, 4),
             contrast=round(float((hb + eps) / (hf + eps)), 3), crowd=round(crowd / len(slots), 3),
             onsheet=round(float(vv[allm].mean()) if allm.any() else 0.0, 3),
             zmean=round(float(zb.mean()), 3) if zb.size else None,
             stroke=round(float(stroke_a / max(area.sum(), 1e-6)), 3), mark_area=round(float(area.sum()), 2))
    f['long'] = bool(L > LEN_MAX)
    geom = dict(cx=mx, cy=my, ux=float(u[0]), uy=float(u[1]), u0=u0, u1=u1,
                half=float(max(BAND_MM, np.abs(np.r_[ext[:, 2], ext[:, 3]] - (mx * vdir[0] + my * vdir[1])).max()) + 0.4),
                slots=[float(s) for s in sc])
    return f, geom


def polygon(geom, um):
    """rotated rectangle around the chain (band +- marks), in native map px"""
    px = 1000.0 / um; u = np.array([geom['ux'], geom['uy']]); v = np.array([-u[1], u[0]]); c = np.array([geom['cx'], geom['cy']])
    a, b, h = geom['u0'] - 1.0, geom['u1'] + 1.0, geom['half']
    pts = [c + u * a - v * h, c + u * b - v * h, c + u * b + v * h, c + u * a + v * h]
    return [[round(float(p[0] * px), 1), round(float(p[1] * px), 1)] for p in pts]


# ------------------------------------------------------------------ glyph CNN (x_glyph) along the chain
class Glyph:
    def __init__(self, model='gA', device='cuda'):
        import torch
        sys.path.insert(0, f'{W}/x_glyph')
        from train import GlyphNet
        self.torch = torch
        path = model if model.endswith('.pt') else f'{W}/x_glyph/models/{model}.pt'
        if device == 'cuda' and not torch.cuda.is_available(): device = 'cpu'
        if device == 'cuda': torch.cuda.set_per_process_memory_fraction(0.08)
        ck = torch.load(path, map_location=device, weights_only=False)
        self.net = GlyphNet(**ck.get('arch', {})).to(device).eval(); self.net.load_state_dict(ck['state'])
        self.device, self.name = device, os.path.basename(path)[:-3]

    def prep(self, z, valid, um):
        from common import a_from_z
        A, M = a_from_z(z, um, valid)
        return A, M

    def score(self, A, M, centres, bs=256):
        """centres: list of (y, x) at 50 um -> logits of 128 px windows centred there (5-fold TTA as evaluate.score_windows)"""
        torch = self.torch; G = 128; h2 = G // 2
        Ap = np.pad(A, h2); Mp = np.pad(M.astype(np.float32), h2)
        out = []
        for i in range(0, len(centres), bs):
            cc = centres[i:i + bs]
            a = np.stack([Ap[y:y + G, x:x + G] for y, x in cc]).astype(np.float32)[:, None]
            m = np.stack([Mp[y:y + G, x:x + G] for y, x in cc]).astype(np.float32)[:, None]
            with torch.no_grad():
                at = torch.from_numpy(a).to(self.device); mt = torch.from_numpy(m).to(self.device)
                ctx = torch.autocast('cuda', dtype=torch.float16) if self.device == 'cuda' else torch.autocast('cpu', enabled=False)
                with ctx:
                    s = self.net(at, mt)
                    for k in (1, 2, 3): s = s + self.net(torch.rot90(at, k, (2, 3)), torch.rot90(mt, k, (2, 3)))
                    s = (s + self.net(at.flip(3), mt.flip(3))) / 5
            out.append(s.float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def grid(self, z, valid, um, step=32):
        """glyph logits of 6.4 mm windows centred on a regular grid (step px at 50 um; 32 = 1.6 mm) over the whole map.
        Returns dict(scores [gh, gw] float32 (NaN where the window is < 50 % on-sheet), frac [gh, gw], step, um50)."""
        A, M = self.prep(z, valid, um)
        G = 128; H, Wd = M.shape
        ii = cv2.integral(np.pad(M.astype(np.uint8), G // 2)); ys = np.arange(0, H, step); xs = np.arange(0, Wd, step)
        frac = np.zeros((len(ys), len(xs)), np.float32)
        for i, y in enumerate(ys):
            frac[i] = (ii[y + G, xs + G] - ii[y, xs + G] - ii[y + G, xs] + ii[y, xs]) / (G * G)
        sel = np.argwhere(frac >= 0.5)
        sc = np.full(frac.shape, np.nan, np.float32)
        if len(sel):
            v = self.score(A, M, [(int(ys[i]), int(xs[j])) for i, j in sel])
            sc[sel[:, 0], sel[:, 1]] = v
        return dict(scores=sc, frac=frac, step=step, um50=50.0, model=self.name)


def grid_null(gg, min_frac=0.7):
    v = gg['scores'][(gg['frac'] >= min_frac) & np.isfinite(gg['scores'])]
    if v.size < 20: return None, None
    med = float(np.median(v)); return med, float(1.4826 * np.median(np.abs(v - med))) + 1e-6


def chain_glyph(geom, gg, step_mm=GSTEP_MM):
    """glyph logits sampled (nearest grid centre) every step_mm along the chain"""
    px = 1000.0 / gg['um50'] / gg['step']; c = np.array([geom['cx'], geom['cy']]); u = np.array([geom['ux'], geom['uy']])
    ts = np.arange(geom['u0'] + 1.0, geom['u1'] - 1.0 + 1e-6, step_mm) if geom['u1'] - geom['u0'] > 2.0 else np.array([(geom['u0'] + geom['u1']) / 2])
    gh, gw = gg['scores'].shape
    yi = np.clip(np.round((c[1] + u[1] * ts) * px).astype(int), 0, gh - 1); xi = np.clip(np.round((c[0] + u[0] * ts) * px).astype(int), 0, gw - 1)
    v = gg['scores'][yi, xi]
    return v[np.isfinite(v)]


# ------------------------------------------------------------------ CT gaps (gap_gate.py on one layer)
def crack_near(ct, valid, um):
    """near-crack mask (bool, ct grid) from one CT layer: thin dark gaps >= 1.5 mm long (gap_gate.gap_stats logic)"""
    px = 1000.0 / um; ct = ct.astype(np.float32)
    v = cv2.erode((valid & (ct > 0)).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if v.sum() < 1000: return None, v
    ms = cv2.GaussianBlur(ct, (0, 0), max(0.5, 0.15 * px))
    f = max(1, int(2 * px // 6)); h, w = ct.shape
    sm = lambda a: cv2.resize(cv2.GaussianBlur(cv2.resize(a, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA), (0, 0), 2 * px / f), (w, h))
    loc = sm(np.where(v, ct, 0)) / (sm(v.astype(np.float32)) + 1e-6)
    g = 1 - ms / (loc + 1e-3)
    crack = (g > max(0.3, np.percentile(g[v], 95))) & v
    n, lab, st, _ = cv2.connectedComponentsWithStats(crack.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool); keep[1:] = np.maximum(st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT]) >= 1.5 * px
    crack = keep[lab]
    k = max(3, int(0.3 * px) | 1)
    near = cv2.dilate(crack.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    return near, v


# ------------------------------------------------------------------ scoring (pre-registered; see the report)
def score_chain(f):
    """hand score, form fixed before validation (S1 pre-registration; no fitting):
       glyph term (clipped -3..6) + log2 contrast (isolated line) + log2(n_slots / 4) - 0.5 crowding - edge penalty
       - 1 if longer than 60 mm; crack >= 2 vetoes (score - 10).
       Glyph term = glyph_p (pooled recipe null) when present, else glyph_z.  The pre-registered run used glyph_z (per-map
       null) and failed (recall 2/6 at the 2/100 dev point); glyph_p is the post-hoc fix (S1 report sec 3.3).
       Calibration (lane recipe, glyph_p): known text lines 8.2-11.2 (LODO glyph models); reviewed-null lanes: dev max 6.8,
       0306B test max 7.3; >= 6 = 0.11 (dev) / 1.0 (0306B test) false lines per 100 cm2."""
    g = f.get('glyph_p') if f.get('glyph_p') is not None else f.get('glyph_z'); g = 0.0 if g is None else float(np.clip(g, -3, 6))
    s = g + np.log2(max(f['contrast'], 0.25)) + np.log2(f['n_slots'] / 4.0) - 0.5 * min(f['crowd'], 2.0)
    s -= 2.0 * max(0.0, 0.85 - f['onsheet']) / 0.85
    if f.get('long'): s -= 1.0
    if f.get('crack') is not None and f['crack'] >= 2.0: s -= 10.0
    return round(float(s), 3)


# ------------------------------------------------------------------ one map
def detect(z, valid, um, angles=ANGLES):
    """marks + chains + geometric features (CPU, < 1 s per winding).  -> (cands, marks, meta)"""
    zs, vs, umw, f = downsample(z, valid, um)
    marks, hot = extract_marks(zs, vs, umw)
    out = []
    for ch in find_chains(marks, angles):
        feats, geom = chain_features(ch, marks, hot, vs, zs, umw)
        out.append(dict(feats=feats, geom=geom, marks=ch['marks']))
    return out, marks, dict(n_marks=len(marks), n_chains=len(out), work_um=round(umw, 2), onsheet_cm2=round(float(valid.sum()) * (um / 1e4) ** 2, 3))


def glyph_feats(cands, gg, recipe='lane'):
    """glyph features of each candidate from a glyph grid (Glyph.grid) -> list of dicts (same order); null stats.
    glyph_p uses the pooled recipe null GLYPH_REF[recipe][model] (scored); glyph_z the map's own null (reported only)."""
    med, sd = grid_null(gg)
    ref = GLYPH_REF.get(recipe, GLYPH_REF['lane']).get(gg.get('model'))
    res = []
    for o in cands:
        v = chain_glyph(o['geom'], gg)
        if v.size == 0: res.append(dict(glyph_mean=None, glyph_max=None, glyph_z=None, glyph_zmax=None, glyph_p=None)); continue
        gm = float(v.mean())
        res.append(dict(glyph_mean=round(gm, 3), glyph_max=round(float(v.max()), 3),
                        glyph_z=None if med is None else round((gm - med) / sd, 3), glyph_zmax=None if med is None else round((float(v.max()) - med) / sd, 3),
                        glyph_p=None if ref is None else round((gm - ref[0]) / ref[1], 3)))
    return res, dict(model=gg.get('model'), recipe=recipe, map_median=None if med is None else round(med, 3), map_sd=None if sd is None else round(sd, 3),
                     pooled=ref)


def crack_feats(cands, marks, ct, ct_um, valid):
    """crack enrichment of each candidate's mark pixels (None per candidate if the CT is unusable)"""
    if ct is None or not cands: return [None] * len(cands)
    vct = cv2.resize(valid.astype(np.uint8), (ct.shape[1], ct.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    near, vv = crack_near(ct, vct, ct_um)
    if near is None: return [None] * len(cands)
    base = max(float(near[vv].mean()), 1e-6); k = 1000.0 / ct_um; out = []
    for o in cands:
        xs = np.concatenate([marks[i]['xs'] for i in o['marks']]) * k; ys = np.concatenate([marks[i]['ys'] for i in o['marks']]) * k
        xi = np.clip(xs.astype(int), 0, ct.shape[1] - 1); yi = np.clip(ys.astype(int), 0, ct.shape[0] - 1)
        out.append(round(float(near[yi, xi].mean() / base), 3))
    return out


def finalize(cands, um, keep_marks=False):
    for o in cands:
        o['score'] = score_chain(o['feats'])
        o['poly'] = polygon(o['geom'], um)
        px = 1000.0 / um; g = o['geom']
        o['slots_xy'] = [[round((g['cx'] + g['ux'] * s) * px, 1), round((g['cy'] + g['uy'] * s) * px, 1)] for s in g['slots']]
        if not keep_marks: o.pop('marks', None)
    cands.sort(key=lambda o: -o['score'])
    return cands


def run_map(z, valid, um, glyph=None, ct=None, ct_um=None, angles=ANGLES, keep_marks=False, recipe='lane'):
    """full pipeline for one map.  glyph: None, a Glyph model, or a precomputed grid dict.  ct: one CT layer (any scale; ct_um = its px size)"""
    t = time.time()
    out, marks, meta = detect(z, valid, um, angles)
    gg = glyph.grid(z, valid, um) if (glyph is not None and not isinstance(glyph, dict) and out) else glyph
    meta['glyph_null'] = None
    if isinstance(gg, dict) and out:
        gf, meta['glyph_null'] = glyph_feats(out, gg, recipe)
        for o, g in zip(out, gf): o['feats'].update(g)
    for o, c in zip(out, crack_feats(out, marks, ct, ct_um or um, valid)):
        o['feats']['crack'] = c; o['feats']['crack_veto'] = c is not None and c >= 2.0
    finalize(out, um, keep_marks)
    meta['secs'] = round(time.time() - t, 1)
    return out, meta


# ------------------------------------------------------------------ lanes / deep-read discovery
def lane_maps(lane, faces='reading'):
    """[(winding, side, z path, ct path or None, ct_um_factor)] for a cloud lane (x_vast_plan/results/<lane>)"""
    R = f'{W}/x_vast_plan/results/{lane}'; res = f'{R}/results'
    um = 8.64
    try:
        env = dict(l.strip().split('=', 1) for l in open(f'{res}/lane.env') if '=' in l); um = float(env['UM'])
    except Exception: pass
    face = {}
    rj = f'{R}/review/review.json'
    if os.path.exists(rj):
        face = {k: v[0] for k, v in json.load(open(rj)).get('faces', {}).items()}
    elif faces == 'reading':
        face = reading_faces(lane)
    out = []
    for d in sorted(glob.glob(f'{R}/maps/*_w[0-9][0-9][0-9]')):
        w = d.rsplit('_', 1)[-1]
        for side, tag in (('fwd', 'fwd'), ('rev', 'reverse')):
            zp = glob.glob(f'{d}/*_d0_{tag}_z.npy')
            if not zp: continue
            if faces == 'reading' and face.get(w) not in (None, side): continue
            ct = f'{res}/ct/{w}.jpg'
            out.append(dict(winding=w, side=side, zmap=zp[0], ct=ct if os.path.exists(ct) else None, ct_factor=3,
                            reading=face.get(w), um=um, name=os.path.basename(d)))
    return out


def reading_faces(lane):
    """reading face per winding, as review_lane.py computes it (x_recto rule via ink_lane.reading_face); {} on failure"""
    R = f'{W}/x_vast_plan/results/{lane}'; res = f'{R}/results'; HERE = f'{W}/x_vast_plan'
    try:
        env = dict(l.strip().split('=', 1) for l in open(f'{res}/lane.env') if '=' in l); scroll = env['SCROLL']
        src = open(f'{HERE}/ink_lane.py').read(); ns = {}
        exec(src[src.index('def reading_face'):src.index('def dirs_for')], {'json': json, 'np': np}, ns)
        umb = (glob.glob(f'{HERE}/state/umb_{scroll}_s75.json') or glob.glob(f'{HERE}/state/umb_{scroll}*'))[0]
        face = {}
        for d in sorted(glob.glob(f'{res}/flat/w[0-9][0-9][0-9]')):
            t = glob.glob(f'{d}/**/*.tifxyz', recursive=True) or [d]
            try:
                f_, share = ns['reading_face'](t[0], umb); face[os.path.basename(d)] = 'fwd' if f_ == 'forward' else 'rev'
            except Exception: pass
        return face
    except Exception:
        return {}


def deepread_maps():
    out = []
    for f in sorted(glob.glob(f'{W}/x_deepread/results/b*/out/*/*_fused.npy')):
        d = os.path.dirname(f); n = os.path.basename(d); side = 'fwd' if os.path.basename(f).startswith('fwd') else 'rev'
        w = n.split('_')[0]
        ct = f'{d}/ct20.png'
        out.append(dict(winding=w, side=side, zmap=f, ct=ct if os.path.exists(ct) else None, ct_factor=1, um=8.64,
                        name=f'DR_{n}', box=d.split('/')[-3]))
    return out


def cached_grid(key, z, valid, um, glyph, cache_dir=None, extra_dirs=()):
    """glyph grid for one map, from cache_dir/<key>__<model>.npz (or extra_dirs) if present, else computed (and saved)"""
    m = glyph.name if not isinstance(glyph, str) else glyph
    for d in ([cache_dir] if cache_dir else []) + list(extra_dirs):
        p = f'{d}/{key}__{m}.npz'
        if os.path.exists(p):
            q = np.load(p); return dict(scores=q['scores'], frac=q['frac'], step=int(q['step']), um50=float(q['um50']), model=m)
    if isinstance(glyph, str): return None
    gg = glyph.grid(z, valid, um)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True); tmp = f'{cache_dir}/.{key}__{m}.{os.getpid()}.npz'
        np.savez_compressed(tmp, scores=gg['scores'], frac=gg['frac'], step=gg['step'], um50=gg['um50']); os.replace(tmp, f'{cache_dir}/{key}__{m}.npz')
    return gg


def load_ct(p):
    if p is None: return None
    a = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return None if a is None else a


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lane'); ap.add_argument('--map'); ap.add_argument('--deepread', action='store_true')
    ap.add_argument('--um', type=float, default=8.64); ap.add_argument('--ct'); ap.add_argument('--ct-factor', type=float, default=1)
    ap.add_argument('--faces', default='reading', choices=['reading', 'both'])
    ap.add_argument('--glyph', default='gA'); ap.add_argument('--device', default='cuda')
    ap.add_argument('--angles', default='', help='e.g. -90:90:5 for unknown orientation (default -15:15:2.5)')
    ap.add_argument('--out'); ap.add_argument('--top', type=int, default=0, help='keep only the top N chains per map (0 = all)')
    ap.add_argument('--glyph-cache', default='', help='dir of cached glyph grids <source>_<map>_<side>__<model>.npz (read + write)')
    ap.add_argument('--recipe', default='auto', choices=['auto', 'lane', 'mq'], help='pooled glyph null: lane maps or max-quality (auto: --deepread = mq)')
    a = ap.parse_args()
    angles = ANGLES if not a.angles else tuple(np.arange(*[float(x) for x in a.angles.split(':')]))
    gl = None if a.glyph == 'none' else Glyph(a.glyph, a.device)
    if a.map:
        jobs = [dict(winding=os.path.basename(a.map), side='?', zmap=a.map, ct=a.ct, ct_factor=a.ct_factor, um=a.um, name=os.path.basename(a.map))]
    elif a.lane:
        jobs = lane_maps(a.lane, a.faces)
    elif a.deepread:
        jobs = deepread_maps()
    else:
        ap.error('one of --lane / --map / --deepread')
    res = dict(source=a.lane or a.map or 'deepread', glyph=a.glyph, params=dict(Z_HI=Z_HI, Z_LO=Z_LO, BAND_MM=BAND_MM, MIN_SLOTS=MIN_SLOTS,
               LEN=[LEN_MIN, LEN_MAX], PITCH_MAX=PITCH_MAX, angles=[float(angles[0]), float(angles[-1])]), maps=[])
    src = a.lane or ('deepread' if a.deepread else 'map')
    for j in jobs:
        z, valid = load_map(j['zmap'])
        ct = load_ct(j.get('ct')); ct_um = j['um'] * j.get('ct_factor', 1)
        if ct is not None and j.get('ct_factor', 1) == 1 and not a.map:
            ct = ct[::3, ::3]; ct_um = j['um'] * 3          # CT always at 1/3 resolution (the lane preview's scale; calibration)
        gg = cached_grid(f"{src}_{j['name']}_{j['side']}", z, valid, j['um'], gl, a.glyph_cache or None) if gl is not None else None
        cands, meta = run_map(z, valid, j['um'], gg, ct, ct_um, angles, recipe=('mq' if a.deepread else 'lane') if a.recipe == 'auto' else a.recipe)
        if a.top: cands = cands[:a.top]
        for c in cands: c.pop('geom', None)
        res['maps'].append(dict(j, **meta, candidates=cands))
        best = cands[0] if cands else None
        print(f"{j['name']} {j['side']}: {meta['n_marks']} marks, {meta['n_chains']} chains, {meta['onsheet_cm2']} cm2, {meta['secs']}s"
              + (f"; best {best['score']} at x {np.mean([p[0] for p in best['poly']]):.0f} y {np.mean([p[1] for p in best['poly']]):.0f}" if best else ''), flush=True)
    if a.out:
        json.dump(res, open(a.out, 'w'))


if __name__ == '__main__':
    main()
