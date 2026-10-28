"""Training data: real negative windows from the screening corpus (train split, no PHerc0343, no zoom/deep),
synthetic text composited into them (positives) and synthetic non-letter artefacts composited the same way
(hard negatives)."""
import glob, math
import numpy as np, cv2, torch
cv2.setNumThreads(1)
from common import *
import synth

CROP = 182   # >= WIN*sqrt(2): rotate then centre-crop


def load_corpus(cats=('train',), srcs=('s2', 'v8')):
    items = []
    for src in srcs:
        for f in sorted(glob.glob(f'{G}/cache/corpus/{src}/*.npz')):
            d = np.load(f)
            if str(d['cat']) not in cats:
                continue
            it = dict(path=f, a=d['a'], mask=d['mask'], model=str(d['model']), scroll=str(d['scroll']))
            ws = onsheet_windows(dict(mask=it['mask']), WIN, 16, 0.7)
            if not ws:
                continue
            it['wins'] = np.array(ws, np.int32)
            items.append(it)
    return items


def rot_window(a, m, y, x, ang, flip):
    """Window of WIN px centred at (y+WIN/2, x+WIN/2) after rotation by ang (deg) and optional flip."""
    cy, cx = y + WIN // 2, x + WIN // 2
    h = CROP // 2
    H, Wd = a.shape
    pa = np.zeros((CROP, CROP), np.float32); pm = np.zeros((CROP, CROP), np.float32)
    y0, x0 = cy - h, cx - h
    ys0, xs0, ys1, xs1 = max(0, y0), max(0, x0), min(H, y0 + CROP), min(Wd, x0 + CROP)
    pa[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0] = a[ys0:ys1, xs0:xs1] * (1 / 255.0 if a.dtype == np.uint8 else 1.0)
    pm[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0] = m[ys0:ys1, xs0:xs1]
    M = cv2.getRotationMatrix2D((h, h), ang, 1.0)
    ra = cv2.warpAffine(pa, M, (CROP, CROP), flags=cv2.INTER_LINEAR)
    rm = cv2.warpAffine(pm, M, (CROP, CROP), flags=cv2.INTER_NEAREST)
    o = (CROP - WIN) // 2
    ra, rm = ra[o:o + WIN, o:o + WIN], rm[o:o + WIN, o:o + WIN] > 0.5
    if flip:
        ra, rm = ra[:, ::-1], rm[:, ::-1]
    return np.ascontiguousarray(ra), np.ascontiguousarray(rm)


def load_realpos(sets):
    """Real known-text windows (protocol-A positives) and in-segment text-free windows (protocol-A negatives)
    from held-out sets used for a leave-one-dataset-out fine-tune."""
    from evaluate import protocols
    pos, neg = [], []
    for p in sorted(glob.glob(f'{G}/cache/heldout/*.npz')):
        it = load_item(p)
        if it['set'] not in sets:
            continue
        P, NA, _ = protocols(it, 16)
        fam = {'v8in': 'v8in', 'v8inPub': 'v8in', 's42': 'fused_ens3', 's43': 'fused_ens3', 'ens3': 'fused_ens3'}.get(it['model'], 'fused_hd')
        if P:
            pos.append(dict(a=it['a'], mask=it['mask'], wins=np.array(P, np.int32), model=fam, name=it['name']))
        if NA:
            neg.append(dict(a=it['a'], mask=it['mask'], wins=np.array(NA, np.int32), model=fam, name=it['name']))
    return pos, neg


def load_legib_cohort():
    """Real legibility-labelled ink-map crops from LimeGS/herculaneum-legibility-proxy-labels-v5 (PHercParis4, 0139,
    1667, 0814, 0172; none of our held-out scrolls). uint8 [N,160,160] stretch160 tensors of ~1 cm windows + labels."""
    import json
    T = np.load(f'{G}/ext/legib_ds/sources/baseline_tensors.npy', allow_pickle=False)
    rows = [json.loads(l) for l in open(f'{G}/ext/legib_ds/sources/inputs.jsonl')]
    y = np.array([r['y'] for r in rows], np.int8)
    sc = np.array([r['physical_scroll'] for r in rows])
    return T, y, sc


def legib_window(img, rng):
    """1 cm 160-px crop -> 50 um grid (200 px, scale jitter), rotate, centre 128 crop, a-map-like intensity."""
    S = int(200 * rng.uniform(0.85, 1.2))
    x = cv2.resize(img.astype(np.float32) / 255, (S, S), interpolation=cv2.INTER_LINEAR)
    if S < CROP:
        p = (CROP - S + 1) // 2; x = np.pad(x, p)
    h = x.shape[0] / 2
    M = cv2.getRotationMatrix2D((h, h), rng.uniform(0, 360), 1.0)
    x = cv2.warpAffine(x, M, (x.shape[1], x.shape[0]))
    o0, o1 = int(h - WIN / 2), int(h - WIN / 2)
    x = x[o0:o0 + WIN, o1:o1 + WIN]
    if rng.random() < 0.5:
        x = x[:, ::-1]
    t = rng.uniform(0.0, 0.4)
    x = np.clip((x - t) / (1 - t), 0, 1)
    return np.ascontiguousarray(x, dtype=np.float32), np.ones((WIN, WIN), bool)


def stretch_like_legib(a, m, rng):
    """Domain mixing: give a window the look of the legib cohort (62.5 um resample + per-window percentile stretch)."""
    x = cv2.resize(cv2.resize(a, (102, 102), interpolation=cv2.INTER_AREA), (WIN, WIN), interpolation=cv2.INTER_LINEAR)
    act = x[x > 0.04]
    if act.size > 50:
        lo, hi = np.percentile(act, [2, 99.5]); x = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
    t = rng.uniform(0.0, 0.4)
    x = np.clip((x - t) / (1 - t), 0, 1) * m
    return x.astype(np.float32)


class SynthDS(torch.utils.data.IterableDataset):
    """Yields (x[1,WIN,WIN] float32, mask[1,WIN,WIN], y, kind_id).
    kind_id: 0 real negative, 1 synthetic text, 2 synthetic artefact, 3 text + artefact."""

    def __init__(self, items, n, seed=0, p_text=0.45, p_art=0.25, p_both=0.05, artefacts=True, model_weights=None):
        model_weights = model_weights or {'fused_hd': 0.6, 'v8in': 0.25, 'fused_ens3': 0.15}
        self.items, self.n, self.seed = items, n, seed
        self.p = (p_text, p_art if artefacts else 0.0, p_both)
        # sample items balanced by model type (fused_hd dominates otherwise)
        models = np.array([it['model'] for it in items])
        w = np.ones(len(items))
        for mdl in set(models):
            sel = models == mdl
            target = (model_weights or {}).get(mdl, 1.0)
            w[sel] = target / sel.sum()
        self.w = w / w.sum()

    def set_real(self, pos, neg, p_realpos=0.3, p_realneg=0.1):
        self.rpos, self.rneg, self.p_rp, self.p_rn = pos, neg, p_realpos, (p_realneg if neg else 0.0)
        return self

    def set_legib(self, T, y, p_pos=0.3, p_neg=0.08, p_mix=0.3):
        self.LT, self.Lpos, self.Lneg = T, np.nonzero(y == 1)[0], np.nonzero(y == 0)[0]
        self.p_lp, self.p_ln, self.p_mix = p_pos, p_neg, p_mix
        return self

    def sample_from(self, pool, rng):
        it = pool[rng.integers(len(pool))]
        y, x = it['wins'][rng.integers(len(it['wins']))]
        y = int(np.clip(y + rng.integers(-8, 9), 0, it['a'].shape[0] - WIN)); x = int(np.clip(x + rng.integers(-8, 9), 0, it['a'].shape[1] - WIN))
        return rot_window(it['a'], it['mask'], y, x, rng.uniform(0, 360), rng.random() < 0.5)

    def sample_bg(self, rng):
        it = self.items[rng.choice(len(self.items), p=self.w)]
        y, x = it['wins'][rng.integers(len(it['wins']))]
        a, m = rot_window(it['a'], it['mask'], int(y), int(x), rng.uniform(0, 360), rng.random() < 0.5)
        self.fam = it['model']
        return a, m

    def __iter__(self):
        torch.set_num_threads(1); cv2.setNumThreads(1)
        wi = torch.utils.data.get_worker_info()
        wid, nw = (wi.id, wi.num_workers) if wi else (0, 1)
        rng = np.random.default_rng(self.seed * 1000 + wid)
        for _ in range(self.n // nw):
            a, m = self.sample_bg(rng)
            u = rng.random()
            pt, pa, pb = self.p
            v = rng.random()
            if getattr(self, 'LT', None) is not None and v < self.p_lp + self.p_ln:
                i = rng.choice(self.Lpos) if v < self.p_lp else rng.choice(self.Lneg)
                a, m = legib_window(self.LT[i], rng); y, k = (1, 6) if v < self.p_lp else (0, 7)
            elif getattr(self, 'rpos', None) and v < self.p_rp:
                a, m = self.sample_from(self.rpos, rng); y, k = 1, 4
            elif getattr(self, 'rneg', None) and v < self.p_rp + self.p_rn:
                a, m = self.sample_from(self.rneg, rng); y, k = 0, 5
            elif u < pt:
                s, _ = synth.make_text(rng, self.fam); a, _ = synth.composite(a, m, s, rng); y, k = 1, 1
            elif u < pt + pb:
                s, _ = synth.make_text(rng, self.fam); s2, _ = synth.make_artefact(rng, family=self.fam)
                a, _ = synth.composite(a, m, np.maximum(s, s2 * rng.uniform(0.5, 1)), rng); y, k = 1, 3
            elif u < pt + pb + pa:
                s, _ = synth.make_artefact(rng, family=self.fam); a, _ = synth.composite(a, m, s, rng); y, k = 0, 2
            else:
                y, k = 0, 0
            if getattr(self, 'LT', None) is not None and k not in (6, 7) and rng.random() < self.p_mix:
                a = stretch_like_legib(a, m, rng)
            # global gain jitter so the net cannot key on exact saturation levels
            if rng.random() < 0.3:
                a = np.clip(a * rng.uniform(0.7, 1.2), 0, 1)
            yield (torch.from_numpy(a[None].astype(np.float32)), torch.from_numpy(m[None].astype(np.float32)),
                   torch.tensor(float(y)), torch.tensor(k))
