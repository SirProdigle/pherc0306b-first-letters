"""v8306lib.py : data, augmentation, loss and checkpoint helpers for the M306 TRAIN lane.

Reimplements the organisers' LOO-w062 fine-tune of v8-in (W/models/community/v8in_1447_loo_w062/README.md and
training/configs/loo_w062.json; their training code is not run, only read via Q1/Q6 notes):
  * tiles: 64 px on a stride-48 grid anchored at (0, 0), admitted only when the certainty mask covers the whole tile
    (w058 1,512 + w060 1,206 = 2,718 tiles, = the card); unlabelled pixels inside the mask are negatives;
  * depth order: 1447 HF layers reversed (23 -> 0), clipped at 200, / 255;
  * augmentation (config 'augmentation' of loo_w062.json): h/v flips p .5; shift-scale-rotate p .75 (rotation +-360 deg,
    scale +-10 %, shift +-15 %); Gaussian or motion blur p .4 (kernel 3..7); coarse dropout p .5 (2 holes 12 x 12 px, label
    zeroed in the hole as the GP's mask_fill_value=0); "fourth augment" p .6 (21..24 layers re-pasted at a random depth
    offset, with p .6 up to 2 layers zeroed); no brightness/contrast;
  * loss: 0.5 x Dice (smp binary: soft labels, aggregated over the micro-batch, 0 when it has no ink) + 0.5 x SoftBCE (smp
    smoothing 0.25: targets 0.25 / 0.75); both restricted to 'keep' pixels (inside the certainty mask);
  * BatchNorm in TRAIN mode at micro-batch 4 (as the released LOO weights show: num_batches_tracked +6,790 = 2,716 / 4 x 10).
Deviation (documented): shift-scale-rotate samples real context from a 128 px window instead of padding the 64 px tile, and
pixels rotated in from outside the certainty mask are ignored (keep = 0) instead of being zero-filled negatives.
"""
import hashlib, json, math, os, sys
import numpy as np, cv2, torch, torch.nn.functional as F

cv2.setNumThreads(1)
W = '/home/liam/projects/vesuvius-work'
P3 = f'{W}/.claude/worktrees/leadA-push3/x_push3'
TR = f'{P3}/m306/train'
BASE_DIR = f'{W}/third_party/v8in'
BASE_SHA = '3b94548d7f9ba9f99bc4b0ad8cafcec6d42585c06304e1bd11e47b5d0f88c4cd'
LOO_DIR = f'{W}/models/community/v8in_1447_loo_w062'
LOO_SHA = '4b28e153e0a631679c79b6c9b8e3e159108d75aeddc6fdea648708ffec5026b5'
D1447 = f'{W}/data/pherc1447'
TILE, DEPTH, CLIP = 64, 24, 200
CTX = 128                                   # context window for shift-scale-rotate (>= 64 * sqrt(2) * 1.1 + 2 * 9.6)
sys.path.insert(0, BASE_DIR)
from ink8um import InkDetector              # noqa: E402


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()


def integral(m):
    return np.pad(m.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))


def box_sum(I, Y, X, s=TILE):
    return I[Y + s, X + s] - I[Y, X + s] - I[Y + s, X] + I[Y, X]


# ------------------------------------------------------------------------------------------------------------- data sources
def shift_bool(a, dy, dx):
    """translate a 2-D bool array by (dy, dx) px (zeros shifted in)"""
    out = np.zeros_like(a); H, Wd = a.shape
    out[max(0, dy):H + min(0, dy), max(0, dx):Wd + min(0, dx)] = a[max(0, -dy):H + min(0, -dy), max(0, -dx):Wd + min(0, -dx)]
    return out


def src_1447(w, erode_px=0, decoy_shift_px=None, stride=48, label_mode='soft', core_r_px=14):
    """PHerc1447 winding w (HF layout).  Returns dict(img (h, w, 24) uint8 MODEL order (reversed, clipped), lab (h, w) float32 in
    [0, 1], keep (h, w) uint8 (certainty mask), org (n, 2) tile top-lefts in the crop, info).  The crop = mask bbox + CTX margin
    (zero-padded beyond the render).  decoy_shift_px = (dy, dx): labels translated by that many px (hallucination control; the
    mask and tile admission stay as they are, so the shifted strokes land on whatever papyrus is there)."""
    import tifffile
    d = f'{D1447}/{w}'
    lab = cv2.imread(f'{d}/labels/inklabels.png', 0); msk = cv2.imread(f'{d}/labels/mask.png', 0) > 0
    H, Wd = msk.shape
    # tile grid = the organisers' (anchored at 0, stride 48, mask covers the tile)
    I = integral(msk)
    Y, X = np.meshgrid(np.arange(0, H - TILE + 1, stride), np.arange(0, Wd - TILE + 1, stride), indexing='ij'); Y, X = Y.ravel(), X.ravel()
    ok = box_sum(I, Y, X) == TILE * TILE
    Y, X = Y[ok], X[ok]
    m = CTX
    y0, y1, x0, x1 = Y.min() - m, Y.max() + TILE + m, X.min() - m, X.max() + TILE + m
    def crop(a, fill=0):
        out = np.full((y1 - y0, x1 - x0) + a.shape[2:], fill, a.dtype)
        sy0, sy1, sx0, sx1 = max(0, y0), min(H, y1), max(0, x0), min(Wd, x1)
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = a[sy0:sy1, sx0:sx1]
        return out
    layers = []
    for i in range(DEPTH - 1, -1, -1):                                         # model order = reversed
        a = tifffile.imread(f'{d}/layers/{i:02d}.tif')
        layers.append(crop(np.minimum(a, CLIP).astype(np.uint8)))
    img = np.ascontiguousarray(np.stack(layers, -1))
    labf = lab.astype(np.float32) / 255.0
    if label_mode == 'hard': labf = (labf > 0.5).astype(np.float32)
    if erode_px:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        labf = cv2.erode(labf, k)
    if decoy_shift_px is not None:
        dy, dx = decoy_shift_px
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        labf = cv2.warpAffine(labf, M, (Wd, H), flags=cv2.INTER_NEAREST, borderValue=0)
    keepm = msk.astype(np.uint8)
    if label_mode == 'core':                                                   # thin targets (P2 v8T): core = erode(r) U skeleton,
        from skimage.morphology import skeletonize                             # band label \ core ignored
        lab_b = labf > 0.5; r = int(core_r_px)
        core = (cv2.erode(lab_b.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0) | skeletonize(lab_b)
        keepm = (msk & ~(lab_b & ~core)).astype(np.uint8); labf = core.astype(np.float32)
    labc = crop(labf); keep = crop(keepm)
    org = np.stack([Y - y0, X - x0], 1).astype(np.int64)
    info = dict(source=f'PHerc1447 {w}', path=d, crop=[int(y0), int(y1), int(x0), int(x1)], tiles=int(len(org)),
                mask_px=int(msk.sum()), ink_frac_in_mask=round(float(labf[msk].mean()), 4), erode_px=erode_px,
                decoy_shift_px=decoy_shift_px, label_mode=label_mode, core_r_px=core_r_px if label_mode == 'core' else None,
                order='reversed (23 -> 0)', target_frac_in_mask=round(float(labf[msk].mean()), 4))
    return dict(img=img, lab=labc, keep=keep, org=org, info=info, name=w)


def src_human(obj, seg, min_keep=0.5, stride=48, decoy_shift_px=None):
    """organiser human labels v2 (A-click DATA registration) on a native catalog render (forward order, 24 layers from layer0),
    resampled to 8.64 um.  lab = ink, keep = (ink | blank) & pap; tiles on a stride-48 grid with >= min_keep of the tile kept.
    Returns a LIST of sources, one per cluster of labelled area (keeps RAM small)."""
    import zarr
    A = f'{P3}/aclick/data'
    tj = json.load(open(f'{A}/labels_train/{obj.replace("PHerc", "")}_{seg}.json'))
    assert tj['split_name'] == 'train', (obj, seg, tj['split_name'])
    z = np.load(f'{A}/labels_registered/{seg}_human_v2.npz')
    keep_all = (z['ink'] | z['blank']) & z['pap']; ink_all = z['ink'] & z['pap']
    if decoy_shift_px is not None:                                             # ink translated in the native frame (px scaled
        f0 = float(tj['native_px_um']) / 8.64                                  # 8.64 -> native um), keep mask unchanged
        ink_all = shift_bool(ink_all, int(round(decoy_shift_px[0] / f0)), int(round(decoy_shift_px[1] / f0))) & keep_all
    v = zarr.open(tj['native_zarr'], mode='r'); v = v if hasattr(v, 'shape') else v['0']
    H, Wd = v.shape[1:]; l0 = int(tj['layer0']); assert not tj['reverse']
    f = float(tj['native_px_um']) / 8.64; m = CTX
    n, cc = cv2.connectedComponents((cv2.dilate(keep_all.astype(np.uint8), np.ones((129, 129), np.uint8)) > 0).astype(np.uint8))
    out = []
    for c in range(1, n):
        reg = (cc == c) & keep_all
        if reg.sum() < 2000: continue
        ys, xs = np.nonzero(reg)
        y0, y1, x0, x1 = max(0, ys.min() - m), min(H, ys.max() + 1 + m), max(0, xs.min() - m), min(Wd, xs.max() + 1 + m)
        vol = np.asarray(v[l0:l0 + DEPTH, y0:y1, x0:x1])
        img = np.ascontiguousarray(np.minimum(vol, CLIP).astype(np.uint8).transpose(1, 2, 0))
        sz = (int(round((x1 - x0) * f)), int(round((y1 - y0) * f)))
        img = np.ascontiguousarray(np.stack([cv2.resize(img[..., i], sz, interpolation=cv2.INTER_LINEAR) for i in range(DEPTH)], -1))
        lab = cv2.resize((ink_all & reg)[y0:y1, x0:x1].astype(np.float32), sz, interpolation=cv2.INTER_LINEAR)
        kp = (cv2.resize(reg[y0:y1, x0:x1].astype(np.uint8), sz, interpolation=cv2.INTER_NEAREST) > 0) & (img.min(-1) > 0)
        I = integral(kp); Hh, Ww = kp.shape
        Y, X = np.meshgrid(np.arange(0, Hh - TILE + 1, stride), np.arange(0, Ww - TILE + 1, stride), indexing='ij'); Y, X = Y.ravel(), X.ravel()
        ok = (box_sum(I, Y, X) >= min_keep * TILE * TILE) & (Y >= m // 2) & (X >= m // 2) & (Y + TILE + m // 2 <= Hh) & (X + TILE + m // 2 <= Ww)
        org = np.stack([Y[ok], X[ok]], 1).astype(np.int64)
        if not len(org): continue
        info = dict(source=f'{obj} {seg} organiser human labels v2 (part {c})', path=tj['native_zarr'], crop=[int(y0), int(y1), int(x0), int(x1)],
                    resample=round(f, 4), tiles=int(len(org)), keep_px=int(kp.sum()), ink_frac_in_keep=round(float(lab[kp].mean()), 4),
                    order='forward', layer0=l0, um=tj['native_px_um'], decoy_shift_px=decoy_shift_px)
        out.append(dict(img=img, lab=lab.astype(np.float32), keep=kp.astype(np.uint8), org=org, info=info, name=f'{obj}_{seg}_{c}'))
    return out


def src_0343(stride=48, label_mode='soft', core_r_px=14, decoy_shift_px=None):
    """PHerc0343 (Erwin's concat_w047-w048 segment, our render fl/0343/concat_66.zarr, centre layer 33, order desc as P2's v8Q),
    jeff-j-chen stroke tracings (MIT, commit 5f6f634, registered to the render in W/x_px343: label_render.npy / mask_render.npy).
    Supervised inside the tracer's train mask (and full 24-layer CT); unlabelled px there = negatives; tiles on the stride-48 grid
    fully inside it (the organisers' admission rule)."""
    import zarr
    lab = np.load(f'{W}/x_px343/label_render.npy') > 0; msk = np.load(f'{W}/x_px343/mask_render.npy') > 0
    if decoy_shift_px is not None:                                             # labels translated (mask / admission unchanged)
        lab = shift_bool(lab, *decoy_shift_px)
    v = zarr.open(f'{W}/fl/0343/concat_66.zarr', mode='r')['0']
    H, Wd = msk.shape; assert v.shape[1:] == (H, Wd), (v.shape, msk.shape)
    ys, xs = np.nonzero(msk); m = CTX
    y0, y1, x0, x1 = max(0, ys.min() - m), min(H, ys.max() + 1 + m), max(0, xs.min() - m), min(Wd, xs.max() + 1 + m)
    st = np.asarray(v[33 - 12:33 + 12, y0:y1, x0:x1])[::-1]                    # desc = model order
    img = np.ascontiguousarray(np.minimum(st, CLIP).astype(np.uint8).transpose(1, 2, 0))
    keep = msk[y0:y1, x0:x1] & (img.min(-1) > 0); labf = lab[y0:y1, x0:x1].astype(np.float32)
    if label_mode == 'core':
        from skimage.morphology import skeletonize
        lb = labf > 0.5; r = int(core_r_px)
        core = (cv2.erode(lb.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0) | skeletonize(lb)
        keep = keep & ~(lb & ~core); labf = core.astype(np.float32)
    I = integral(keep); Hh, Ww = keep.shape
    # grid anchored at the render origin (as the organisers' at the image origin), expressed in crop coordinates
    gy, gx = (-y0) % stride, (-x0) % stride
    Y, X = np.meshgrid(np.arange(gy, Hh - TILE + 1, stride), np.arange(gx, Ww - TILE + 1, stride), indexing='ij'); Y, X = Y.ravel(), X.ravel()
    ok = box_sum(I, Y, X) == TILE * TILE
    org = np.stack([Y[ok], X[ok]], 1).astype(np.int64)
    info = dict(source='PHerc0343 jeff-j-chen tracings (MIT)', path=f'{W}/fl/0343/concat_66.zarr', crop=[int(y0), int(y1), int(x0), int(x1)],
                tiles=int(len(org)), keep_px=int(keep.sum()), ink_frac_in_keep=round(float(labf[keep].mean()), 4), order='desc (c 33)',
                label_mode=label_mode, decoy_shift_px=decoy_shift_px)
    return dict(img=img, lab=labf, keep=keep.astype(np.uint8), org=org, info=info, name='0343')


def load_sources(cfg):
    out = []
    for s in cfg['sources']:
        sh = s.get('decoy_shift_mm')
        dsp = None if sh is None else (int(round(sh[0] * 1000 / 8.64)), int(round(sh[1] * 1000 / 8.64)))
        if s['kind'] == '1447':
            d = src_1447(s['winding'], erode_px=s.get('erode_px', 0), decoy_shift_px=dsp, label_mode=s.get('label_mode', 'soft'),
                         core_r_px=s.get('core_r_px', 14))
        elif s['kind'] == '0343':
            d = src_0343(label_mode=s.get('label_mode', 'soft'), core_r_px=s.get('core_r_px', 14), decoy_shift_px=dsp)
        elif s['kind'] == 'human_v2':
            for d in src_human(s['object'], s['segment'], min_keep=s.get('min_keep', 0.5), decoy_shift_px=dsp):
                d['weight_map'] = None; d['group'] = s.get('group', 'default'); out.append(d)
            continue
        else:
            raise SystemExit(f'unknown source kind {s["kind"]}')
        d['weight_map'] = None; d['group'] = s.get('group', 'default')
        if s.get('faint_weight'):
            d['weight_map'] = faint_weight_map(d, s['faint_weight'])
        out.append(d)
    return out


def faint_weight_map(d, fw):
    """faint-ink emphasis: weight = 1 + fw['gain'] * (ink label) * (1 - base v8-in prob) using the released base prediction of the
    winding (predictions/v8in.png, same grid).  Returns float32 (h, w) on the crop."""
    w = d['name']; y0, y1, x0, x1 = d['info']['crop']
    p = cv2.imread(f'{D1447}/{w}/predictions/v8in.png', 0).astype(np.float32) / 255.0
    H, Wd = p.shape
    out = np.zeros((y1 - y0, x1 - x0), np.float32)
    sy0, sy1, sx0, sx1 = max(0, y0), min(H, y1), max(0, x0), min(Wd, x1)
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = p[sy0:sy1, sx0:sx1]
    return (1.0 + fw['gain'] * d['lab'] * (1.0 - out)).astype(np.float32)


# ------------------------------------------------------------------------------------------------------------- sampling
def epoch_order(n, seed, epoch):
    return np.random.default_rng([seed, 7919, epoch]).permutation(n)


class Stream:
    """the global micro-batch stream.  Default (organisers' 'shuffle' sampler): epoch e = the union tile list shuffled with
    (seed, e), cut into micro-batches of `micro` (last partial dropped); optimiser step s uses micro-batches [s * accum, (s + 1) *
    accum).  With group_weights {group: w} (sources carry 'group'): every tile slot draws a group with probability w, then a tile
    uniformly from that group's union list (seeded by (seed, micro-batch, slot)); `steps` must then be given."""
    def __init__(self, sources, micro, accum, seed, epochs=None, steps=None, group_weights=None):
        self.idx = np.concatenate([np.stack([np.full(len(d['org']), i), np.arange(len(d['org']))], 1) for i, d in enumerate(sources)])
        self.n = len(self.idx); self.micro, self.accum, self.seed = micro, accum, seed
        self.per_epoch = self.n // micro
        self.gw = group_weights
        if group_weights:
            assert steps, 'group sampling needs steps'
            self.groups = sorted(group_weights); g_of = np.array([sources[i].get('group', 'default') for i in self.idx[:, 0]])
            self.gidx = [self.idx[g_of == g] for g in self.groups]
            assert all(len(x) for x in self.gidx), 'empty group'
            p = np.array([group_weights[g] for g in self.groups], np.float64); self.gp = p / p.sum()
        self.total_micro = epochs * self.per_epoch if epochs else steps * accum
        self.steps = math.ceil(self.total_micro / accum)
        self._cache = {}

    def micro_batch(self, j):
        if self.gw:
            rng = np.random.default_rng([self.seed, 15485863, j])
            gs = rng.choice(len(self.groups), self.micro, p=self.gp)
            return np.stack([self.gidx[g][rng.integers(len(self.gidx[g]))] for g in gs])
        e, r = divmod(j, self.per_epoch)
        if e not in self._cache:
            self._cache = {e: epoch_order(self.n, self.seed, e)}
        o = self._cache[e][r * self.micro:(r + 1) * self.micro]
        return self.idx[o]

    def step_micros(self, s):
        return list(range(s * self.accum, min((s + 1) * self.accum, self.total_micro)))


def motion_kernel(rng, k):
    ker = np.zeros((k, k), np.float32); c = k // 2
    ang = rng.uniform(0, np.pi)
    for t in np.linspace(-c, c, 4 * k):
        y, x = int(round(c + t * np.sin(ang))), int(round(c + t * np.cos(ang)))
        ker[y, x] = 1
    return ker / ker.sum()


def augment_tile(d, y, x, rng, A):
    """-> (img (24, 64, 64) float32 /255, lab (64, 64) float32, keep (64, 64) float32, wmap (64, 64) float32 or None)"""
    img, lab, keep, wm = d['img'], d['lab'], d['keep'], d.get('weight_map')
    if rng.random() < A['ssr_p']:
        cy, cx = y + TILE // 2, x + TILE // 2; h = CTX // 2
        sl = (slice(cy - h, cy + h), slice(cx - h, cx + h))
        ang = rng.uniform(-A['rotate'], A['rotate']); sc = 1.0 + rng.uniform(-A['scale'], A['scale'])
        sh = rng.uniform(-A['shift'], A['shift'], 2) * TILE
        M = cv2.getRotationMatrix2D((h - 0.5, h - 0.5), ang, sc)
        M[0, 2] += (TILE / 2 - h) + sh[0]; M[1, 2] += (TILE / 2 - h) + sh[1]
        ti = cv2.warpAffine(img[sl], M, (TILE, TILE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        tl = cv2.warpAffine(lab[sl], M, (TILE, TILE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        tk = cv2.warpAffine(keep[sl], M, (TILE, TILE), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        tw = None if wm is None else cv2.warpAffine(wm[sl], M, (TILE, TILE), flags=cv2.INTER_LINEAR, borderValue=1)
    else:
        ti = img[y:y + TILE, x:x + TILE]; tl = lab[y:y + TILE, x:x + TILE]; tk = keep[y:y + TILE, x:x + TILE]
        tw = None if wm is None else wm[y:y + TILE, x:x + TILE]
    ti = np.ascontiguousarray(ti); tl = np.ascontiguousarray(tl, dtype=np.float32); tk = tk.astype(np.float32)
    if rng.random() < 0.5: ti, tl, tk = ti[:, ::-1], tl[:, ::-1], tk[:, ::-1]; tw = None if tw is None else tw[:, ::-1]
    if rng.random() < 0.5: ti, tl, tk = ti[::-1], tl[::-1], tk[::-1]; tw = None if tw is None else tw[::-1]
    ti = np.ascontiguousarray(ti)
    if rng.random() < A['blur_p']:
        k = int(rng.choice([3, 5, 7]))
        if rng.random() < 0.5: ti = cv2.GaussianBlur(ti, (k, k), 0)
        else: ti = cv2.filter2D(ti, -1, motion_kernel(rng, k))
    tl = np.ascontiguousarray(tl)
    if rng.random() < A['dropout_p']:
        ti = ti.copy(); tl = tl.copy()
        for _ in range(A['dropout_holes']):
            hh = ww = A['dropout_px']
            yy, xx = int(rng.integers(0, TILE - hh + 1)), int(rng.integers(0, TILE - ww + 1))
            ti[yy:yy + hh, xx:xx + ww] = 0; tl[yy:yy + hh, xx:xx + ww] = 0
    a = ti.transpose(2, 0, 1).astype(np.float32) / 255.0                   # (24, 64, 64)
    if rng.random() < A['fourth_p']:                                        # GP "fourth augment" (depth jitter + layer cutout)
        k = int(rng.integers(int(DEPTH * 0.9), DEPTH + 1)); s = int(rng.integers(0, DEPTH - k + 1)); o = int(rng.integers(0, DEPTH - k + 1))
        b = np.zeros_like(a); b[o:o + k] = a[s:s + k]
        if rng.random() < A['fourth_cut_p']:
            tmp = np.arange(o, o + k); rng.shuffle(tmp); b[tmp[:int(rng.integers(0, 3))]] = 0
        a = b
    return a, tl, np.ascontiguousarray(tk), (None if tw is None else np.ascontiguousarray(tw, dtype=np.float32))


AUG_DEFAULT = dict(ssr_p=0.75, rotate=360.0, scale=0.1, shift=0.15, blur_p=0.4, dropout_p=0.5, dropout_holes=2, dropout_px=12,
                   fourth_p=0.6, fourth_cut_p=0.6)


def make_micro(sources, pairs, seed, j, A, pool=None):
    """micro-batch j: pairs (n, 2) of (source index, tile index) -> tensors x (n, 24, 64, 64), y, keep, w (n, 1, 64, 64)"""
    def one(t):
        i, (si, ti) = t
        rng = np.random.default_rng([seed, 104729, j, i])
        d = sources[si]; y, x = d['org'][ti]
        return augment_tile(d, int(y), int(x), rng, A)
    res = list(pool.map(one, enumerate(pairs))) if pool else [one(t) for t in enumerate(pairs)]
    x = torch.from_numpy(np.stack([r[0] for r in res]))
    y = torch.from_numpy(np.stack([r[1] for r in res]))[:, None]
    k = torch.from_numpy(np.stack([r[2] for r in res]))[:, None]
    w = None if res[0][3] is None else torch.from_numpy(np.stack([r[3] for r in res]))[:, None]
    return x, y, k, w


# ------------------------------------------------------------------------------------------------------------- loss / schedule
def loss_fn(logits, y, keep, wmap=None, smooth=0.25, w_dice=0.5, w_bce=0.5):
    """smp SoftBCEWithLogits(smooth_factor) + smp DiceLoss(binary, from_logits) on keep pixels; wmap (optional) weights BCE px."""
    ys = (1 - y) * smooth + y * (1 - smooth)
    bce = F.binary_cross_entropy_with_logits(logits, ys, reduction='none')
    wk = keep if wmap is None else keep * wmap
    bce = (bce * wk).sum() / wk.sum().clamp_min(1)
    p = torch.sigmoid(logits) * keep; yv = y * keep
    if wmap is not None: p, yv = p * wmap, yv * wmap
    inter = (p * yv).sum(); card = p.sum() + yv.sum()
    dice = 1 - (2 * inter) / card.clamp_min(1e-7)
    if yv.sum() <= 0: dice = dice * 0
    return w_dice * dice + w_bce * bce, float(dice), float(bce)


def lr_at(step, steps, lr, lr_min, warm_frac=0.0):
    warm = int(warm_frac * steps)
    if warm and step < warm: return lr * (step + 1) / warm
    q = (step - warm) / max(1, steps - warm)
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * min(1.0, q)))


# ------------------------------------------------------------------------------------------------------------- model / ckpt
def load_base(init='base'):
    s = sha256(f'{BASE_DIR}/model.safetensors')
    if s != BASE_SHA: raise SystemExit(f'base v8-in sha256 mismatch {s}')
    m = InkDetector.from_pretrained(BASE_DIR)
    if init == 'loo':
        from safetensors.torch import load_file
        s2 = sha256(f'{LOO_DIR}/model.safetensors')
        if s2 != LOO_SHA: raise SystemExit(f'LOO sha256 mismatch {s2}')
        m.load_state_dict(load_file(f'{LOO_DIR}/model.safetensors'), strict=True)
    return m


MODEL_CONFIG = {"clip_max": 200, "encoder_depth": 50, "in_depth": 24, "input_upsample_depth": 96, "input_upsample_hw": 256,
                "reverse_layers": True, "stride": 21, "tile_size": 64}


def save_ckpt(model, outdir, meta):
    """outdir/{model.safetensors, config.json (InkDetector.from_pretrained), meta.json}; returns sha256 of the weights"""
    from safetensors.torch import save_file
    os.makedirs(outdir, exist_ok=True)
    sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(sd, f'{outdir}/model.safetensors.tmp'); os.replace(f'{outdir}/model.safetensors.tmp', f'{outdir}/model.safetensors')
    json.dump(MODEL_CONFIG, open(f'{outdir}/config.json', 'w'), indent=2)
    s = sha256(f'{outdir}/model.safetensors')
    json.dump(dict(meta, sha256=s), open(f'{outdir}/meta.json', 'w'), indent=1)
    return s
