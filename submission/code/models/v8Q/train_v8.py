"""train_v8.py ARM [--steps 850] [--accum 16] [--micro 2] [--lr 1e-5] [--max-minutes 25] [--bench]

Fine-tune v8-in (init = released base weights, the team's init "v8-in epoch 6") the organisers' way: >= 2 labelled
windings of the 116 keV / 8.64 um protocol, a held-out set never seen (0343 tracings; lead A is never seen either).
ARMS
  v8E : PHerc1447 w058 + w060 (HF layers, model order = reversed), refined labels ERODED (27 px ellipse, O1 arm E).
  v8O : same, original labels (a re-run of the team's LOO-w062 recipe with this loop; only if needed as a control).
  v8P : v8E + the pick-2 0306B tracing (x_followup/p02_label/, own patch fl/sqm/PHerc0306B_a306_row; only when it exists;
        tiles never overlap lead A's exclusion zone, which is on another patch).
Recipe (team's loo_w062.json where possible): 64 px tiles admitted only if fully inside the certainty mask, origins on an
8 px grid, windings sampled in proportion to their tiles; loss 0.5 Dice + 0.5 BCE with label smoothing 0.25; AdamW
(wd 1e-6, not on norms/biases), lr 1e-5 cosine to 1e-6 per step (5 % warm-up), grad clip 1, effective batch 32
(micro-batches, BatchNorm statistics FROZEN -- deviation: the team trained BN in train mode at batch 4), bf16 autocast;
augmentation: flips + rot90, the team's "fourth augment" depth jitter (p 0.6: 90-100 % of the 24 layers re-pasted at a
random depth offset; p 0.6 of that: up to 2 layers zeroed), coarse dropout (p 0.5, 2 holes <= 20 %), contrast +-10 %.
850 optimiser steps = 27,200 tiles (team: 850 steps x 32). Final checkpoint, no selection.
Resumable: ckpt/<ARM>/last.pt; exits 3 after --max-minutes; writes ckpt/<ARM>/final.safetensors + done.json (with sha256).
"""
import argparse, json, math, os, sys, time
import numpy as np, cv2, tifffile, torch, torch.nn.functional as F
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v8lib as L                                                                 # noqa: E402
from ink8um import InkDetector                                                    # noqa: E402

SRC = {'w058': f'{L.W}/data/pherc1447/w058', 'w060': f'{L.W}/x_refine/trials/O1_overlabel/data/hf/w060'}
ARMS = {'v8E': (['w058', 'w060'], 27), 'v8O': (['w058', 'w060'], 0),
        # P-4 cross-lead arms (Q1 proposal 2 / Q2 P-4; groups sampled 1:1:1): final, F1 = 0343 held out, F2 = w060 held out
        'v8P': ([['w058', 'w060'], ['0343'], ['p02']], 27), 'v8P_F1': ([['w058', 'w060'], ['p02']], 27),
        'v8P_F2': ([['w058'], ['0343'], ['p02']], 27),
        # FINDER arms (PREREG amendment 1): lead A tracing T2 on one side of N6's split (or all), + 1447 w058 + w060 at 1:1
        'fL': (['w058', 'w060', 'A:L'], 0), 'fR': (['w058', 'w060', 'A:R'], 0), 'fF': (['w058', 'w060', 'A:F'], 0),
        'fALL': (['w058', 'w060', 'A:ALL'], 0),
        # P-4a (PREREG amendment 9): fv8's recipe without any lead-A data; w060 HELD OUT for no-harm (no other 1447 set is local)
        'v8Q': ([['w058'], ['0343']], 0), 'v8S': ([['w058'], ['0343'], ['p02s']], 0), 'v8S_F': ([['w058'], ['0343'], ['p02sF']], 0),
        # amendment 10: v8T = v8Q's data with positives ERODED to the stroke cores (band between core and label ignored)
        'v8T': ([['w058c'], ['0343c']], 0)}
DEFAULT_STEPS = {'fL': 300, 'fR': 300, 'fF': 300, 'fALL': 300, 'v8Q': 300, 'v8S': 300, 'v8S_F': 300, 'v8T': 300}
CORE_R_MM = 0.12                                                                  # amendment 10: erosion radius of the stroke cores


def core_labels(d, px_um=8.64):
    """amendment 10 (v8T): target = label eroded by CORE_R_MM (ellipse, radius r px) U the label's skeleton (>= 1 px of every
    stroke kept); the band label \\ core is IGNORED (keep = 0); unlabelled px stay negatives as in v8Q.  Both 1447 and 0343 are
    8.64 um, so r = 14 px."""
    from skimage.morphology import skeletonize
    r = int(round(CORE_R_MM * 1000 / px_um)); lab = d['tgt'] > 0.5
    core = (cv2.erode(lab.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0) | skeletonize(lab)
    d = dict(d); d['tgt'] = core.astype(np.float32); d['keep'] = ~(lab & ~core)
    d['info'] = dict(d['info'], core_r_px=r, core_frac_of_label=round(float(core.sum() / max(lab.sum(), 1)), 4))
    return d


def a_finder(side):
    """lead A tiles for the finder: (24, h, w) render layers 8..31 (asc = A's reading order) over the labelled bbox."""
    import zarr
    z = np.load(f'{L.T}/finder/labels.npz')
    pos, neg, valid = z['pos'], z['neg'], z['valid']
    if side == 'F':
        dp, di = z['decoy_pos'], z['decoy_ign']; pos = pos | dp; neg = neg & ~(dp | di)
    sm = z['left'] if side in ('L', 'F') else (z['right'] if side == 'R' else np.ones_like(valid))
    pos, neg = pos & sm, neg & sm
    lab = pos | neg
    ys, xs = np.nonzero(lab); y0, y1, x0, x1 = max(0, ys.min() - 64), min(5640, ys.max() + 65), max(0, xs.min() - 64), min(5640, xs.max() + 65)
    st = np.asarray(zarr.open(f'{L.W}/x_focus306/renders/w074own.zarr', mode='r')['0'][8:32, y0:y1, x0:x1])
    st = np.minimum(st, 200).astype(np.uint8)
    sl = (slice(y0, y1), slice(x0, x1))
    ok_px = sm[sl] & valid[sl] & (st > 0).all(0)
    ii, il, ip = integral(ok_px), integral(lab[sl]), integral(pos[sl])
    H, Wd = ok_px.shape
    Y, X = np.meshgrid(np.arange(0, H - 63, 8), np.arange(0, Wd - 63, 8), indexing='ij'); Y, X = Y.ravel(), X.ravel()
    bs = lambda I: I[Y + 64, X + 64] - I[Y, X + 64] - I[Y + 64, X] + I[Y, X]           # noqa: E731
    ok = bs(ii) == 64 * 64
    nl, npos = bs(il), bs(ip)
    any_ = np.stack([Y, X], 1)[ok & (nl >= 0.01 * 4096)]; ptile = np.stack([Y, X], 1)[ok & (npos >= 0.05 * 4096)]
    return dict(st=st, tgt=pos[sl].astype(np.float32), keep=lab[sl], org=any_, porg=ptile, info=dict(
        winding=f'PHerc0306B lead A tracing T2 side {side}', crop=[int(y0), int(y1), int(x0), int(x1)], tiles=int(len(any_)),
        pos_tiles=int(len(ptile)), pos_px=int(pos.sum()), neg_px=int(neg.sum())))


def integral(m):
    return np.pad(m.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))


def winding(w, erode):
    lab = cv2.imread(f'{SRC[w]}/labels/inklabels.png', 0)
    msk = cv2.imread(f'{SRC[w]}/labels/mask.png', 0) > 0
    ys, xs = np.nonzero(msk); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    st = np.stack([np.minimum(tifffile.imread(f'{SRC[w]}/layers/{i:02d}.tif')[y0:y1, x0:x1], 200).astype(np.uint8)
                   for i in range(23, -1, -1)], axis=0)                             # (24, h, w), model order (reversed)
    lab, msk = lab[y0:y1, x0:x1], msk[y0:y1, x0:x1]
    tgt = (lab > 127)
    if erode: tgt = cv2.erode(tgt.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode, erode))) > 0
    H, Wd = msk.shape; ii = integral(msk)
    Y, X = np.meshgrid(np.arange(0, H - 63, 8), np.arange(0, Wd - 63, 8), indexing='ij'); Y, X = Y.ravel(), X.ravel()
    ok = (ii[Y + 64, X + 64] - ii[Y, X + 64] - ii[Y + 64, X] + ii[Y, X]) == 64 * 64
    return dict(st=st, tgt=tgt.astype(np.float32), org=np.stack([Y[ok], X[ok]], 1), info=dict(
        winding=w, crop=[int(y0), int(y1), int(x0), int(x1)], tiles=int(ok.sum()), target_frac=round(float(tgt[msk].mean()), 4),
        label_frac=round(float((lab > 127)[msk].mean()), 4), erode_px=erode))


def src_0343(erode=0):
    """PHerc0343 (Erwin's segment, concat_66, c 33, reading order desc), jeff-j-chen tracings: supervised inside the tracer's
    train mask (unlabelled = background there), tiles fully inside it."""
    import zarr
    lab = np.load(f'{L.W}/x_px343/label_render.npy') > 0; msk = np.load(f'{L.W}/x_px343/mask_render.npy') > 0
    ys, xs = np.nonzero(msk); y0, y1, x0, x1 = max(0, ys.min() - 8), ys.max() + 9, max(0, xs.min() - 8), xs.max() + 9
    st = np.asarray(zarr.open(f'{L.W}/fl/0343/concat_66.zarr', mode='r')['0'][33 - 12:33 + 12, y0:y1, x0:x1])[::-1]   # desc
    st = np.ascontiguousarray(np.minimum(st, 200)).astype(np.uint8)
    lab, msk = lab[y0:y1, x0:x1], msk[y0:y1, x0:x1] & (st > 0).all(0)
    tgt = cv2.erode(lab.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode, erode))) > 0 if erode else lab
    ii = integral(msk); H, Wd = msk.shape
    Y, X = np.meshgrid(np.arange(0, H - 63, 8), np.arange(0, Wd - 63, 8), indexing='ij'); Y, X = Y.ravel(), X.ravel()
    ok = (ii[Y + 64, X + 64] - ii[Y, X + 64] - ii[Y + 64, X] + ii[Y, X]) == 64 * 64
    return dict(st=st, tgt=tgt.astype(np.float32), org=np.stack([Y[ok], X[ok]], 1), info=dict(
        winding='PHerc0343 jeff-j-chen tracings', crop=[int(y0), int(y1), int(x0), int(x1)], tiles=int(ok.sum()),
        target_frac=round(float(tgt[msk].mean()), 4), erode_px=erode))


P02_DIR = os.environ.get('P02_DIR', f'{L.W}/x_followup/p02_label')        # env override only for dry runs


def p02_label():
    """the papyrologist's pick-2 tracing on the pick-2 render grid (x_followup/B_depth/render.zarr).  Accepted: label_render.npy
    (bool, render frame) or *.png + transform.json {y0, x0, scale} (png px -> render px = (x0 + x*scale, y0 + y*scale))."""
    import glob, zarr
    H, Wd = zarr.open(f'{L.W}/x_followup/B_depth/render.zarr', mode='r')['0'].shape[1:]
    if os.path.exists(f'{P02_DIR}/label_render.npy'): return np.load(f'{P02_DIR}/label_render.npy') > 0
    pngs = sorted(glob.glob(f'{P02_DIR}/*.png'))
    if pngs and os.path.exists(f'{P02_DIR}/transform.json'):
        t = json.load(open(f'{P02_DIR}/transform.json'))
        im = cv2.imread(pngs[-1], cv2.IMREAD_UNCHANGED); im = (im[..., :3].max(-1) if im.ndim == 3 else im) > 127
        bb = cv2.resize(im.astype(np.uint8), (int(im.shape[1] * t['scale']), int(im.shape[0] * t['scale'])), interpolation=cv2.INTER_NEAREST) > 0
        out = np.zeros((H, Wd), bool); y0, x0 = int(t['y0']), int(t['x0'])
        out[y0:y0 + bb.shape[0], x0:x0 + bb.shape[1]] = bb[:H - y0, :Wd - x0]; return out
    raise SystemExit(f'pick-2 tracing not usable yet: need {P02_DIR}/label_render.npy or a png + transform.json')


def p02_data(erode=0, ring=25, near_mm=1.5):
    """pick-2 own patch render (c 20, asc = training orientation, eval/depth_profile.json['pick2_rect']).  pos = tracing;
    ignore = a ring of `ring` px (outline-level tracing); negatives = px within near_mm of the tracing that are base-cold is a
    PRE-REGISTRATION choice made when the tracing lands (default here: within near_mm of the strokes, not pos / ring); every
    training tile must lie outside p4/exclusion.npz['excl_25_render'] (pick-2 cells < 2.5 mm in 3-D from A's claim area)."""
    import zarr
    pos = p02_label(); ex = np.load(f'{L.T}/p4/exclusion.npz')['excl_25_render']
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring + 1, 2 * ring + 1))
    ign = (cv2.dilate(pos.astype(np.uint8), ker) > 0) & ~pos
    near = cv2.distanceTransform((~pos).astype(np.uint8), cv2.DIST_L2, 5) <= near_mm * 1000 / 8.64
    neg = near & ~pos & ~ign
    tgt = cv2.erode(pos.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode, erode))) > 0 if erode else pos
    lab = pos | neg
    ys, xs = np.nonzero(lab); y0, y1, x0, x1 = max(0, ys.min() - 64), ys.max() + 65, max(0, xs.min() - 64), xs.max() + 65
    st = np.asarray(zarr.open(f'{L.W}/x_followup/B_depth/render.zarr', mode='r')['0'][8:32, y0:y1, x0:x1])
    st = np.minimum(st, 200).astype(np.uint8); sl = (slice(y0, y1), slice(x0, x1))
    okpx = (st > 0).all(0) & ~ex[sl]
    il, ip, io = integral(lab[sl]), integral(tgt[sl]), integral(okpx); H, Wd = okpx.shape
    Y, X = np.meshgrid(np.arange(0, H - 63, 8), np.arange(0, Wd - 63, 8), indexing='ij'); Y, X = Y.ravel(), X.ravel()
    bs = lambda I: I[Y + 64, X + 64] - I[Y, X + 64] - I[Y + 64, X] + I[Y, X]           # noqa: E731
    ok = bs(io) == 64 * 64
    any_ = np.stack([Y, X], 1)[ok & (bs(il) >= 0.01 * 4096)]; ptile = np.stack([Y, X], 1)[ok & (bs(ip) >= 0.05 * 4096)]
    return dict(st=st, tgt=tgt[sl].astype(np.float32), keep=lab[sl], org=any_, porg=ptile, info=dict(
        winding='PHerc0306B pick 2 (w050 row) papyrologist tracing', crop=[int(y0), int(y1), int(x0), int(x1)], tiles=int(len(any_)),
        pos_tiles=int(len(ptile)), pos_px=int(tgt.sum()), neg_px=int(neg.sum()), excluded_px_in_crop=int(ex[sl].sum())))


def p02s_data(decoy=False):
    """amendment 9 (v8S): pick-2 PSEUDO-labels from p4/v8s_labels.npz (p4a_labels.py; pick 2's own Rv2 + TRACE maps, no tracing):
    pos = fused z > 2.5 in the main-row band (eroded 2 px), neg = z < 0 in RECT + 2 mm, rest ignored (loss-masked).  decoy=True
    (v8S_F) adds the translated decoy positives (their 25 px ring ignored).  Render x_followup/B_depth/render.zarr layers 8..31
    (c 20, d0, asc); every tile lies fully on CT and fully outside p4/exclusion.npz['excl_25_render']."""
    import zarr
    z = np.load(f'{L.T}/p4/v8s_labels.npz'); cy, cx = int(z['y0']), int(z['x0'])
    pos, neg, cov = z['pos'], z['neg'], z['cov']
    if decoy: pos = pos | z['dpos']; neg = neg & ~(z['dpos'] | z['dign'])
    Hc, Wc = pos.shape
    ex = np.load(f'{L.T}/p4/exclusion.npz')['excl_25_render'][cy:cy + Hc, cx:cx + Wc]
    lab = pos | neg
    ys, xs = np.nonzero(lab); y0, y1, x0, x1 = max(0, ys.min() - 64), min(Hc, ys.max() + 65), max(0, xs.min() - 64), min(Wc, xs.max() + 65)
    st = np.asarray(zarr.open(f'{L.W}/x_followup/B_depth/render.zarr', mode='r')['0'][8:32, cy + y0:cy + y1, cx + x0:cx + x1])
    st = np.minimum(st, 200).astype(np.uint8); sl = (slice(y0, y1), slice(x0, x1))
    okpx = cov[sl] & (st > 0).all(0) & ~ex[sl]
    il, ip, io = integral(lab[sl]), integral(pos[sl]), integral(okpx); H, Wd = okpx.shape
    Y, X = np.meshgrid(np.arange(0, H - 63, 8), np.arange(0, Wd - 63, 8), indexing='ij'); Y, X = Y.ravel(), X.ravel()
    bs = lambda I: I[Y + 64, X + 64] - I[Y, X + 64] - I[Y + 64, X] + I[Y, X]           # noqa: E731
    ok = bs(io) == 64 * 64
    any_ = np.stack([Y, X], 1)[ok & (bs(il) >= 0.01 * 4096)]; ptile = np.stack([Y, X], 1)[ok & (bs(ip) >= 0.05 * 4096)]
    return dict(st=st, tgt=pos[sl].astype(np.float32), keep=lab[sl], org=any_, porg=ptile, info=dict(
        winding='PHerc0306B pick 2 (w050 row) PSEUDO-labels' + (' + decoy' if decoy else ''), crop=[int(cy + y0), int(cy + y1), int(cx + x0), int(cx + x1)],
        tiles=int(len(any_)), pos_tiles=int(len(ptile)), pos_px=int(pos[sl].sum()), neg_px=int(neg[sl].sum()),
        excluded_px_in_crop=int(ex[sl].sum())))


def sample(step, k, seed, data, n, a_data=None):
    """n tiles; with a_data (finder arms) every second tile is a lead-A tile (1:1), alternating positive-centred / any."""
    rng = np.random.default_rng(seed * 1_000_003 + step * 64 + k)
    sizes = np.array([len(d['org']) for d in data], np.float64)
    xs, ys, ms = [], [], []
    for i in range(n):
        if a_data is not None and i % 2 == 1:
            d = a_data; org = d['porg'] if (k % 2 == 0 and len(d['porg'])) else d['org']
            y, x = org[rng.integers(len(org))]
            m = d['keep'][y:y + 64, x:x + 64].astype(np.float32)
        else:
            d = data[rng.choice(len(data), p=sizes / sizes.sum())]
            y, x = d['org'][rng.integers(len(d['org']))]
            m = np.ones((64, 64), np.float32)
        a = d['st'][:, y:y + 64, x:x + 64].astype(np.float32) / 255.0; t = d['tgt'][y:y + 64, x:x + 64]
        r = rng.integers(4); a, t, m = np.rot90(a, r, (1, 2)), np.rot90(t, r), np.rot90(m, r)
        if rng.random() < 0.5: a, t, m = a[:, :, ::-1], t[:, ::-1], m[:, ::-1]
        if rng.random() < 0.6:                                                       # depth jitter ("fourth augment")
            k_ = int(rng.integers(22, 25)); s = int(rng.integers(0, 24 - k_ + 1)); o = int(rng.integers(0, 24 - k_ + 1))
            b = np.zeros_like(a); b[o:o + k_] = a[s:s + k_]
            if rng.random() < 0.6:
                for z in rng.choice(24, int(rng.integers(1, 3)), replace=False): b[z] = 0
            a = b
        if rng.random() < 0.5:                                                       # coarse dropout
            for _h in range(2):
                hh, ww = int(rng.integers(1, 13)), int(rng.integers(1, 13)); yy, xx = int(rng.integers(0, 64 - hh)), int(rng.integers(0, 64 - ww))
                a = a.copy(); a[:, yy:yy + hh, xx:xx + ww] = 0
        a = np.clip(a * rng.uniform(0.9, 1.1), 0, 200 / 255)
        xs.append(np.ascontiguousarray(a)); ys.append(np.ascontiguousarray(t)); ms.append(np.ascontiguousarray(m))
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))[:, None], torch.from_numpy(np.stack(ms))[:, None]


def sample_groups(step, k, seed, groups, n):
    """P-4 / v8P: fixed per-source quotas -- tile (k * n + i) comes from group (k * n + i) % len(groups); inside a group sources
    are drawn in proportion to their tiles; sources with 'porg' alternate positive-centred / any tiles.  Same augmentation as
    sample().  (The finder arms keep sample() unchanged.)"""
    rng = np.random.default_rng(seed * 1_000_003 + step * 64 + k)
    xs, ys, ms = [], [], []
    for i in range(n):
        g = groups[(k * n + i) % len(groups)]
        sz = np.array([len(d['org']) for d in g], np.float64); d = g[rng.choice(len(g), p=sz / sz.sum())]
        org = d['porg'] if ('porg' in d and len(d['porg']) and rng.random() < 0.5) else d['org']
        y, x = org[rng.integers(len(org))]
        m = d['keep'][y:y + 64, x:x + 64].astype(np.float32) if 'keep' in d else np.ones((64, 64), np.float32)
        a = d['st'][:, y:y + 64, x:x + 64].astype(np.float32) / 255.0; t = d['tgt'][y:y + 64, x:x + 64]
        r = rng.integers(4); a, t, m = np.rot90(a, r, (1, 2)), np.rot90(t, r), np.rot90(m, r)
        if rng.random() < 0.5: a, t, m = a[:, :, ::-1], t[:, ::-1], m[:, ::-1]
        if rng.random() < 0.6:
            k_ = int(rng.integers(22, 25)); s = int(rng.integers(0, 24 - k_ + 1)); o = int(rng.integers(0, 24 - k_ + 1))
            b = np.zeros_like(a); b[o:o + k_] = a[s:s + k_]
            if rng.random() < 0.6:
                for z in rng.choice(24, int(rng.integers(1, 3)), replace=False): b[z] = 0
            a = b
        if rng.random() < 0.5:
            for _h in range(2):
                hh, ww = int(rng.integers(1, 13)), int(rng.integers(1, 13)); yy, xx = int(rng.integers(0, 64 - hh)), int(rng.integers(0, 64 - ww))
                a = a.copy(); a[:, yy:yy + hh, xx:xx + ww] = 0
        if CONTRAST_IN_GROUPS: a = np.clip(a * rng.uniform(0.9, 1.1), 0, 200 / 255)     # amendment 9: parity with sample() (fv8)
        xs.append(np.ascontiguousarray(a)); ys.append(np.ascontiguousarray(t)); ms.append(np.ascontiguousarray(m))
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))[:, None], torch.from_numpy(np.stack(ms))[:, None]


CONTRAST_IN_GROUPS = True


def loss_fn(logits, y, m, smooth=0.25):
    ys = y * (1 - smooth) + smooth / 2
    bce = F.binary_cross_entropy_with_logits(logits, ys, reduction='none')
    bce = (bce * m).sum() / m.sum().clamp_min(1)
    p = torch.sigmoid(logits) * m; yv = y * m
    dice = 1 - (2 * (p * yv).sum() + 1) / (p.sum() + yv.sum() + 1)
    return 0.5 * bce + 0.5 * dice


def lr_at(step, steps, lr, lr_min=1e-6):
    warm = max(1, int(0.05 * steps))
    if step < warm: return lr * (step + 1) / warm
    q = (step - warm) / max(1, steps - warm)
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * q))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('arm', choices=list(ARMS)); ap.add_argument('--steps', type=int, default=None)
    ap.add_argument("--accum", type=int, default=32); ap.add_argument("--micro", type=int, default=1)
    ap.add_argument('--lr', type=float, default=1e-5); ap.add_argument('--seed', type=int, default=130697)
    ap.add_argument('--max-minutes', type=float, default=25); ap.add_argument('--bench', action='store_true')
    a = ap.parse_args()
    out = f'{L.T}/ckpt/{a.arm}'; os.makedirs(out, exist_ok=True)
    if os.path.exists(f'{out}/done.json') and not a.bench: print('already done', out); sys.exit(0)
    if a.arm == 'fALL' and not a.bench:                                             # amendment 1 / 9: only for a PASSED finder
        fe = json.load(open(f'{L.T}/eval/fe.json')) if os.path.exists(f'{L.T}/eval/fe.json') else {}
        if fe.get('wfL_passed') is not True: print('gate fe closed: no fALL', flush=True); sys.exit(0)
    t0 = time.time()
    if a.steps is None: a.steps = DEFAULT_STEPS.get(a.arm, 850)
    wins, erode = ARMS[a.arm]
    groups = None
    if isinstance(wins[0], list):                                                   # grouped arms (v8P*, v8Q, v8S*)
        mk = lambda w: (core_labels(winding(w[:-1], 0)) if w == 'w058c' else core_labels(src_0343(0)) if w == '0343c'   # noqa: E731
                        else winding(w, erode) if w.startswith('w0') else src_0343(0) if w == '0343' else p02s_data(False) if w == 'p02s'
                        else p02s_data(True) if w == 'p02sF' else p02_data(0))
        groups = [[mk(w) for w in g] for g in wins]; data = [d for g in groups for d in g]; a_data = None
    else:
        data = [winding(w, erode) for w in wins if not w.startswith('A:')]
        a_data = next((a_finder(w[2:]) for w in wins if w.startswith('A:')), None)
    info = [d['info'] for d in data] + ([a_data['info']] if a_data else [])
    print(a.arm, json.dumps(info), f'({time.time() - t0:.0f}s)', flush=True)
    dev = 'cuda'
    model = InkDetector.from_pretrained(L.PATHS['base']).to(dev).train()
    for mod in model.modules():
        if isinstance(mod, (torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)): mod.eval()      # frozen running statistics
    decay = [p for n, p in model.named_parameters() if p.ndim > 1]
    no_decay = [p for n, p in model.named_parameters() if p.ndim <= 1]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': 1e-6}, {'params': no_decay, 'weight_decay': 0.0}], lr=a.lr)
    step, log = 0, []
    if os.path.exists(f'{out}/last.pt') and not a.bench:
        ck = torch.load(f'{out}/last.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); step = int(ck['step'])
        log = json.load(open(f'{out}/log.json')) if os.path.exists(f'{out}/log.json') else []
        print(f'resumed at step {step}', flush=True)

    def save():
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'step': step}, f'{out}/last.pt.tmp')
        os.replace(f'{out}/last.pt.tmp', f'{out}/last.pt'); json.dump(log, open(f'{out}/log.json', 'w'))

    chunk_t, last_save, run_l = time.time(), time.time(), []
    torch.cuda.reset_peak_memory_stats()
    while step < a.steps:
        for g in opt.param_groups: g['lr'] = lr_at(step, a.steps, a.lr)
        opt.zero_grad(set_to_none=True); tot = 0.0
        for k in range(a.accum):
            x, y, m = sample_groups(step, k, a.seed, groups, a.micro) if groups else sample(step, k, a.seed, data, a.micro, a_data)
            x, y, m = x.to(dev, non_blocking=True), y.to(dev), m.to(dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lg = model(x)
            loss = loss_fn(lg.float(), y, m); (loss / a.accum).backward(); tot += loss.item() / a.accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        step += 1; run_l.append(tot)
        if step % 10 == 0 or a.bench:
            rec = dict(step=step, loss=round(float(np.mean(run_l)), 4), lr=opt.param_groups[0]['lr']); log.append(rec); run_l = []
            print(f'{a.arm} step {step}/{a.steps} loss {rec["loss"]:.4f} lr {rec["lr"]:.2e} {(time.time() - chunk_t) / 60:.1f} min '
                  f'peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB', flush=True)
        if a.bench and step >= 3:
            print(f'bench: {(time.time() - chunk_t) / step:.2f} s/step ({a.accum * a.micro} tiles)'); sys.exit(0)
        if time.time() - last_save > 600: save(); last_save = time.time()
        if time.time() - chunk_t > a.max_minutes * 60 and step < a.steps:
            save(); print(f'chunk end at step {step}', flush=True); sys.exit(3)
    save(); model.eval()
    fin = f'{out}/final.safetensors'
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, fin)
    json.dump(dict(arm=a.arm, data=info, steps=a.steps, effective_batch=a.accum * a.micro, lr=a.lr, seed=a.seed,
                   bn='frozen running statistics', sha256=L.sha256(fin), init_sha256=L.sha256(f'{L.PATHS["base"]}/model.safetensors')),
              open(f'{out}/done.json', 'w'), indent=1)
    os.remove(f'{out}/last.pt'); print('done', out, flush=True)
