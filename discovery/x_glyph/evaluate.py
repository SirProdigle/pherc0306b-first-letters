"""Validate a glyph model on the held-out known-text sets and characterise it on the negative corpus.

usage: evaluate.py MODEL.pt [--tag NAME]
writes x_glyph/eval_<tag>.json and prints a summary.

In-segment protocols (50 um grid, 128 px = 6.4 mm windows, stride 16 px):
  A: pos = window centred on text (central 3.2 mm >= 10 % labelled), neg = no label within window + 0.5 mm
  B: pos as A, neg = central 3.2 mm has no label (text may touch the window edge)   [needed for 0841/1447]
Cross-corpus (triage-relevant): protocol-A positives vs every on-sheet window of the HELD-OUT negative patches
  of the matching model family (never trained on), plus recall at the negative-corpus p99 / p99.9 thresholds.
"Matched" AUC: each positive is paired with the negative of closest mean(a) -> does the score add anything beyond
  the amount of ink signal in the window?
"""
import sys, glob, json, argparse
import numpy as np, cv2, torch
from common import *
from evalset import blob_feats
import data

FAMILY = {'hec': 'fused_hd', 'hecTTA': 'fused_hd', 'dn': 'fused_hd', 'fusedHD': 'fused_hd', 's42': 'fused_ens3', 's43': 'fused_ens3',
          'ens3': 'fused_ens3', 'v8in': 'v8in', 'v8inPub': 'v8in'}


def win_stack(it, ws):
    A = np.stack([it['a'][y:y + WIN, x:x + WIN] for y, x in ws]).astype(np.float32)
    if A.max() > 1.5:
        A /= 255.0
    M = np.stack([it['mask'][y:y + WIN, x:x + WIN] for y, x in ws]).astype(np.float32)
    return A, M


@torch.no_grad()
def score_windows(net, it, ws, bs=128, tta=True):
    if not ws:
        return np.zeros(0, np.float32)
    A, M = win_stack(it, ws)
    out = []
    for i in range(0, len(A), bs):
        a = torch.from_numpy(A[i:i + bs, None]).cuda(); m = torch.from_numpy(M[i:i + bs, None]).cuda()
        with torch.autocast('cuda', dtype=torch.float16):
            s = _tta(net, a, m, tta)
        out.append(s.float().cpu())
    return torch.cat(out).numpy()


def _tta(net, a, m, tta):
    s = net(a, m)
    if tta:      # D4 subset: rot90 x3 + flip
        for k in (1, 2, 3):
            s = s + net(torch.rot90(a, k, (2, 3)), torch.rot90(m, k, (2, 3)))
        s = s + net(a.flip(3), m.flip(3))
        s = s / 5
    return s


def mean_a(it, ws):
    A, M = win_stack(it, ws)
    return (A * M).sum((1, 2)) / np.maximum(M.sum((1, 2)), 1)


def protocols(it, stride=16):
    m = it['mask'].copy()
    if 'evalmask' in it:
        m &= it['evalmask']
    lab = it['lab'] > 0.3
    labd = cv2.dilate(lab.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0
    im = cv2.integral(m.astype(np.uint8)); il = cv2.integral((lab & m).astype(np.uint8)); ild = cv2.integral(labd.astype(np.uint8))
    ilr = cv2.integral(lab.astype(np.uint8))
    S = lambda ii, y, x, h: ii[y + h, x + h] - ii[y, x + h] - ii[y + h, x] + ii[y, x]
    c, h2 = WIN // 4, WIN // 2
    P, NA, NB = [], [], []
    for y, x in windows(m.shape, WIN, stride):
        if S(im, y, x, WIN) < 0.7 * WIN * WIN:
            continue
        on_c = S(im, y + c, x + c, h2)
        if on_c > 0.5 * h2 * h2 and S(il, y + c, x + c, h2) / on_c >= 0.10:
            P.append((y, x))
        else:
            if S(ild, y, x, WIN) == 0:
                NA.append((y, x))
            if S(ilr, y + c, x + c, h2) == 0:
                NB.append((y, x))
    return P, NA, NB


def matched_auc(sp, ap, sn, an, rng=np.random.default_rng(0)):
    if len(sp) == 0 or len(sn) == 0:
        return float('nan')
    o = np.argsort(an); an_s = an[o]; sn_s = sn[o]
    idx = np.clip(np.searchsorted(an_s, ap), 0, len(an_s) - 1)
    idx2 = np.clip(idx - 1, 0, len(an_s) - 1)
    pick = np.where(np.abs(an_s[idx2] - ap) < np.abs(an_s[idx] - ap), idx2, idx)
    return auc(sp, sn_s[pick])


def corpus_items(cat, srcs=('s2', 'v8', 'zoom', 'deep', 'val343hd')):
    out = []
    for src in srcs:
        for f in sorted(glob.glob(f'{G}/cache/corpus/{src}/*.npz')):
            d = np.load(f)
            if str(d['cat']) != cat:
                continue
            it = {k: d[k] for k in d.files}
            for k in list(it):
                if it[k].ndim == 0:
                    it[k] = it[k].item()
            it['path'] = f
            out.append(it)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('model'); ap.add_argument('--tag', default=None); ap.add_argument('--no-tta', action='store_true')
    a = ap.parse_args()
    tag = a.tag or os.path.basename(a.model).replace('.pt', '')
    torch.cuda.set_per_process_memory_fraction(0.2)
    torch.set_num_threads(4)
    tta = not a.no_tta
    if a.model == 'legib':
        import legib
        net = legib.Legib()
        score_windows = lambda net_, it, ws, tta=True: net.score(it, ws)
    else:
        from train import load_model
        net = load_model(a.model)
        score_windows = globals()['score_windows']
    R = dict(model=a.model, heldout={}, cross={}, corpus={}, p343={}, hard={})

    # ---- negative corpus (held-out negative patches): per-family window scores
    neg = {}
    for it in corpus_items('heldneg'):
        ws = onsheet_windows(it, WIN, 16, 0.7)
        if not ws:
            continue
        s = score_windows(net, it, ws, tta=tta); ma = mean_a(it, ws)
        fam = it['model']
        neg.setdefault(fam, dict(s=[], ma=[], pmax=[], names=[]))
        neg[fam]['s'].append(s); neg[fam]['ma'].append(ma); neg[fam]['pmax'].append(float(s.max())); neg[fam]['names'].append(os.path.basename(it['path']))
    for fam, d in neg.items():
        d['s'] = np.concatenate(d['s']); d['ma'] = np.concatenate(d['ma'])
        R['corpus'][fam] = dict(n_windows=int(len(d['s'])), n_patches=len(d['pmax']),
                                pct={str(q): round(float(np.percentile(d['s'], q)), 3) for q in (50, 90, 99, 99.9)},
                                patch_max_pct={str(q): round(float(np.percentile(d['pmax'], q)), 3) for q in (50, 90, 99)},
                                corr_with_mean_a=round(float(np.corrcoef(d['s'], d['ma'])[0, 1]), 3))
        print('corpus', fam, json.dumps(R['corpus'][fam]), flush=True)
    allneg = np.concatenate([d['s'] for d in neg.values()])
    R['corpus']['all'] = dict(n_windows=int(len(allneg)), pct={str(q): round(float(np.percentile(allneg, q)), 3) for q in (50, 90, 99, 99.9)})

    # ---- held-out known text
    for p in sorted(glob.glob(f'{G}/cache/heldout/*.npz')):
        it = load_item(p); name = it['name']
        P, NA, NB = protocols(it)
        sP, sA, sB = (score_windows(net, it, w, tta=tta) for w in (P, NA, NB))
        mP, mA, mB = (mean_a(it, w) if w else np.zeros(0) for w in (P, NA, NB))
        r = dict(npos=len(P), nnegA=len(NA), nnegB=len(NB),
                 cnn_A=round(auc(sP, sA), 3), base_A=round(auc(mP, mA), 3), cnn_A_matched=round(matched_auc(sP, mP, sA, mA), 3),
                 cnn_B=round(auc(sP, sB), 3), base_B=round(auc(mP, mB), 3), cnn_B_matched=round(matched_auc(sP, mP, sB, mB), 3),
                 pos_median=round(float(np.median(sP)), 3) if len(sP) else None)
        fam = FAMILY.get(it['model'])
        if fam in neg:
            d = neg[fam]
            t99, t999 = np.percentile(d['s'], 99), np.percentile(d['s'], 99.9)
            b99 = np.percentile(d['ma'], 99)
            R['cross'][name] = dict(family=fam, cnn=round(auc(sP, d['s']), 3), base_mean_a=round(auc(mP, d['ma']), 3),
                                    cnn_matched=round(matched_auc(sP, mP, d['s'], d['ma']), 3),
                                    recall_at_p99=round(float((sP > t99).mean()), 3), recall_at_p999=round(float((sP > t999).mean()), 3),
                                    base_recall_at_p99=round(float((mP > b99).mean()), 3),
                                    negA_above_p99=round(float((sA > t99).mean()), 3) if len(sA) else None,
                                    negB_above_p99=round(float((sB > t99).mean()), 3) if len(sB) else None)
            r['cross'] = R['cross'][name]
        R['heldout'][name] = r
        print(name, json.dumps(r), flush=True)

    # ---- PHerc0343 screening patches (probable text) and rejected hard negatives: per-patch max / top5
    allmax = np.concatenate([np.array(d['pmax']) for d in neg.values()])
    for cat, key in (('p343', 'p343'), ('hard', 'hard')):
        for it in corpus_items(cat):
            ws = onsheet_windows(it, WIN, 16, 0.7)
            if not ws:
                continue
            s = score_windows(net, it, ws, tta=tta)
            top = np.sort(s)[::-1]
            fam = it['model'] if it['model'] in neg else 'fused_hd'
            R[key][os.path.basename(it['path'])] = dict(max=round(float(top[0]), 3), top5=round(float(top[:5].mean()), 3),
                                                        frac_above_p99=round(float((s > np.percentile(neg[fam]['s'], 99)).mean()), 4),
                                                        max_rank_vs_heldneg_patches=round(float((allmax < top[0]).mean()), 3))
        print(key, json.dumps(R[key], indent=0)[:3000], flush=True)
    json.dump(R, open(f'{G}/eval_{tag}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
