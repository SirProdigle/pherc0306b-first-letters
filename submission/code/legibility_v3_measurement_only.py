"""legib.py : agreement with the papyrologist's tracings and the v8Q legibility gate, for lead A's V3 pairs vs v2's pairs.

MEASUREMENT ONLY: the tracings T1 / T2 (x_followup/A_label, mapped onto P1's V5Rc carrier exactly as lane P2's score_carrier.py does)
are used only to score maps; nothing is drawn, trained or chosen from them.  Copies of P2's score_carrier.py / score_A.py (code/legib/).
Arms (mean of per-model z over the valid px of the scored window, as score_carrier.py): rv2+v8q (v2 main image), rv2+tr (v2 views 1/2/6),
v3, v3+tr, v3+v8q, and single rv2 / tr / v8q for reference.  All arms on the SAME px (common coverage).
Per arm and tracing: line / rect AUC and AP, skeleton recall at area 1, t80 stroke width, letters separated, far marks (score_A.score_one).
Paired bootstrap of dAP (2 mm u-blocks, 2000 resamples, 90 % CI) over A's whole line (u -13.5..11.5, v -2.5..4.0) against the v2 main
pair, and the v8Q legibility gate (vs T2, as PREREG amendment 9): t80 width <= base + .05 mm, skeleton recall >= base - .02, letters
separated >= base - 1, far marks <= 1.25 x base.  Writes legib.json."""
import os, sys, json
import numpy as np
A = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, f'{A}/code/legib')
from score_carrier import tracings_carrier, load                                  # noqa: E402
from score_A import score_one                                                     # noqa: E402

MAPS = f'{A}/maps'; NB = 4096
ARMS = {'rv2+v8q': ('rv2', 'v8q'), 'rv2+tr': ('rv2', 'tr'), 'v3': ('v3',), 'v3+tr': ('v3', 'tr'), 'v3+v8q': ('v3', 'v8q'),
        'rv2': ('rv2',), 'tr': ('tr',), 'v8q': ('v8q',)}
BASES = ('rv2+v8q', 'rv2+tr')


def ap_hist(hp, hn):
    tp = np.cumsum(hp[..., ::-1], -1); fp = np.cumsum(hn[..., ::-1], -1)
    prec = tp / np.maximum(tp + fp, 1); rec = tp / np.maximum(tp[..., -1:], 1)
    drec = np.diff(np.concatenate([np.zeros(rec.shape[:-1] + (1,)), rec], -1), axis=-1)
    return (drec * prec).sum(-1)


def ranks01(a, m):
    """rank-preserving map of a[m] to [0, 1) (AP depends only on the order)"""
    out = np.zeros_like(a, np.float32); v = a[m]; o = np.argsort(v, kind='stable'); r = np.empty(len(v), np.float32)
    r[o] = np.arange(len(v), dtype=np.float32) / len(v); out[m] = r; return out


def boot(a0, a1, sc, T, U, rng, nboot=2000):
    u = U[sc]; blk = np.floor((u - u.min()) / 2.0).astype(int); nb = int(blk.max()) + 1
    y = T[sc]; H = {}
    for k, a in (('b', a0), ('f', a1)):
        idx = np.clip((a[sc] * NB).astype(int), 0, NB - 1)
        hp = np.zeros((nb, NB)); hn = np.zeros((nb, NB))
        np.add.at(hp, (blk[y], idx[y]), 1); np.add.at(hn, (blk[~y], idx[~y]), 1); H[k] = (hp, hn)
    w = np.stack([np.bincount(b, minlength=nb) for b in rng.integers(0, nb, size=(nboot, nb))]).astype(float)
    ap = {k: float(ap_hist(h[0].sum(0), h[1].sum(0))) for k, h in H.items()}
    d = ap_hist(w @ H['f'][0], w @ H['f'][1]) - ap_hist(w @ H['b'][0], w @ H['b'][1])
    return dict(ap_base=round(ap['b'], 4), ap_arm=round(ap['f'], 4), dap=round(ap['f'] - ap['b'], 4), sd=round(float(d.std()), 4),
                ci90=[round(float(np.percentile(d, 5)), 4), round(float(np.percentile(d, 95)), 4)], n_blocks=nb, n_px=int(sc.sum()))


def legible(b, f):
    tb, tf = b['t80'], f['t80']
    chk = dict(width=tf['width_mm'] <= tb['width_mm'] + 0.05, skel=f['skel_recall_area1'] >= b['skel_recall_area1'] - 0.02,
               separated=tf['separated'] >= tb['separated'] - 1, far=tf['far_marks'] <= 1.25 * tb['far_marks'])
    return dict(checks=chk, passed=all(chk.values()))


def main():
    (t1, t2), U, V = tracings_carrier()
    raw = {k: load(f'{MAPS}/fwd_{k}.npy') for k in ('rv2', 'tr', 'v8q', 'v3')}
    cov = np.logical_and.reduce([np.isfinite(a) & (a > 0) for a in raw.values()])
    arms = {}
    for name, mods in ARMS.items():
        zs = [np.where(cov, (raw[k] - raw[k][cov].mean()) / (raw[k][cov].std() + 1e-6), 0) for k in mods]
        arms[name] = np.mean(zs, 0).astype(np.float32)
    sc = (U >= -13.5) & (U <= 11.5) & (V >= -2.5) & (V <= 4.0) & cov
    res = dict(common_px=int(cov.sum()), line_px=int(sc.sum()), arms={k: list(v) for k, v in ARMS.items()})
    sco = {name: {tn: score_one(a, cov, T, U, V) for tn, T in (('T1', t1), ('T2', t2))} for name, a in arms.items()}
    res['scores'] = sco
    rk = {name: ranks01(a, sc) for name, a in arms.items()}
    rng = np.random.default_rng(0); res['vs'] = {}
    for base in BASES:
        for name in arms:
            if name == base or name in ('rv2', 'tr', 'v8q'): continue
            r = {tn: boot(rk[base], rk[name], sc, T, U, rng) for tn, T in (('T1', t1), ('T2', t2))}
            r['legibility_vs_T2'] = legible(sco[base]['T2'], sco[name]['T2'])
            res['vs'][f'{name} vs {base}'] = r
            print(f'{name} vs {base}:', {t: (r[t]["dap"], r[t]["ci90"]) for t in ("T1", "T2")}, 'legible', r['legibility_vs_T2'], flush=True)
    for name in arms:
        s = sco[name]['T2']
        print(f'{name:8s} T2 line_ap {s["line_ap"]:.4f} rect_ap {s["rect_ap"]:.4f} skel {s["skel_recall_area1"]:.3f} '
              f't80 width {s["t80"]["width_mm"]:.3f} sep {s["t80"]["separated"]}/{s["t80"]["components"]} far {s["t80"]["far_marks"]}', flush=True)
    json.dump(res, open(f'{A}/legib.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
