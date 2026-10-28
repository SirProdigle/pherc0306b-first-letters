"""p4a_judge.py [v8Q] [v8S] [wise] : PREREG amendment 9 verdicts (CPU; measurement only).  Writes eval/p4a_judge.json.

v8Q / v8S : no-harm on w060 HF (noharm_v8.json, + p1_judge.q4 on the maps), A whole-line bar + Q4 legibility (boot_carrier),
            v8S controls when their maps exist (decoy on pick 2, depth shuffle on A's line3).
wise      : WiSE a0.5 of fv8: F-d (0343 no-harm), F-a pooled (eval/boot.json), F-b (eval/finder.json), F-c decoy, F-e if present.
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v8lib as L                                                                 # noqa: E402

EV, MAPS = f'{L.T}/eval', f'{L.T}/maps'
K = 1000.0 / 8.64


def jl(f):
    return json.load(open(f)) if os.path.exists(f) else {}


def corr(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    return float(np.corrcoef(a, b)[0, 1]) if a.size > 2 and a.std() > 0 and b.std() > 0 else float('nan')


def noharm(tag):
    from dsweep import data
    from p1_judge import q4
    nh = jl(f'{EV}/noharm_v8.json'); b, f = nh.get('base:w060hf:s21:d0'), nh.get(f'{tag}:w060hf:s21:d0')
    if not b or not f: return None
    _, _, _, lab, reg = data('w060hf')
    out = dict(base=dict(auc=b['auc'], ap=b['ap']), ft=dict(auc=f['auc'], ap=f['ap']), dap=round(f['ap'] - b['ap'], 4),
               passed=f['ap'] >= b['ap'] - 0.01)
    for k, t in (('base', 'base'), ('ft', tag)):
        mf = f'{MAPS}/nh_{t}_w060hf.npy'
        if os.path.exists(mf): out[k].update(q4(np.load(mf).astype(np.float32), lab, reg))
    return out


def a_line(tag):
    ff = f'{MAPS}/V5Rc_line3_{tag}_asc.npy'
    if not os.path.exists(ff): return None
    import subprocess
    subprocess.check_call([sys.executable, f'{HERE}/boot_carrier.py', tag, f'{MAPS}/V5Rc_box3_base_dl0_asc.npy', ff])
    return jl(f'{EV}/p4a_{tag}.json')


def shuffle(tag):
    """F-e style: r(id-flip d0 map, layer-shuffled map) over A's whole line on the carrier, model vs base"""
    pairs = {'base': (f'{MAPS}/parts/V5Rc_box3_base_dl0_asc_s21_id_d+0.npy', f'{MAPS}/V5Rc_box3_base_dl0_asc.json', f'{MAPS}/V5Rc_line3shuf1_base_asc.npy'),
             tag: (f'{MAPS}/parts/V5Rc_line3_{tag}_asc_s21_id_d+0.npy', f'{MAPS}/V5Rc_line3_{tag}_asc.json', f'{MAPS}/V5Rc_line3shuf1_{tag}_asc.npy')}
    if not all(os.path.exists(p) for v in pairs.values() for p in v): return None
    Y, X = np.mgrid[0:4900, 0:7780]; U, V = X / K - 36.0, Y / K - 21.0
    line = (U >= -13.5) & (U <= 11.5) & (V >= -2.5) & (V <= 4.0); out = {}
    for k, (fo, jo, fs) in pairs.items():
        y0, y1, x0, x1 = json.load(open(jo))['box']; o = np.full((4900, 7780), np.nan, np.float32); o[y0:y1, x0:x1] = np.load(fo).astype(np.float32)
        y0, y1, x0, x1 = json.load(open(fs[:-4] + '.json'))['box']; s = np.full((4900, 7780), np.nan, np.float32); s[y0:y1, x0:x1] = np.load(fs).astype(np.float32)
        m = line & np.isfinite(o) & np.isfinite(s) & (o > 0) & (s > 0)
        out[k] = dict(r=round(corr(o[m], s[m]), 4), n=int(m.sum()))
    out['passed'] = out[tag]['r'] <= out['base']['r'] + 0.05
    return out


def decoy(tag='v8S_F'):
    z = np.load(f'{L.T}/p4/v8s_labels.npz'); cy, cx = int(z['y0']), int(z['x0'])
    dp = np.zeros((6440, 6440), bool); h, w = z['dpos'].shape; dp[cy:cy + h, cx:cx + w] = z['dpos']
    out = {}
    for k in ('base', tag, 'v8S'):
        f = f'{MAPS}/P2_decoy_{k}_asc.npy'
        if not os.path.exists(f): continue
        y0, y1, x0, x1 = json.load(open(f[:-4] + '.json'))['box']; a = np.load(f).astype(np.float32); d = dp[y0:y1, x0:x1]
        v = a > 0; share = d[v].mean(); t = np.quantile(a[v], 1 - share)
        out[k] = dict(r=round(corr(a[v], d[v].astype(np.float32)), 4), recall=round(float((a[d & v] >= t).mean()), 4), share=round(float(share), 4))
    if 'base' in out and tag in out:
        out['passed'] = out[tag]['r'] <= out['base']['r'] + 0.10 and out[tag]['recall'] <= out['base']['recall'] + 0.10
    return out or None


def wise():
    nh = jl(f'{EV}/noharm_v8.json'); base = nh.get('base:0343:s21:d0', {}).get('ap')
    fd = {k: nh.get(f'wise_{k}_a0.5:0343:s21:d0', {}).get('ap') for k in ('fL', 'fR')}
    out = dict(F_d=dict(base=base, **fd, passed=None if None in fd.values() else all(v >= base - 0.01 for v in fd.values())))
    b = jl(f'{EV}/boot.json').get('WiSE fv8 a0.5 (W(fL)->right, W(fR)->left) vs v8 base')
    if b: out['F_a'] = dict(T1=b['T1'], T2=b['T2'], passed=all(b[t]['dap'] >= 0.02 and b[t]['ci90'][0] > 0 for t in ('T1', 'T2')))
    fe = jl(f'{EV}/finder.json'); ho = fe.get('heldout', {})
    try:
        fb = {sd: ho[f'WiSE fv8 {a} a0.5 | heldout {sd}']['offline_hot_r50'] / ho[f'v8 base | heldout {sd}']['offline_hot_r50'] for a, sd in (('L', 'right'), ('R', 'left'))}
        out['F_b'] = dict(ratio=fb, passed=all(v <= 1.5 for v in fb.values()))
    except KeyError: pass
    dc = fe.get('decoy', {})
    if 'WiSE fv8 F a0.5' in dc and 'v8 base' in dc:
        w, b0 = dc['WiSE fv8 F a0.5'], dc['v8 base']
        out['F_c'] = dict(wise=w, base=b0, passed=w['r_decoy_label'] <= b0['r_decoy_label'] + 0.10 and w['decoy_recall'] <= b0['decoy_recall'] + 0.10)
    return out


if __name__ == '__main__':
    what = sys.argv[1:] or ['v8Q', 'v8S', 'wise']
    res = jl(f'{EV}/p4a_judge.json')
    for tag in what:
        if tag == 'wise': res['wise_fv8'] = wise(); print('wise', json.dumps(res['wise_fv8']), flush=True); continue
        r = dict(noharm=noharm(tag), a_line=a_line(tag))
        if tag == 'v8S': r['shuffle'] = shuffle('v8S'); r['decoy'] = decoy('v8S_F')
        al = r['a_line'] or {}
        r['verdict'] = dict(bar=al.get('bar'), legibility=(al.get('legibility') or {}).get('passed'),
                            noharm=(r['noharm'] or {}).get('passed'))
        if tag == 'v8S': r['verdict'].update(shuffle=(r['shuffle'] or {}).get('passed'), decoy=(r['decoy'] or {}).get('passed'))
        res[tag] = r
        print(tag, 'verdict', r['verdict'], 'noharm', r['noharm'] and {k: r['noharm'][k] for k in ('dap', 'passed')},
              'A', al and {t: al[t] for t in ('T1', 'T2')}, flush=True)
    json.dump(res, open(f'{EV}/p4a_judge.json', 'w'), indent=1)
