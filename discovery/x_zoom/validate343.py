"""validate343.py : which max-quality settings help on PHerc0343 stroke labels (x_px343), scored on a crop.

Crop of Erwin's segment (fl/0343/concat_66.zarr, reverse face, c=33) around letters 5-18; maps are scored inside the
tracer's mask, away from the crop border (MARGIN px), with x_px343's AUC / d' / AP.  Each run is cached in
x_zoom/val343/<name>.npy so the final recipe can be assembled from parts."""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zoomlib as Z                                                        # noqa: E402
W = Z.W; OUT = f'{W}/x_zoom/val343'; os.makedirs(OUT, exist_ok=True)
R = f'{W}/fl/0343/concat_66.zarr'; C, UM, MARGIN = 33, 8.64, 160
Y0, Y1, X0, X1 = 450, 2350, 2400, 5300
LAB = np.load(f'{W}/x_px343/label_render.npy')[Y0:Y1, X0:X1]; MSK = np.load(f'{W}/x_px343/mask_render.npy')[Y0:Y1, X0:X1]


def score(a, valid):
    reg = MSK & valid; reg[:MARGIN] = reg[-MARGIN:] = False; reg[:, :MARGIN] = reg[:, -MARGIN:] = False
    sys.path.insert(0, f'{W}/x_px343')
    pos, neg = a[reg & LAB], a[reg & ~LAB]
    lo, hi = min(pos.min(), neg.min()), max(pos.max(), neg.max())
    hp, _ = np.histogram(pos, 1024, (lo, hi)); hn, _ = np.histogram(neg, 1024, (lo, hi))
    auc = float((hp * (np.cumsum(hn) - hn / 2.0)).sum() / (hp.sum() * hn.sum()))
    tp = np.cumsum(hp[::-1]); fp = np.cumsum(hn[::-1]); prec = tp / np.maximum(tp + fp, 1); rec = tp / hp.sum()
    ap = float(np.sum(np.diff(np.concatenate([[0], rec])) * prec))
    return dict(auc=round(auc, 4), dprime=round(float((pos.mean() - neg.mean()) / (neg.std() + 1e-6)), 3), ap=round(ap, 4))


def get(name, fn):
    f = f'{OUT}/{name}.npy'
    if os.path.exists(f): return np.load(f)
    t = time.time(); a = fn(); np.save(f, a.astype(np.float32)); print(f'  {name}: {time.time() - t:.0f}s', flush=True); return a


if __name__ == '__main__':
    runs = sys.argv[1:] or ['base']
    vol, box = Z.crop(R, Y0, Y1, X0, X1); valid = vol[C] > 0
    res = {}
    spec = {   # name -> (model, d, kwargs)
        'hec_s32_d0': ('hecate', 0, dict(stride=32)), 'hec_s16_d0': ('hecate', 0, dict(stride=16)),
        'hec_s32_tta_d0': ('hecate', 0, dict(stride=32, tta8=True)),
        'hec_s32_d-1': ('hecate', -1, dict(stride=32)), 'hec_s32_d1': ('hecate', 1, dict(stride=32)),
        'rv2_o50_d0': ('rv2', 0, dict(overlap=0.5)), 'rv2_o75_d0': ('rv2', 0, dict(overlap=0.75)),
        'rv2_o50_tta_d0': ('rv2', 0, dict(overlap=0.5, tta8=True)),
        'rv2_o50_d-1': ('rv2', -1, dict(overlap=0.5)), 'rv2_o50_d1': ('rv2', 1, dict(overlap=0.5)),
        'tr_o50_d0': ('trace', 0, dict(overlap=0.5)), 'tr_o75_d0': ('trace', 0, dict(overlap=0.75)),
        'tr_o50_tta_d0': ('trace', 0, dict(overlap=0.5, tta8=True)),
    }
    import re
    def parse(name):   # e.g. hec_s16_d-1, rv2_o50_tta_d1, tr_o50_tta_d0
        m = re.match(r'(hec|rv2|tr)_([so])(\d+)(_tta)?_d(-?\d+)$', name)
        mdl = {'hec': 'hecate', 'rv2': 'rv2', 'tr': 'trace'}[m[1]]
        kw = dict(stride=int(m[3])) if m[2] == 's' else dict(overlap=int(m[3]) / 100)
        if m[4]: kw['tta8'] = True
        return mdl, int(m[5]), kw
    for name in runs:
        if name.startswith('FUSE:'):     # FUSE:label=a+b|c   ('|' averages maps of one model over depth before z-fusion)
            label, expr = name[5:].split('=')
            parts = [np.mean([np.load(f'{OUT}/{x}.npy') for x in grp.split('|')], 0) for grp in expr.split('+')]
            res[label] = score(Z.zfuse(parts, valid), valid); print(label, res[label], flush=True); continue
        mdl, d, kw = spec[name] if name in spec else parse(name)
        fn = (lambda: Z.hecate_map(vol, UM, C, d, 'reverse', **kw)) if mdl == 'hecate' else (lambda: Z.villa_map(mdl, vol, C, d, 'reverse', **kw))
        a = get(name, fn); res[name] = score(a, valid); print(name, res[name], flush=True)
    json.dump(res, open(f'{OUT}/scores_{int(time.time())}.json', 'w'), indent=1)
