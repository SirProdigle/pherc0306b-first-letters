"""seam306.py RENDER C ZMAP BBOX KEY : strip-seam test for a PHerc0306B cloud-lane row, with x_focus/seam_test.py's definitions.

Row fit = PCA of the fused-z>2 marks inside BBOX (render px). Measures edge energy / density / coherence / fibre-orientation
steps across the row at d-4..+4 (seam_test.measure) and a null of 40 random positions on the same render with the same
direction and length (row +-4 mm excluded). 0343 text-line references come from x_focus/out/seam/seam_test.json (same code).
"""
import sys, json, os
import numpy as np
W = '/home/liam/projects/vesuvius-work'
sys.path.insert(0, f'{W}/x_focus')
import seam_test as ST                                                       # noqa: E402

render, c, zf, bbox, key = sys.argv[1], int(sys.argv[2]), sys.argv[3], [int(v) for v in sys.argv[4].split(',')], sys.argv[5]
z = np.load(zf).astype(np.float32); x0, y0, x1, y1 = bbox
ys, xs = np.nonzero(z[y0:y1, x0:x1] > 2); xs = xs + x0; ys = ys + y0
P = np.column_stack([xs, ys]).astype(float); m = P.mean(0)
w, V = np.linalg.eigh(np.cov((P - m).T)); u = V[:, np.argmax(w)]
if u[0] < 0: u = -u
s = (P - m) @ u; s0, s1 = np.percentile(s, [2, 98]) / ST.PX; LEN = s1 - s0
mc = m + u * ST.PX * (s0 + s1) / 2
L = ST.load_layers(render, c)
res = {'row': ST.measure(L, mc, u, -LEN / 2, LEN / 2), 'row_fit': dict(centre_px=mc.tolist(), u=u.tolist(), length_mm=round(float(LEN), 1))}
vv = np.array([-u[1], u[0]]); H, Wd = L[0].shape; ok0 = L[4] > 0
def cc():
    while True:
        p = np.array([ST.rng.uniform(0, Wd), ST.rng.uniform(0, H)])
        if ok0[int(p[1]), int(p[0])]: return p
def ex(p):
    d = p - mc; return abs(d @ vv) / ST.PX < 4.0 and abs(d @ u) / ST.PX < LEN / 2 + 4
res['null'] = ST.null(L, cc, u, LEN, exclude=ex)
keys = ['E_rel_near', 'E_rel_at', 'step_density', 'step_coherence', 'step_cos2', 'step_sin2', 'step_max']
ref = json.load(open(f'{W}/x_focus/out/seam/seam_test.json'))['summary']
g = res['row']['agg']
summ = {'row': {k: dict(value=round(g[k], 3), pct_vs_own_null=ST.pct(g[k], [x[k] for x in res['null']])) for k in keys},
        'own_null_p50_p95': {k: [round(float(np.nanpercentile([x[k] for x in res['null']], q)), 3) for q in (50, 95)] for k in keys},
        'n_null': len(res['null']), 't_E_mm': round(g['t_E'], 2),
        'ref_343_line1': {k: ref['343_line1'][k] for k in keys}, 'ref_343_line2': {k: ref['343_line2'][k] for k in keys},
        'ref_A_userrow': {k: ref['A_row'][k] for k in keys}}
res['summary'] = summ
os.makedirs(f'{W}/x_focus306/out/seam', exist_ok=True)
json.dump(res, open(f'{W}/x_focus306/out/seam/{key}.json', 'w'), indent=1, default=lambda o: None)
print(json.dumps({k: summ[k] for k in ('row', 'own_null_p50_p95', 'n_null', 't_E_mm')}, indent=0))
