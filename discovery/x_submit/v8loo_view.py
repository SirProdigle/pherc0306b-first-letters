"""v8loo_view.py : lead A's claim rectangle seen by the PHerc1447-trained v8-in (x_submit/v8loo.py output), beside the claim
image (Hecate + Reader v2 + TRACE) and the Reader v2 + TRACE fallback, all in the same levelled frame, scale and contrast mapping
(per-map z-score over on-sheet pixels, z 1..3 as ink).  Writes x_submit/v8loo/A_rectangle_four_views.jpg and copies it to the draft.
"""
import sys, json, shutil
import numpy as np, cv2, zarr
sys.path.insert(0, '/home/liam/projects/vesuvius-work/x_submit')
import make_submission_A as M                                            # frame, sample, page, label, RECT, paths
O = '/home/liam/projects/vesuvius-work/x_submit/v8loo'
order = sys.argv[1] if len(sys.argv) > 1 else 'False'
roi = json.load(open(f'{M.OUT}/controls/panel_spec.json'))['target']['roi']; y0, y1, x0, x1 = roi
c, d, n = M.frame(); r = M.RECT
ct = np.asarray(zarr.open(M.RENDER, mode='r')['0'][20]).astype(np.float32)
per = {m: np.load(f'{M.DR}/fwd_{f}.npy').astype(np.float32) for m, f in (('Hecate', 'hec'), ('Reader v2', 'rv2'), ('TRACE', 'tr'))}
valid = np.logical_and.reduce([np.isfinite(a) & (a != 0) for a in per.values()]) & (ct > 0)
zz = {m: M.zs(np.nan_to_num(a), valid) for m, a in per.items()}
v8 = np.zeros(ct.shape, np.float32); v8[y0:y1, x0:x1] = np.load(f'{O}/A_v8loo_reverse{order}.npy').astype(np.float32)
v8v = valid.copy(); v8v[:] = False; v8v[y0:y1, x0:x1] = valid[y0:y1, x0:x1]
z8 = M.zs(v8, v8v)
S = lambda a, nn=False: M.sample(a, c, d, n, r['u0'], r['u1'], r['v0'], r['v1'], nn=nn)
vr = S(valid.astype(np.uint8), True) > 0
views = [('claim image: Hecate + Reader v2 + TRACE', np.mean(list(zz.values()), 0) * valid),
         ('Reader v2 + TRACE only', np.mean([zz['Reader v2'], zz['TRACE']], 0) * valid),
         ('v8-in fine-tuned on PHerc1447 text (independent model, never saw PHerc0306B)', z8)]
rows = [M.label(M.page(S(z), vr), t) for t, z in views]
img = np.concatenate(rows, 0)
cv2.imwrite(f'{O}/A_rectangle_four_views.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 92])
shutil.copy2(f'{O}/A_rectangle_four_views.jpg', f'{M.OUT}/controls/independent_model_v8in_1447.jpg')
a = S(np.mean(list(zz.values()), 0) * valid); b = S(z8); m = vr & (a != 0) & (b != 0)
print('r(claim, v8loo) in rectangle', round(float(np.corrcoef(a[m], b[m])[0, 1]), 3), 'hot overlap', round(float(((a > 2) & (b > 2) & m).sum() / max(1, (((a > 2) | (b > 2)) & m).sum())), 3))
