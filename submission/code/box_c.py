"""box_c.py : the owner's drawn box for lead A (20 Oct, screenshot on v4's AFTER image): keep the left edge, pull the right edge in to
just past the last letter (nu), take the bottom down to just above the dark cluster at bottom right (next row), top as high as the
4 cm2 budget allows without cutting the lone blob above the line.  Edges sit >= CLR mm outside the quarter-tone extent of the marks
they hug.  Scored as BOXES_PREREG.md.  Model images only."""
import json, sys
import numpy as np, cv2
A10 = '/home/liam/projects/vesuvius-work/x_req/20261020/A10'; sys.path.insert(0, A10)
import box_search as S
sys.path.insert(0, f'{A10}/SUBMISSION_v4_owner_only/code')
import make_claim_image_v2 as C
CTX = S.CTX; K = S.K; CLR = 0.2
x0, y0 = S.px(CTX['u0'], CTX['v0']); x1, y1 = S.px(CTX['u1'], CTX['v1']); sl = (slice(y0, y1), slice(x0, x1))
ct = cv2.imread(f'{S.M}/ct20.png', -1)[sl]
v3 = np.load(f'{S.M}/fwd_v3.npy', mmap_mode='r')[sl].astype(np.float32); tr = np.load(f'{S.M}/fwd_tr.npy', mmap_mode='r')[sl].astype(np.float32)
valid = (ct > 0) & np.isfinite(v3) & np.isfinite(tr) & (v3 != 0)
d, lv = S.view(('v3', 'tr'), sl, valid)
marks = {m['i']: m for m in json.load(open(f'{A10}/box_search/marks.json'))}
nq, labq = cv2.connectedComponents(((d >= 0.25) & valid).astype(np.uint8), connectivity=8)
def ext(i):   # quarter-tone extent (u0, u1, v0, v1) of the component holding mark i's centre
    m = marks[i]; s = labq[int((m['cv'] - CTX['v0']) * K), int((m['cu'] - CTX['u0']) * K)]
    ys, xs = np.where(labq == s)
    return dict(u0=CTX['u0'] + xs.min() / K, u1=CTX['u0'] + (xs.max() + 1) / K, v0=CTX['v0'] + ys.min() / K, v1=CTX['v0'] + (ys.max() + 1) / K, npx=len(xs))
E = {i: ext(i) for i in (43, 58, 57, 62, 67, 12, 13, 10, 34, 36, 37, 38)}
for i, e in E.items(): print(i, {k: round(v, 2) for k, v in e.items()})
u0 = min(-15.0, round(E[43]['u0'] - CLR, 2)); u1 = round(max(E[58]['u1'], E[57]['u1']) + CLR, 2)
v0 = round(max(E[12]['v1'], E[13]['v1']) + CLR, 2)                       # just above the bottom-right cluster
vmax_area = v0 + 395.0 / (u1 - u0); vmax_blob = E[67]['v0'] - CLR         # budget vs not cutting blob 67
v1 = round(min(vmax_area, vmax_blob), 2)
B = dict(u0=u0, u1=u1, v0=v0, v1=v1)
ins = lambda m, b: b['u0'] <= m['u0'] and m['u1'] <= b['u1'] and b['v0'] <= m['v0'] and m['v1'] <= b['v1']   # noqa: E731
cin = lambda m, b: b['u0'] <= m['cu'] <= b['u1'] and b['v0'] <= m['cv'] <= b['v1']                          # noqa: E731
cut = lambda m, b: (m['u0'] < b['u1'] and m['u1'] > b['u0'] and m['v0'] < b['v1'] and m['v1'] > b['v0']) and not ins(m, b)  # noqa: E731
MESH = f'{A10}/SUBMISSION_v4_owner_only/mesh/PHerc0306B_20250521133212_w074own_touchup_V5Rc'
def score(b):
    conf = [i for i, m in marks.items() if m['confirmed'] and ins(m, b)]
    return dict(b, S=len(conf), conf=conf, stray=sum(1 for m in marks.values() if m['cls'] != 'letter' and cin(m, b)),
                letter_cut=[i for i, m in marks.items() if m['cls'] == 'letter' and cut(m, b)],
                flat_cm2=round((b['u1'] - b['u0']) * (b['v1'] - b['v0']) / 100, 4), mesh_cm2=round(C.mesh_box_area(MESH, b), 4))
V4 = dict(u0=-15.0, u1=25.0, v0=-5.0, v1=4.875); V4B = dict(u0=-15.0, u1=25.0, v0=-7.35, v1=2.525)
res = dict(extents_quarter_tone=E, limits=dict(vmax_area=vmax_area, vmax_blob=vmax_blob), v4=score(V4), v4b=score(V4B), v4c=score(B))
for k in ('v4', 'v4b', 'v4c'):
    r = res[k]; print(k, {x: r[x] for x in ('u0', 'u1', 'v0', 'v1', 'S', 'stray', 'letter_cut', 'flat_cm2', 'mesh_cm2')})
print('v4c lost vs v4', sorted(set(res['v4']['conf']) - set(res['v4c']['conf'])), 'gained', sorted(set(res['v4c']['conf']) - set(res['v4']['conf'])))
json.dump(res, open(f'{A10}/box_c.json', 'w'), indent=1, default=float)
R = S.rgb(d, valid); ua, ub, va, vb = -16.5, 26.5, -11.0, 6.5
X0, Y0 = S.px(ua, va); X1, Y1 = S.px(ub, vb); c = R[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0].copy()
for bb, col in ((V4, (220, 90, 30)), (B, (40, 40, 230))):
    p0 = S.px(bb['u0'], bb['v0']); p1 = S.px(bb['u1'], bb['v1']); cv2.rectangle(c, (p0[0] - X0, p0[1] - Y0), (p1[0] - X0, p1[1] - Y0), col, 6)
c = np.flipud(c).copy(); s = 1800 / c.shape[1]; c = cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
bar = np.full((44, c.shape[1], 3), 255, np.uint8)
cv2.putText(bar, f"blue: v4 box   red: your box (u {B['u0']:.2f}..{B['u1']:.2f} x v {B['v0']:.2f}..{B['v1']:.2f} mm, {res['v4c']['mesh_cm2']:.2f} cm2 on the mesh)", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
cv2.imwrite(f'{A10}/BOX_C_preview_owner_only.jpg', np.concatenate([bar, c]), [cv2.IMWRITE_JPEG_QUALITY, 90])
