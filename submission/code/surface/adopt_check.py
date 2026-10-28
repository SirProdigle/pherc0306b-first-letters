"""adopt_check.py TRACKFILE CARRIER_DIR TRACK_DIR [OUT_DIR] : the label-free adoption rule for a touch-up (research lane Q5 rank-1 proposal, applied by P1), CT/geometry
only.  (1)+(2) recto score: run recto.py on the arm's render (reported separately).  (3) sheet identity: share of the touched nodes
where >= 2 independent surfaces (lane z8400 w074, lane z8400o w074, m7 track V3, R1) lie within 4 vx of the touch-up's offset (bar 80 %).
(4) seam continuity: |d offset| between neighbouring nodes inside the feather zone (bar <= 2 vx per 20 vx cell; reported as median /
p90 / share > 2, with the same statistic of the ORIGINAL there for scale)."""
import os, sys, json
import numpy as np
from scipy import ndimage as ndi
tf = sys.argv[1]; CD, TD = sys.argv[2], sys.argv[3]; OUT = sys.argv[4] if len(sys.argv) > 4 else TD   # I2: dirs as arguments
Tz = np.load(tf); key = [k for k in Tz.files if k.startswith('V5')][0]
f, m, w = Tz[key], Tz['mask'], Tz['w']
C = np.load(f'{CD}/carrier.npz'); T0 = np.load(f'{TD}/track.npz'); T3 = np.load(f'{TD}/track_m7.npz')
v0 = T0['V0_orig']
cands = {'lane z8400 w074': C['x_z84_w074'], 'lane z8400o w074': C['x_z84o_w074'], 'm7 track V3': T3['V2_trackP'], 'R1': C['x_R1']}
cnt = np.zeros(f.shape, int)
for k, x in cands.items(): cnt += (np.isfinite(x) & (np.abs(x - f) <= 4)).astype(int)
ident = float((cnt[m] >= 2).mean())
indep_nolanes = float(((np.isfinite(cands['lane z8400 w074']) & (np.abs(cands['lane z8400 w074'] - f) <= 4)) &
                       (np.isfinite(cands['lane z8400o w074']) & (np.abs(cands['lane z8400o w074'] - f) <= 4)))[m].mean())
fz = (w > 0.02) & (w < 0.98)
dh = np.abs(np.diff(f, axis=1)); dv = np.abs(np.diff(f, axis=0)); dh0 = np.abs(np.diff(v0, axis=1)); dv0 = np.abs(np.diff(v0, axis=0))
zh = fz[:, 1:] | fz[:, :-1]; zv = fz[1:] | fz[:-1]
e = np.concatenate([dh[zh], dv[zv]]); e0 = np.concatenate([dh0[zh], dv0[zv]])
res = dict(arm=key, touched_nodes=int(m.sum()), identity_ge2_within4=round(ident, 3), both_lane_copies_within4=round(indep_nolanes, 3),
           seam=dict(edges=int(e.size), median=round(float(np.median(e)), 2), p90=round(float(np.percentile(e, 90)), 2), share_gt2=round(float((e > 2).mean()), 3)),
           seam_original_same_edges=dict(median=round(float(np.median(e0)), 2), p90=round(float(np.percentile(e0, 90)), 2), share_gt2=round(float((e0 > 2).mean()), 3)))
print(json.dumps(res, indent=1)); json.dump(res, open(f'{OUT}/adopt_check_{os.path.basename(tf)[:-4]}.json', 'w'), indent=1)
# where are the steep seam edges?  split by the 4 cm2 rectangle (v <= +2.53 as rendered, u <= +23.54) vs outside
U, V = C['u'], C['v']
inr = (V <= 2.53) & (U <= 23.54) & (V >= -8.66) & (U >= -12.2)
ih = (inr[:, 1:] & inr[:, :-1])[zh]; iv = (inr[1:] & inr[:-1])[zv]; ii = np.concatenate([ih, iv])
for nm, sel in (('inside the 4 cm2 rectangle', ii), ('outside it', ~ii)):
    if sel.any():
        print(f'seam edges {nm}: n {sel.sum()}, median {np.median(e[sel]):.2f}, p90 {np.percentile(e[sel], 90):.2f}, share > 2 {np.mean(e[sel] > 2):.3f} '
              f'(original: median {np.median(e0[sel]):.2f}, p90 {np.percentile(e0[sel], 90):.2f}, share > 2 {np.mean(e0[sel] > 2):.3f})')
