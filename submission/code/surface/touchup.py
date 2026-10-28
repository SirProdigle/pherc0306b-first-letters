"""touchup.py CARRIER_DIR MESH_DIR [NAME [ONLY [OUT_DIR]]] : V5 "consensus touch-up" (defined 15 Oct 01:30, CT/geometry only, before V5 has any ink).
Keep the ORIGINAL (V0 = A, else w074big) everywhere, EXCEPT where independent constructions agree with each other and not with it:
  the two independent cloud spiral-fit copies of winding w074 (lanes z8400 and z8400o; above the line also lane z9900 w074) and the
  m7 tracker V3 lie within 4 vx of one another, while the original is > 4 vx away.  There the surface is moved to V3 (the m7 sheet,
  which the fits confirm).  Mask opened (1 cell), grown by 1 cell, kept only in components >= 0.5 mm2; feathered over 0.35 mm.
Writes MESH_DIR/V5_touchup/ + track_v5.npz (mask, weights, offsets) and a mask image."""
import os, sys, json
import numpy as np, tifffile, cv2
from scipy import ndimage as ndi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p1geom as g                                                                   # noqa: E402
Cd, Md = sys.argv[1:3]
NAME = sys.argv[3] if len(sys.argv) > 3 else 'V5_touchup'
ONLY = tuple(float(x) for x in sys.argv[4].split(',')) if len(sys.argv) > 4 else None   # keep only components whose centroid is in u0,u1,v0,v1
OUT = sys.argv[5] if len(sys.argv) > 5 else Md                                        # I2: output dir (P1: the track dir)
C = np.load(f'{Cd}/carrier.npz'); P, n, ok, U, V = C['P'], C['n'], C['ok'], C['u'], C['v']; cell = float(C['CELL'])
T = np.load(f'{Md}/track.npz'); T3 = np.load(f'{Md}/track_m7.npz')
v0 = T['V0_orig']; v3 = T3['V2_trackP']
l1, l2 = C['x_z84_w074'], C['x_z84o_w074']
agree_lanes = np.isfinite(l1) & np.isfinite(l2) & (np.abs(l1 - l2) <= 4)
lane = np.where(agree_lanes, (l1 + l2) / 2, np.nan)
# z9900 lane w074 (above the line), crossings computed here
if os.path.exists(f'{Md}/x_z99_w074.npy'):
    l3 = np.load(f'{Md}/x_z99_w074.npy')
else:
    S = g.Surf(f'{g.W}/x_vast_plan/results/PHerc0306B_z9900/results/flat/w074/tifxyz/flatten.tifxyz', 'z99', up=4)
    t, _ = S.cross(np.where(ok[..., None], P, np.nan).reshape(-1, 3), n.reshape(-1, 3), span=60); l3 = t.reshape(U.shape)
    np.save(f'{Md}/x_z99_w074.npy', l3)
cons = np.where(np.isfinite(lane), lane, np.nan)
only99 = ~np.isfinite(cons) & np.isfinite(l3)
cons = np.where(only99, l3, cons)
m = np.isfinite(cons) & (np.abs(v3 - cons) <= 4) & (np.abs(v0 - cons) > 4) & ok
m = ndi.binary_opening(m, iterations=1); m = ndi.binary_dilation(m, iterations=1) & ok
lab, nl = ndi.label(m); sizes = ndi.sum(m, lab, range(1, nl + 1)) * cell ** 2
keep = np.isin(lab, 1 + np.nonzero(sizes >= 0.5)[0]); m = keep
if ONLY is not None:
    lab2, n2 = ndi.label(m); cen = ndi.center_of_mass(m, lab2, range(1, n2 + 1))
    ids = [k + 1 for k, (ci, cj) in enumerate(cen) if ONLY[0] <= -36 + cj * cell <= ONLY[1] and ONLY[2] <= -21 + ci * cell <= ONLY[3]]
    m = np.isin(lab2, ids); print('kept components', len(ids), 'of', n2, 'area mm2', round(float(m.sum() * cell ** 2), 1))
w = np.clip(ndi.gaussian_filter(m.astype(float), 0.35 / cell / 2) * 2, 0, 1); w = np.maximum(w, m)
f = w * v3 + (1 - w) * v0
print(f'touch-up mask {m.mean():.3f} of the grid ({m.sum() * cell ** 2:.1f} mm2, {int((sizes >= 0.5).sum())} components); '
      f'consensus available at {np.isfinite(cons).mean():.2f}; V3~lanes & orig off: before cleaning ...')
for k, (u0, u1, a0, a1) in dict(L=(-15, 12, -2.5, 2.5), RA=(-20, 15, -8, -2.5), RB=(-20, 15, 2.5, 8), LL=(-35, -20, -4, 4), LR=(12, 30, -4, 4),
                                CA=(-30, 25, -20, -8), CB=(-30, 25, 8, 20)).items():
    r = (U >= u0) & (U < u1) & (V >= a0) & (V < a1)
    print(f'   {k}: touched {m[r].mean():.3f}; consensus available {np.isfinite(cons[r]).mean():.2f}; orig within 4 of consensus '
          f'{np.mean(np.abs(v0[r] - cons[r])[np.isfinite(cons[r])] <= 4) if np.isfinite(cons[r]).any() else float("nan"):.2f}')
np.savez_compressed(f'{OUT}/track_v5.npz' if NAME == 'V5_touchup' else f'{OUT}/track_{NAME}.npz', V5_touchup=f.astype(np.float32), mask=m, w=w.astype(np.float32), cons=cons.astype(np.float32))
d = f'{OUT}/{NAME}'; os.makedirs(d, exist_ok=True)
Q = P + f[..., None] * n; Q[~ok] = -1
for k, c in enumerate('xyz'): tifffile.imwrite(f'{d}/{c}.tif', Q[..., k].astype(np.float32))
meta = json.load(open(f'{g.SURF["A"]}/meta.json'))
for k_ in ('vc_gsfs_params', 'vc_gsfs_mode', 'elapsed_time_s', 'max_gen', 'seed', 'area_vx2', 'area_cm2'): meta.pop(k_, None)
meta.update(uuid=NAME, name=NAME, source='P1_surface_refine/touchup.py',
            bbox=[np.nanmin(Q[ok], 0).tolist(), np.nanmax(Q[ok], 0).tolist()],
            p1=dict(rule='original except where lanes z8400/z8400o (or z9900) w074 and the m7 track agree within 4 vx and the original is > 4 vx off'))
json.dump(meta, open(f'{d}/meta.json', 'w'), indent=1)
im = np.zeros(U.shape + (3,), np.uint8); im[np.isfinite(cons)] = (60, 60, 60); im[m] = (0, 200, 255)
im = cv2.resize(im, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
cv2.rectangle(im, (int((-19.9 + 36) / cell * 4), int((-7.8 + 21) / cell * 4)), (int((12.3 + 36) / cell * 4), int((4.6 + 21) / cell * 4)), (255, 255, 255), 2)
cv2.imwrite(f'{OUT}/touchup_mask_{NAME}.jpg', im); print('wrote', d)
