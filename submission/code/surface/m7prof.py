"""m7prof.py CARRIER_DIR : profiles of the published m7 surface prediction (PHerc0306B 20250521133212-surface-m7-L0-th0.2, full
resolution) along the carrier normals, same nodes / depths / 3x3 footprint as the CT profiles (carrier.py); value = mean of the
binary m7 mask over the footprint, scaled to 0..100 so track.py's prominence threshold has the same meaning (2 = 2 % of the
footprint).  Writes CARRIER_DIR/prof_m7.npy (float16)."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p1geom as g                                                                   # noqa: E402
from carrier import tangents                                                        # noqa: E402

d = sys.argv[1]
C = np.load(f'{d}/carrier.npz'); P, n, ok = C['P'], C['n'], C['ok']
ts = np.load(f'{d}/prof_ts.npy') if os.path.exists(f'{d}/prof_ts.npy') else np.arange(-60, 61, dtype=float)
V = g.Vol(which='surface_url', threads=24)
tu, tv = tangents(np.nan_to_num(P)); foot = 7
offs = [(a, b) for a in (-foot, 0, foot) for b in (-foot, 0, foot)]
out = np.zeros(P.shape[:2] + (len(ts),), np.float16); t0 = time.time()
for r0 in range(0, P.shape[0], 12):
    sl = slice(r0, min(P.shape[0], r0 + 12)); acc = np.zeros(P[sl].shape[:2] + (len(ts),), np.float32)
    for a_, b_ in offs:
        q = P[sl] + a_ * tu[sl] + b_ * tv[sl]
        L = q[:, :, None, :] + ts[None, None, :, None] * n[sl][:, :, None, :]; L[~ok[sl]] = np.nan
        acc += V.sample(L, order=0) > 0
    out[sl] = (100.0 * acc / len(offs)).astype(np.float16)
    print(f'  rows {r0}-{sl.stop}  {time.time() - t0:.0f}s', flush=True)
np.save(f'{d}/prof_m7.npy', out); print('done')
