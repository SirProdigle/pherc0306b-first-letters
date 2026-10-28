"""control343.py CONTROL.npz OUT.json : environment check for the max-quality recipe on a rented box.

CONTROL.npz (made locally by make_control.py) holds the x_zoom/validate343 crop of Erwin's PHerc0343 segment
(fl/0343/concat_66.zarr, y 450-2350, x 2400-5300), layers 23..43 only (read layer c=10 here = 33 in the full render), plus
the x_px343 stroke label and mask crops.  Runs the same recipe as ink_full.py (reverse face) and scores AUC / d' / AP with
validate343.score's region (mask, 160 px from the crop border).  Local reference: AP .693, d' 3.44, AUC .894.
"""
import sys, json, time
import numpy as np
sys.path.insert(0, '/home/liam/projects/vesuvius-work/x_deepread')
from ink_full import recipe                                            # noqa: E402

MARGIN = 160
d = np.load(sys.argv[1]); vol, LAB, MSK = d['vol'], d['lab'].astype(bool), d['msk'].astype(bool)
t = time.time(); m = recipe(vol, 10, 8.64, 'reverse', log=lambda s: print(s, flush=True)); dt = time.time() - t
valid = vol[10] > 0; a = m['fused']
reg = MSK & valid; reg[:MARGIN] = reg[-MARGIN:] = False; reg[:, :MARGIN] = reg[:, -MARGIN:] = False
pos, neg = a[reg & LAB], a[reg & ~LAB]
lo, hi = min(pos.min(), neg.min()), max(pos.max(), neg.max())
hp, _ = np.histogram(pos, 1024, (lo, hi)); hn, _ = np.histogram(neg, 1024, (lo, hi))
auc = float((hp * (np.cumsum(hn) - hn / 2.0)).sum() / (hp.sum() * hn.sum()))
tp = np.cumsum(hp[::-1]); fp = np.cumsum(hn[::-1]); prec = tp / np.maximum(tp + fp, 1); rec = tp / hp.sum()
apv = float(np.sum(np.diff(np.concatenate([[0], rec])) * prec))
res = dict(auc=round(auc, 4), dprime=round(float((pos.mean() - neg.mean()) / (neg.std() + 1e-6)), 3), ap=round(apv, 4),
           seconds=round(dt, 1), local_reference=dict(auc=0.894, dprime=3.44, ap=0.693), ok=abs(apv - 0.693) <= 0.01)
json.dump(res, open(sys.argv[2], 'w'), indent=1); print(res, flush=True)
