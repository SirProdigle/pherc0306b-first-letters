"""cards_from_tri.py TRI_JSON MAPS_DIR RENDERS_DIR OUT_DIR UM [--review2 5] [--label TEXT] : review cards for a lane.

Cards (CANDIDATES/_tools/card.py layout) for every tri_rank REVIEW window plus the first --review2 REVIEW-2 windows
(tri_rank already sorts REVIEW-2 by fewest soft-gate failures, then row score). The ring is auto-placed on the densest ink inside
the window. Also writes OUT_DIR/index.tsv (one row per card: rank, queue, gates)."""
import argparse, json, os, sys
import numpy as np, zarr, cv2
cv2.setNumThreads(int(os.environ.get('OMP_NUM_THREADS', '4')))
sys.path.insert(0, '/home/liam/projects/vesuvius-work/CANDIDATES/_tools')
from card import make_card, auto_roi

ap = argparse.ArgumentParser()
ap.add_argument('tri'); ap.add_argument('maps'); ap.add_argument('renders'); ap.add_argument('out'); ap.add_argument('um', type=float)
ap.add_argument('--review2', type=int, default=5); ap.add_argument('--label', default=''); ap.add_argument('--win', type=int, default=2600)
a = ap.parse_args()
rows = json.load(open(a.tri))
sel = [r for r in rows if r['queue'] == 'REVIEW'] + [r for r in rows if r['queue'].startswith('REVIEW-2')][:a.review2]
os.makedirs(a.out, exist_ok=True)
idx = open(os.path.join(a.out, 'index.tsv'), 'w')
idx.write('rank\tcard\tqueue\tscore\trow_off\tfib_off\trel_tex\tcrack\tshape\trecur\n')
for i, r in enumerate(sel, 1):
    n, side, x0 = r['name'], r['side'], r['x0']
    zf = os.path.join(a.maps, n, f"{n}_d0_{'reverse' if side == 'rev' else 'fwd'}_z.npy")
    rv = zarr.open(os.path.join(a.renders, n + '.zarr'), mode='r')['0']
    ct = np.asarray(rv[rv.shape[0] // 2]).astype(np.float32); z = np.load(zf).astype(np.float32)
    v = ct > 0; m = np.zeros_like(v); m[:, x0:x0 + a.win] = True
    try:
        roi = auto_roi(ct, z, a.um, r_mm=5.0, valid=v & m)
    except Exception:
        roi = (ct.shape[0] // 2, x0 + a.win // 2, int(5000 / a.um), int(5000 / a.um))
    name = f"{i:02d}_{'R' if r['queue'] == 'REVIEW' else 'R2'}_{n}_{side}_x{x0}_CARD.jpg"
    notes = [f"{a.label} {n} {'back' if side == 'rev' else 'front'} face, window x {x0}-{x0 + a.win}, row score {r['score']}",
             f"tri_rank: {r['queue']}",
             f"rows {r['row_off']} deg off winding, fibres {r['fib_off']} off, papyrus texture {r.get('rel_tex')}, crack {r.get('crack')}, "
             f"shape {r.get('shape')}, recur w+-2 {r.get('recur')}",
             'Compare with CANDIDATES/0_REFERENCE_known_text/PHerc0343_spiral-lane_positive-control_w047_reverse_CARD.jpg']
    make_card(ct, z, roi, a.um, f"{n} {side} x{x0}", notes, os.path.join(a.out, name), valid=v)
    idx.write('\t'.join(str(x) for x in (i, name, r['queue'], r['score'], r['row_off'], r['fib_off'], r.get('rel_tex'), r.get('crack'),
                                         r.get('shape'), r.get('recur'))) + '\n')
    print(name, flush=True)
idx.close()
print('CARDS', len(sel))
