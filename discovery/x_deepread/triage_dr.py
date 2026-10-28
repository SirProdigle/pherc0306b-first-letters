"""triage_dr.py [--no-glyph] : window triage of the cloud deep-read maps (max-quality recipe) for every PHerc0306B winding.

The deep-read maps (x_deepread/results/b*/out/wNNN[_f]/fwd_fused.npy) sit on the same grid as the z8400 lane's standard maps
(same flatten.tifxyz, 2560 x W px at 8.64 um).  For each 2600 px window (1300 step, the tri_rank windows) on the reading face
(fwd), this computes what the lane computed box-side, but on the max-quality map:
  row  : spectral_rows.score (row periodicity vs tile-shuffled nulls), masked with the lane's quality tiles
  angle: row direction, and the CT fibre families (fiber_align on ct20, the render's centre layer) -> row_off / fib_off
  shape: share of hot area in elongated marks (stroke_feats.shape_horiz)
  glyph: x_glyph gA CNN window score (fl/scripts/glyph_win.window_scores), GPU via flock gpu.lock
It then joins the lane's standard-quality row score (results/tri_rank.json, where admitted) and glyph score (maps/*/glyph_win.json).
The 0306B windows are mostly blank papyrus, so each score is also given as a rank and a robust z within this scroll
(median / MAD over all deep-read windows): a text window should be an outlier on row or glyph AND pass the hard gates.
Output: x_deepread/triage_dr.json, printed table sorted by max(row z, glyph z).
"""
import sys, os, glob, json
import numpy as np, cv2
cv2.setNumThreads(4)
W = '/home/liam/projects/vesuvius-work'; D = f'{W}/x_deepread'; L = f'{W}/x_vast_plan/results/PHerc0306B_z8400'
sys.path.insert(0, f'{W}/fl/scripts')
from spectral_rows import score, qmask_for                     # noqa: E402
from fiber_align import alignment                               # noqa: E402
from stroke_feats import shape_horiz                            # noqa: E402
WIN, STEP, UM = 2600, 1300, 8.64
off90 = lambda ang: abs(((ang - 90) + 90) % 180 - 90)


def maps():
    out = {}
    for f in sorted(glob.glob(f'{D}/results/b*/out/w0*/fwd_fused.npy')):
        d = os.path.dirname(f); w = os.path.basename(d).split('_')[0]
        if w not in out or not os.path.basename(d).endswith('_f'): out[w] = d     # w0NN and w0NN_f are the same job
    return out


def windows(w, d, net=None):
    z = np.load(f'{d}/fwd_fused.npy').astype(np.float32)
    ct = cv2.imread(f'{d}/ct20.png', cv2.IMREAD_GRAYSCALE)
    lf = f'{L}/maps/L0306bacw_{w}/L0306bacw_{w}_d0_fwd_z.npy'
    q = qmask_for(lf)
    if q is not None and q.shape != z.shape:
        qp = np.zeros(z.shape, bool); h, ww = min(q.shape[0], z.shape[0]), min(q.shape[1], z.shape[1]); qp[:h, :ww] = q[:h, :ww]; q = qp
    gl = {}
    if net is not None:
        from glyph_win import window_scores
        gl = {x0: g for x0, g in window_scores(np.where(q, z, 0) if q is not None else z, UM, net)}
    rows = []
    for x0 in range(0, max(1, z.shape[1] - WIN + 1), STEP):
        zz = z[:, x0:x0 + WIN]; qq = None if q is None else q[:, x0:x0 + WIN]
        if (zz != 0).mean() < 0.3 or (qq is not None and qq.mean() < 0.3): continue
        try: sc = score(zz, qmask=qq)
        except Exception: continue
        if sc.get('score') is None: continue
        cc = ct[:, x0:x0 + WIN].astype(np.float32); vv = cc > 0
        if qq is not None: vv &= qq
        try:
            al = alignment(cc, vv, sc['angle']); fibs = al['fibre_angles']
        except Exception: fibs = []
        sh = shape_horiz(np.where(vv, zz, 0), vv, UM)[0]
        rows.append(dict(w=w, x0=x0, row=round(float(sc['score']), 2), row_angle=sc['angle'], row_off=round(off90(sc['angle']), 1),
                         fib_off=round(min(off90(x) for x in fibs), 1) if fibs else None,
                         shape=None if sh is None else round(float(sh), 3), glyph=gl.get(x0), onsheet=round(float(vv.mean()), 2)))
    return rows


if __name__ == '__main__':
    net = None
    if '--no-glyph' not in sys.argv:
        import torch
        sys.path.insert(0, f'{W}/x_glyph'); from train import load_model
        torch.cuda.set_per_process_memory_fraction(0.06)
        net = load_model(f'{W}/x_glyph/models/gA.pt')
    std_row = {(t['name'].rsplit('_', 1)[1], t['x0']): t for t in json.load(open(f'{L}/results/tri_rank.json')) if t['side'] == 'fwd'}
    std_gl = {}
    for f in glob.glob(f'{L}/maps/*/glyph_win.json'):
        w = os.path.basename(os.path.dirname(f)).rsplit('_', 1)[1]
        for x0, g in json.load(open(f)).get('fwd', []): std_gl[(w, x0)] = g
    allr = []
    for w, d in maps().items():
        r = windows(w, d, net); allr += r; print(w, len(r), 'windows', flush=True)
    rz = lambda k: (lambda v: (np.median(v), 1.4826 * np.median(np.abs(v - np.median(v))) + 1e-6))(np.array([r[k] for r in allr if r[k] is not None]))
    (mr, sr), (mg, sg) = rz('row'), rz('glyph') if net is not None else ((0, 1))
    for r in allr:
        r['row_z'] = round((r['row'] - mr) / sr, 2); r['glyph_z'] = None if r['glyph'] is None else round((r['glyph'] - mg) / sg, 2)
        s = std_row.get((r['w'], r['x0'])); r['std_row'] = None if s is None else s['score']; r['std_queue'] = None if s is None else s['queue'][:40]
        r['std_glyph'] = std_gl.get((r['w'], r['x0']))
        r['hard_ok'] = r['row_off'] <= 20 and r['fib_off'] is not None and r['fib_off'] <= 12
    allr.sort(key=lambda r: -max(r['row_z'], r['glyph_z'] or -9))
    print(f'null: row median {mr:.2f} MAD-sd {sr:.2f}; glyph median {mg:.2f} MAD-sd {sg:.2f}; {len(allr)} windows')
    for r in allr[:40]:
        print(f"{r['w']} x{r['x0']:<5} row {r['row']:6.2f} (z {r['row_z']:5.1f}, std {r['std_row']}) glyph {r['glyph']} (z {r['glyph_z']}, std {r['std_glyph']}) "
              f"rows {r['row_off']:4.1f} fib {r['fib_off']} shape {r['shape']} {'OK' if r['hard_ok'] else 'hard-fail'}")
    json.dump(dict(null=dict(row=[mr, sr], glyph=[mg, sg]), windows=allr), open(f'{D}/triage_dr.json', 'w'), indent=1, default=float)
