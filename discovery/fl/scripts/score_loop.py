"""score_loop.py [--once] : write spectral.json (row-periodicity scores, both sides) into every finished maps dir
under fl/*/screen2/maps/ that lacks one. Runs forever (every 120 s) unless --once."""
import sys, os, glob, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from spectral_rows import score, qmask_for, count_rows
from fiber_align import alignment
import zarr, cv2
W = '/home/liam/projects/vesuvius-work'
while True:
    n = 0
    for d in sorted(glob.glob(f'{W}/fl/*/screen2/maps/*/')):
        out = d + 'spectral.json'
        if os.path.exists(out) or not os.path.exists(d + 'done_d0'):
            continue
        if not glob.glob(d + '*_d0_*_z.npy') and 'skipped' not in open(d + 'done_d0').read():
            continue                                   # pulled from a worker: its own scorer writes spectral.json
        if 'skipped' in open(d + 'done_d0').read():
            json.dump({'skipped': True}, open(out, 'w')); continue
        res = {}
        for f in glob.glob(d + '*_d0_*_z.npy'):
            side = 'rev' if '_reverse_' in f else 'fwd'
            try:
                zz = np.load(f); qm = qmask_for(f)
                res[side] = score(zz, qmask=qm)
                if res[side].get('score') is not None and res[side]['score'] >= 3:
                    res[side].update(count_rows(zz, res[side]['angle'], qm))
                    r = os.path.join(os.path.dirname(os.path.dirname(d.rstrip('/'))), 'renders', os.path.basename(d.rstrip('/')) + '.zarr')
                    if os.path.exists(r):
                        ct = np.asarray(zarr.open(r, mode='r')['0'][15]).astype(np.float32); vv = ct > 0
                        if qm is not None:
                            vv &= cv2.resize(qm.astype(np.uint8), (vv.shape[1], vv.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                        zr = zz.astype(np.float32)
                        if zr.shape != ct.shape: zr = cv2.resize(zr, (ct.shape[1], ct.shape[0]))
                        vb = (ct > 0) & (zr != 0); sg = 1.5 * 1000 / 8.64
                        mb = cv2.GaussianBlur(vb.astype(np.float32), (0, 0), sg) + 1e-6
                        zl = cv2.GaussianBlur(zr * vb, (0, 0), sg) / mb; cl = cv2.GaussianBlur(ct * vb, (0, 0), sg) / mb
                        res[side]['bright_corr'] = float(np.corrcoef(zl[vb], cl[vb])[0, 1]) if vb.sum() > 1000 else None
                        al = alignment(ct, vv, res[side]['angle'])
                        st = al['fibre_strength']
                        al['aligned_dominant'] = bool(al['min_diff_deg'] is not None and len(st) == 2 and
                            min(abs(res[side]['angle'] - al['fibre_angles'][0]), 180 - abs(res[side]['angle'] - al['fibre_angles'][0])) <= 15)
                        res[side]['align'] = al
            except Exception as e:
                res[side] = {'error': repr(e)}
        res['max'] = max([v.get('score') for v in res.values() if isinstance(v, dict) and v.get('score') is not None] + [-9])
        res['max_multirow'] = max([v.get('score') for v in res.values() if isinstance(v, dict) and v.get('score') is not None and v.get('n_gaps_ok', 0) >= 1] + [-9])
        res['version'] = 2
        res['max_aligned'] = max([v.get('score') for v in res.values() if isinstance(v, dict) and v.get('score') is not None
                                  and (v.get('align') or {}).get('aligned_dominant')] + [-9])
        res['max_ink'] = max([v.get('score') for v in res.values() if isinstance(v, dict) and v.get('score') is not None
                              and (v.get('align') or {}).get('aligned_dominant') and (v.get('bright_corr') or -1) > 0.1] + [-9])
        res['version'] = 3
        json.dump(res, open(out, 'w')); n += 1
        if res['max_ink'] >= 4.5:
            print(f'HIGH {res["max"]:.2f} {d}', flush=True)
    if '--once' in sys.argv:
        break
    time.sleep(120)
