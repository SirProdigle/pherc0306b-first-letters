"""window_score.py MAPS_ROOT [--win 2600] [--step 1300] [--redo] : for every finished maps/<winding>/ dir, slide windows (full height,
width WIN px of the 31-layer render grid ~ 2.4 cm at 9.36 um) along the winding and score row periodicity with
fl/scripts/spectral_rows.score on each side; writes spectral_win.json {side: [[x0, score], ...], best: [score, side, x0]}."""
import sys, os, glob, json
import numpy as np
sys.path.insert(0, '/home/liam/projects/vesuvius-work/fl/scripts')
from spectral_rows import score, qmask_for
from fiber_align import alignment
import zarr, cv2
import os as _os; cv2.setNumThreads(int(_os.environ.get('OMP_NUM_THREADS', '4')))  # OpenCV ignores OMP caps; uncapped it used all 32 cores
QTHR = float(os.environ.get('FL_QUALITY_THR', '0.10'))   # same tile rule and threshold as the ink step's --mask (fastink.quality_mask)


def render_qmask(rdir, d, tile=128):
    qf = d + 'quality_tiles.npy'
    if not os.path.exists(qf):
        v = np.asarray(zarr.open(rdir, mode='r')['0']); c = v[v.shape[0] // 2]; valid = c > 0
        med = float(np.median(c[valid])) if valid.any() else 1.0
        ny, nx = valid.shape[0] // tile, valid.shape[1] // tile; sc = np.full((ny, nx), np.nan, np.float32)
        for i in range(ny):
            for j in range(nx):
                vv = valid[i * tile:(i + 1) * tile, j * tile:(j + 1) * tile]
                if vv.mean() < 0.8: continue
                pr = v[:, i * tile:(i + 1) * tile, j * tile:(j + 1) * tile][:, vv].astype(np.float32).mean(1)
                sc[i, j] = (pr.max() - pr.min()) / max(med, 1.0)
        np.save(qf, sc)
    t = np.nan_to_num(np.load(qf), nan=-1) >= QTHR
    return np.kron(t, np.ones((tile, tile), bool))


def zskip(z, tile=128):
    """tiles the lane skipped (fastink --mask): >= 90 % of the tile's valid pixels at the map's constant minimum"""
    valid = z != 0; sk = np.zeros_like(valid)
    if not valid.any(): return sk
    at_min = valid & (z == z[valid].min())
    for i in range(0, z.shape[0], tile):
        for j in range(0, z.shape[1], tile):
            v = valid[i:i + tile, j:j + tile]; n = v.sum()
            if n and at_min[i:i + tile, j:j + tile].sum() >= 0.9 * n: sk[i:i + tile, j:j + tile] = v
    return sk


root = sys.argv[1]
WIN = int(sys.argv[sys.argv.index('--win') + 1]) if '--win' in sys.argv else 2600
STEP = int(sys.argv[sys.argv.index('--step') + 1]) if '--step' in sys.argv else 1300
REDO = '--redo' in sys.argv          # recompute existing spectral_win.json
ALIGN_MIN = -99.0                    # angles for EVERY window (was >= 4.5): two-row text (1447 w058, 0841) scores ~0 and must still reach tri_rank's gates
for d in sorted(glob.glob(os.path.join(root, '*/'))):
    out = d + 'spectral_win.json'
    if (os.path.exists(out) and not REDO) or not os.path.exists(d + 'done_d0') or 'skipped' in open(d + 'done_d0').read():
        continue
    res, best = {}, (-9, None, None)
    rdir = os.path.join(os.path.dirname(os.path.dirname(d.rstrip('/'))), 'renders', os.path.basename(d.rstrip('/')) + '.zarr')
    ct = np.asarray(zarr.open(rdir, mode='r')['0'][15]).astype(np.float32) if os.path.exists(rdir) else None
    al_list = []
    for f in glob.glob(d + '*_d0_*_z.npy'):
        side = 'rev' if '_reverse_' in f else 'fwd'
        z = np.load(f).astype(np.float32); q = qmask_for(f)
        if q is None and os.path.exists(rdir):      # fast_loop never writes quality_tiles.npy: rebuild it from the render, else the
            q = render_qmask(rdir, d)                # skipped low-quality tiles (z ~ -2 blocks) score as "rows" (x_depth: 1218 fold 31.5 -> 1.0)
        if q is None:                                # no render either (maps pulled from a cloud lane): keep pixels outside the
            q = (z != 0) & ~zskip(z)                 # tiles the lane skipped, which sit at the map's constant minimum
        if q is not None and q.shape != z.shape:
            qp = np.zeros(z.shape, bool); h, w = min(q.shape[0], z.shape[0]), min(q.shape[1], z.shape[1]); qp[:h, :w] = q[:h, :w]; q = qp
        H, Wd = z.shape; xs = list(range(0, max(1, Wd - WIN + 1), STEP)) or [0]
        lst = []
        for x0 in xs:
            zz = z[:, x0:x0 + WIN]; qq = None if q is None else q[:H, x0:x0 + WIN]
            if (zz != 0).mean() < 0.3:
                continue
            try:
                s = score(zz, qmask=qq).get('score')
            except Exception:
                s = None
            if s is not None:
                if s >= ALIGN_MIN and ct is not None:
                    try:
                        sc = score(zz, qmask=qq); cc = ct[:, x0:x0 + WIN]; vv = cc > 0
                        if qq is not None:
                            vv &= cv2.resize(qq.astype(np.uint8), (vv.shape[1], vv.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                        al = alignment(cc, vv, sc['angle'])
                        dom = al['fibre_angles'][0] if al['fibre_angles'] else None
                        okal = dom is not None and min(abs(sc['angle'] - dom), 180 - abs(sc['angle'] - dom)) <= 15
                        zr = zz.astype(np.float32)
                        if zr.shape != cc.shape: zr = cv2.resize(zr, (cc.shape[1], cc.shape[0]))
                        vb = (cc > 0) & (zr != 0); sg = 1.5 * 1000 / 8.64
                        mb = cv2.GaussianBlur(vb.astype(np.float32), (0, 0), sg) + 1e-6
                        zl = cv2.GaussianBlur(zr * vb, (0, 0), sg) / mb; cl = cv2.GaussianBlur(cc * vb, (0, 0), sg) / mb
                        bc = float(np.corrcoef(zl[vb], cl[vb])[0, 1]) if vb.sum() > 1000 else None
                        al_list.append([side, x0, round(float(s), 2), sc['angle'], al['fibre_angles'], bool(okal), bc])
                    except Exception as e:
                        al_list.append([side, x0, round(float(s), 2), None, repr(e), False])
                lst.append([x0, round(float(s), 2)])
                if s > best[0]:
                    best = (round(float(s), 2), side, x0)
        res[side] = lst
    res['best'] = best
    res['align'] = al_list
    json.dump(res, open(out, 'w'))
    print(os.path.basename(d.rstrip('/')), best, flush=True)
