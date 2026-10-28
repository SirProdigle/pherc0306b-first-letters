"""ink_lane.py RENDERS_DIR MAPS_DIR UM [--post-done FILE] [--vram-gb 12] [--max N] : lane-local ink loop (the fast_loop.py
recipe, without the shared x_recipe_erwin/hr_globs.txt so it can't pick up another lane's renders).

GPU-resident Hecate (stride 32, bf16, no TTA) + a villa ink_9um-format model (VillaFast, 17 layers), both faces, d=0, plain
z-fusion 1:1, off-sheet tiles skipped (--mask). Per scan resolution (coordinator 2026-10-11): 9.36 um -> Hecate + TRACE v3B;
8.64 um -> Hecate + Reader v2 ranked, TRACE map saved at weight 0.
Inks every RENDERS_DIR/<name>.zarr that has .done into MAPS_DIR/<name>/ (<name>_d0_{fwd,reverse}_{z.npy,overlay.png}, per-model
PNGs, timing, done_d0). Exits once --post-done exists and every finished render is inked (or after --max renders)."""
import argparse, glob, json, os, sys, time
from pathlib import Path
W = Path('/home/liam/projects/vesuvius-work')
sys.path.insert(0, str(W / 'x_research2/speed'))
import cv2
cv2.setNumThreads(int(os.environ.get('OMP_NUM_THREADS', '4')))
import fastpipe as FP

ap = argparse.ArgumentParser()
ap.add_argument('renders'); ap.add_argument('maps'); ap.add_argument('um', type=float)
ap.add_argument('--post-done', default=''); ap.add_argument('--vram-gb', type=float, default=float(os.environ.get('FAST_VRAM_GB', '12')))
ap.add_argument('--max', type=int, default=0)
ap.add_argument('--face-from', nargs=2, metavar=('FLAT_DIR', 'UMB_JSON'), default=None,
                help='ink only the reading face of <tag>_<wNNN>.zarr, judged from FLAT_DIR/<wNNN>/tifxyz/flatten.tifxyz and the umbilicus')
ap.add_argument('--face-min', type=float, default=0.8, help='share of agreeing vertices needed to drop the other face')
a = ap.parse_args()


def reading_face(tifxyz, umb_json):
    """'reverse' if the render's grid normal (x[c+1]-x[c-1]) x (y[r+1]-y[r-1]) (= increasing vc_render_tifxyz layer) points away
    from the umbilicus, else 'forward'; plus the share of vertices agreeing.  The ink models read only when their layer order
    increases towards the scroll centre (the recto faces the umbilicus): on PHerc0343 the other face scores pixel AUC .54-.64
    vs .86-.87 (x_recto/reading_face.py)."""
    import numpy as np, tifffile
    P = np.stack([tifffile.imread(f'{tifxyz}/{c}.tif') for c in 'xyz'], -1).astype(np.float64)
    P[P[..., 0] == -1] = np.nan
    n = np.cross(P[1:-1, 2:] - P[1:-1, :-2], P[2:, 1:-1] - P[:-2, 1:-1])
    d = json.load(open(umb_json)); cp = d['control_points'] if isinstance(d, dict) else d
    cp = sorted(((p['z'], p['x'], p['y']) for p in cp) if isinstance(cp[0], dict) else ((p[2], p[0], p[1]) for p in cp))
    uz, ux, uy = (np.array(v, float) for v in zip(*cp))
    q = P[1:-1, 1:-1]
    rad = np.stack([q[..., 0] - np.interp(q[..., 2], uz, ux), q[..., 1] - np.interp(q[..., 2], uz, uy), np.zeros(q.shape[:2])], -1)
    s = np.sign((n * rad).sum(-1)); s = s[np.isfinite(s) & (s != 0)]
    out = float((s > 0).mean()) if s.size else 0.5
    return ('reverse', out) if out >= 0.5 else ('forward', 1 - out)


def dirs_for(render):
    if not a.face_from:
        return ('forward', 'reverse')
    w = Path(render).name[:-5].rsplit('_', 1)[-1]
    t = Path(a.face_from[0]) / w / 'tifxyz' / 'flatten.tifxyz'
    try:
        face, share = reading_face(t, a.face_from[1])
    except Exception as e:
        print('face: fallback to both faces', w, repr(e)[:200], flush=True); return ('forward', 'reverse')
    print(f'face {w}: {face} ({share:.3f})', flush=True)
    return (face,) if share >= a.face_min else ('forward', 'reverse')


FP.CK['trace'] = W / 'models/community/trace/v3B_best.pth'
FP.RECIPES['all3'] = dict(models=['hec', 'rv2', 'trace'], weights={'hec': 1.0, 'rv2': 1.0, 'trace': 1.0}, hec_stride=32,
                          hec_depths=[0], rv2_windows=[0], tta='none', v8=False)
models, weights = (['hec', 'trace'], {'hec': 1.0, 'trace': 1.0, 'rv2': 0.0}) if a.um > 9 else \
                  (['hec', 'rv2', 'trace'], {'hec': 1.0, 'rv2': 1.0, 'trace': 0.0})
pipe = FP.Pipe('all3', vram_gb=a.vram_gb)
pipe.cfg['models'], pipe.cfg['weights'] = models, weights
_cap = int(os.environ.get('INK_MAX_BATCH') or 0)       # small-VRAM runs (local dry run on a shared GPU); boxes leave it unset
if _cap:
    for _m in pipe.M.values():
        def _capped(*args, _f=_m.predict, **kw):
            if 'batch' in kw: kw['batch'] = min(kw['batch'], _cap)
            return _f(*args, **kw)
        _m.predict = _capped
n = 0; tries = {}
while True:
    todo = sorted((r for r in glob.glob(os.path.join(a.renders, '*.zarr')) if os.path.exists(r + '/.done')), key=os.path.getmtime)
    left = [r for r in todo if not (Path(a.maps) / Path(r).name[:-5] / 'done_d0').exists() and tries.get(r, 0) < 2]   # 2 tries per render
    if not left:
        if a.post_done and os.path.exists(a.post_done) or (a.max and n >= a.max):
            print('INK_DONE', n, flush=True); break
        time.sleep(30); continue
    for r in left:
        name = Path(r).name[:-5]; out = Path(a.maps) / name; out.mkdir(parents=True, exist_ok=True); t0 = time.time()
        try:
            pipe.run(r, str(out), d=0, dirs=dirs_for(r), um=a.um, mask=True)
        except Exception as e:
            tries[r] = tries.get(r, 0) + 1; import torch; torch.cuda.empty_cache()
            print('FAIL', name, f'try {tries[r]}', repr(e)[:300], flush=True); (out / 'ink_failed').write_text(repr(e)[:2000]); time.sleep(10); continue
        (out / f'{name}.rv2done').write_text(f'{time.time() - t0:.0f}s\n')
        (out / 'done_d0').write_text('+'.join(m for m in models if weights[m] > 0) + ' fast\n')
        n += 1; print(name, f'{time.time() - t0:.0f}s', flush=True)
        if a.max and n >= a.max: break
