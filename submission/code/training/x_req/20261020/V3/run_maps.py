"""run_maps.py MODEL[,MODEL] JOB[,JOB] : Reader V3 vs Reader v2 benchmark maps (x_req/20261020/READER_V3_PLAN.md, OPERATIONAL.md).

MODEL: rv2 | v3 | trace | v3s (V3, single pass: d0, no TTA).   JOB: c841 | w058 | w060 | art (the four PHerc0813 atlas windows,
both faces) | face (w058 + w060, d0, no TTA, forward and reverse: which face the model prefers).
After the requested models/jobs it always runs the coordinator's supplementary queue (SUPP below).
Identical recipe for both models: VillaFast, 17-layer window centred on c (s = c + d - 8), d in {-1, 0, 1} averaged, mirror4 TTA,
overlap 0.5, batch 8.  Skips outputs that exist; stops after ~25 min (exit 3) so each GPU-lock hold stays under 30 min.
Checkpoints: sha256-checked, then loaded with torch.load(weights_only=True) (villa's own loader uses weights_only=False).
Run under the GPU lock (see gpu.sh)."""
import sys, os, json, time, hashlib
import numpy as np, torch, zarr, tifffile, cv2

W = '/home/liam/projects/vesuvius-work'; OUT = f'{W}/x_req/20261020/V3/maps'; os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, f'{W}/x_research2/speed')
CK = {'rv2': (f'{W}/models/community/reader-v2/reader-v2-step040000.pth', None),
      'v3': (f'{W}/models/community/reader-v3/reader-v3.pth', 'ea405417450abd8ace1b8ff79ebce3adb646c5e48ab95da46844d77efe984d8a'),
      'trace': (f'{W}/models/community/trace/v3B_best.pth', '812ffe2b42b3293974052ebbee60a0fe939ca2bfc3f03a517a0bc2e2339e5da2')}
CK['v3s'] = CK['v3']
RECIPE = {'rv2': ((-1, 0, 1), 'mirror4'), 'v3': ((-1, 0, 1), 'mirror4'), 'trace': ((-1, 0, 1), 'mirror4'), 'v3s': ((0,), 'none')}
SUPP = [('trace', 'c841,w058,w060,art'), ('v3s', 'c841,w058,w060,art'), ('rv2', 'face'), ('v3', 'face')]
HF = {'w058': f'{W}/data/pherc1447/w058', 'w060': f'{W}/x_refine/trials/O1_overlabel/data/hf/w060'}
ATL = f'{W}/x_atlas/dl'
ART = [('z5888_w020', 'PHerc0813_z5888_w020', 160, 2760), ('z7696_w020', 'PHerc0813_z7696_w020', 0, 2020),
       ('z13088_w040', 'PHerc0813_z13088_w040', 0, 2600), ('z12496_w060', 'PHerc0813_z12496_w060', 2660, 5260)]
DEPTHS = (-1, 0, 1)
T0 = time.time()


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()


def load(model):
    import fastink as FI
    from vesuvius.ink_detection.inference import infer as I
    p, want = CK[model]; got = sha(p)
    if want and got != want: raise SystemExit(f'{model}: sha256 mismatch {got}')
    I.load_checkpoint = lambda path: torch.load(str(path), map_location='cpu', weights_only=True)   # safe load
    vf = FI.VillaFast(p, compile=None, device='cuda')
    vf.model.eval(); vf.fwd = vf.model
    print(f'{model}: sha256 {got[:12]}, patch {vf.cfg.patch_size}, amp {vf.amp}', flush=True)
    return vf


def map3(vf, vol, c, face, recipe=None):
    """vol (D, H, W) uint8; mean over the recipe's depths (default DEPTHS, mirror4) of the map, float32 in [0, 1]."""
    depths, tta = recipe or (DEPTHS, 'mirror4')
    acc = []
    for d in depths:
        s = c + d - 8; idx = np.arange(s, s + 17)
        assert s >= 0 and idx[-1] < vol.shape[0], (s, vol.shape)
        if face == 'reverse': idx = idx[::-1]
        w = np.ascontiguousarray(vol[idx])
        a, _ = vf.predict(None, 0, window=(w, w), overlap=0.5, batch=8, tta=tta)
        acc.append(a.astype(np.float32) / 255.)
    return np.mean(acc, 0)


def done(m, j):
    names = {'c841': [f'{m}_c841'], 'w058': [f'{m}_w058'], 'w060': [f'{m}_w060'],
             'art': [f'{m}_art_{t}_{f}' for t, *_ in ART for f in ('forward', 'reverse')],
             'face': [f'{m}_face_{w}_{f}' for w in ('w058', 'w060') for f in ('forward', 'reverse')]}[j]
    return all(os.path.exists(f'{OUT}/{n}.npy') for n in names)


def bbox(m, pad):
    ys, xs = np.nonzero(m); H, Wd = m.shape
    return (max(0, ys.min() - pad), min(H, ys.max() + 1 + pad), max(0, xs.min() - pad), min(Wd, xs.max() + 1 + pad))


def save(name, p, meta):
    np.save(f'{OUT}/{name}.npy', p.astype(np.float16)); json.dump(meta, open(f'{OUT}/{name}.json', 'w'))


def job(vf, model, j):
    rc = RECIPE[model]; meta_r = dict(depths=list(rc[0]), tta=rc[1])
    if j == 'face':
        for w in ('w058', 'w060'):
            msk = cv2.imread(f'{HF[w]}/labels/mask.png', 0) > 0
            y0, y1, x0, x1 = bbox(msk, 64); vol = None
            for face in ('forward', 'reverse'):
                out = f'{model}_face_{w}_{face}'
                if os.path.exists(f'{OUT}/{out}.npy'): continue
                if vol is None: vol = np.stack([tifffile.imread(f'{HF[w]}/layers/{i:02d}.tif')[y0:y1, x0:x1] for i in range(24)])
                save(out, map3(vf, vol, 12, face, ((0,), 'none')), dict(src=HF[w], box=[int(y0), int(y1), int(x0), int(x1)], c=12, face=face,
                                                                       depths=[0], tta='none'))
        return
    if j == 'c841':
        out = f'{model}_c841'
        if os.path.exists(f'{OUT}/{out}.npy'): return
        vol = np.asarray(zarr.open(f'{W}/x_gates/PHerc0841/renders/pc841_w00.zarr', mode='r')['0'][:])
        save(out, map3(vf, vol, 15, 'reverse', rc), dict(src='x_gates/PHerc0841/renders/pc841_w00.zarr', c=15, face='reverse', **meta_r))
    elif j in ('w058', 'w060'):
        out = f'{model}_{j}'
        if os.path.exists(f'{OUT}/{out}.npy'): return
        msk = cv2.imread(f'{HF[j]}/labels/mask.png', 0) > 0
        y0, y1, x0, x1 = bbox(msk, 64)
        vol = np.stack([tifffile.imread(f'{HF[j]}/layers/{i:02d}.tif')[y0:y1, x0:x1] for i in range(24)])
        if vol.dtype != np.uint8: raise SystemExit(f'{j}: layers are {vol.dtype}, expected uint8')
        save(out, map3(vf, vol, 12, 'reverse', rc), dict(src=HF[j], box=[int(y0), int(y1), int(x0), int(x1)], c=12, face='reverse', **meta_r))
    elif j == 'art':
        for tag, d, xa, xb in ART:
            for face in ('forward', 'reverse'):
                out = f'{model}_art_{tag}_{face}'
                if os.path.exists(f'{OUT}/{out}.npy'): continue
                if time.time() - T0 > 25 * 60: print('chunk end', flush=True); sys.exit(3)
                z = [x for x in os.listdir(f'{ATL}/{d}/surface-volumes') if x.endswith('.zarr')][0]
                vol = np.asarray(zarr.open(f'{ATL}/{d}/surface-volumes/{z}', mode='r')['0'][:, :, xa:xb])
                save(out, map3(vf, vol, 15, face, rc), dict(src=f'{ATL}/{d}', x=[xa, xb], c=15, face=face, **meta_r))
                print(f'  {out} {time.time() - T0:.0f}s', flush=True)


if __name__ == '__main__':
    todo = [(m, sys.argv[2]) for m in sys.argv[1].split(',')] + SUPP
    for m, js in todo:
        vf = None
        for j in js.split(','):
            if time.time() - T0 > 25 * 60: print('chunk end', flush=True); sys.exit(3)
            if done(m, j): continue
            vf = vf or load(m)
            t = time.time(); job(vf, m, j); print(f'{m} {j} done {time.time() - t:.0f}s', flush=True)
        del vf; torch.cuda.empty_cache()
    print('all done', flush=True)
