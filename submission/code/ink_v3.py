"""ink_v3.py RENDER.zarr OUT_DIR [--crop y0,y1,x0,x1] [--max-min 25] : Reader V3 with lead A's v2 claim recipe (v3 package).

Copy of CANDIDATES/SUBMISSION_v2_PHerc0306B/code/ink_v2.py's arithmetic: zoomlib.villa_map (VillaFast, overlap 0.5, 8-fold dihedral
TTA = rot90 x mirror4), averaged over depths d -1 / 0 / +1 around render layer 20, face 'forward', row tiles + 192 px context, stitched
at the tile centres; NaN outside the crop.  Only change: the model is Reader V3 (models/community/reader-v3/reader-v3.pth, sha256
checked, loaded with torch.load(weights_only=True); villa's own loader would use weights_only=False).  Restartable per row tile
(parts in OUT_DIR/v3parts/), exits 3 after --max-min minutes so a GPU hold stays < 30 min.  Writes OUT_DIR/fwd_v3.npy (float16)."""
import os, sys, json, time, hashlib, argparse
import numpy as np, zarr, torch

W = os.environ.get('FL_ROOT', '/home/liam/projects/vesuvius-work')
sys.path.insert(0, f'{W}/x_zoom')
import zoomlib as Z                                                   # noqa: E402

CK = os.environ.get('READER_V3', f'{W}/models/community/reader-v3/reader-v3.pth')          # run.sh links a MODELS_DIR copy here
SHA = 'ea405417450abd8ace1b8ff79ebce3adb646c5e48ab95da46844d77efe984d8a'
DEPTHS = (-1, 0, 1); CTX = 192; CONTEXT_CROP = '116,4167,694,7639'


def load_v3():
    h = hashlib.sha256()
    with open(CK, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    if h.hexdigest() != SHA: raise SystemExit('reader-v3.pth sha256 mismatch')
    from vesuvius.ink_detection.inference import infer as I
    I.load_checkpoint = lambda path: torch.load(str(path), map_location='cpu', weights_only=True)
    Z.CK['v3'] = CK
    Z._M['v3'] = Z.FI.VillaFast(CK, compile=None)
    print('v3 loaded, patch', Z._M['v3'].cfg.patch_size, flush=True)


def run_tile(vol, c, batch):
    while True:
        try:
            return np.mean([Z.villa_map('v3', vol, c, d, 'forward', overlap=0.5, tta8=True, batch=batch) for d in DEPTHS], 0)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch <= 1: raise
            batch //= 2; print(f'    OOM -> batch {batch}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('render'); ap.add_argument('out'); ap.add_argument('--c', type=int, default=20)
    ap.add_argument('--vram', type=float, default=6.0); ap.add_argument('--tile', type=int, default=1400); ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--crop', default=CONTEXT_CROP); ap.add_argument('--max-min', type=float, default=25)
    a = ap.parse_args(); os.makedirs(f'{a.out}/v3parts', exist_ok=True); t0 = time.time()
    if os.path.exists(f'{a.out}/fwd_v3.npy'): print('done already'); sys.exit(0)
    arr = zarr.open(a.render, mode='r')['0']; D, H, Wd = arr.shape
    tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram / tot))
    load_v3()
    cy0, cy1, cx0, cx1 = (int(x) for x in a.crop.split(','))
    q0, q1 = max(0, cx0 - CTX), min(Wd, cx1 + CTX)
    for y0 in range(cy0, cy1, a.tile):
        part = f'{a.out}/v3parts/rows_{y0}.npy'
        if os.path.exists(part): continue
        if (time.time() - t0) / 60 > a.max_min: print('time cap: exit 3', flush=True); sys.exit(3)
        y1 = min(cy1, y0 + a.tile); r0 = max(0, y0 - CTX); r1 = min(H, y1 + CTX)
        vol = np.asarray(arr[:, r0:r1, q0:q1])
        p = run_tile(vol, a.c, a.batch) if (vol[a.c] > 0).any() else np.zeros((r1 - r0, q1 - q0), np.float32)
        np.save(part, p[y0 - r0:y1 - r0, cx0 - q0:cx1 - q0].astype(np.float16))
        print(f'  v3 rows {y0}-{y1}  {time.time() - t0:.0f}s  max mem {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB', flush=True)
    out = np.full((H, Wd), np.nan, np.float32)
    for y0 in range(cy0, cy1, a.tile):
        y1 = min(cy1, y0 + a.tile); out[y0:y1, cx0:cx1] = np.load(f'{a.out}/v3parts/rows_{y0}.npy').astype(np.float32)
    np.save(f'{a.out}/fwd_v3.tmp.npy', out.astype(np.float16)); os.replace(f'{a.out}/fwd_v3.tmp.npy', f'{a.out}/fwd_v3.npy')
    json.dump(dict(shape=[D, H, Wd], crop=a.crop, seconds=round(time.time() - t0, 1), model='reader-v3.pth sha256 ' + SHA,
                   recipe='zoomlib.villa_map tta8 (rot90 x mirror4), d-1..+1 around c, face fwd, overlap 0.5, row tiles + 192 px context'),
              open(f'{a.out}/fwd_v3.json', 'w'))
    print('done', a.out, flush=True)
