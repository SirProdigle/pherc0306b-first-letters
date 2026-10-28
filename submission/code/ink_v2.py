"""ink_v2.py RENDER.zarr OUT_DIR [--models rv2,tr] [--crop y0,y1,x0,x1] : the claim recipe on the v2 mesh's render
(lane I2 copy of lane P1's ink_p1.py).  Only change: the repo root comes from FL_ROOT (Docker: /opt/fl)
and the device from INK_DEVICE; the arithmetic, tiling and window are P1's, so the maps are the ones the lane made.

Recipe = x_deepread/ink_full.py's Reader v2 and TRACE members: villa inference (VillaFast), overlap 0.5, 8-fold dihedral TTA,
each model averaged over depths d -1 / 0 / +1 around render layer 20 (the mesh surface), reading face 'forward' (+n).
Tiling: the render is read in row tiles (--tile rows + 192 px context each side), stitched at the tile centres; with --crop
only that window (+ 192 px context) is inked and the output is the full-size raster with NaN outside the window.
Default window = the claim image's context u -30..+30 x v -20..+15 mm = render px y 116..4167, x 694..7639.
VRAM: capped at --vram GB (torch per-process memory fraction); a CUDA OOM retries the tile at half the batch.
Writes fwd_<m>.npy (float16 probabilities, restartable: finished models are skipped), ct<c>.png (layer c), timing.json."""
import os, sys, json, time, argparse
import numpy as np, cv2, zarr, torch
cv2.setNumThreads(int(os.environ.get('OMP_NUM_THREADS', '4')))
W = os.environ.get('FL_ROOT', '/home/liam/projects/vesuvius-work')
sys.path.insert(0, f'{W}/x_zoom')
import zoomlib as Z                                                   # noqa: E402

DEPTHS = (-1, 0, 1)
CTX = 192
CONTEXT_CROP = '116,4167,694,7639'


def run_tile(m, vol, c, um, batch):
    while True:
        try:
            if m == 'hec':
                return np.mean([Z.hecate_map(vol, um, c, d, 'forward', stride=16, batch=batch) for d in DEPTHS], 0)
            return np.mean([Z.villa_map({'rv2': 'rv2', 'tr': 'trace'}[m], vol, c, d, 'forward', overlap=0.5, tta8=True, batch=batch) for d in DEPTHS], 0)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch <= 1: raise
            batch //= 2; print(f'    OOM -> batch {batch}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('render'); ap.add_argument('out'); ap.add_argument('--c', type=int, default=20)
    ap.add_argument('--models', default='rv2,tr'); ap.add_argument('--um', type=float, default=8.64)
    ap.add_argument('--vram', type=float, default=7.5); ap.add_argument('--tile', type=int, default=1400); ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--crop', default=CONTEXT_CROP, help='y0,y1,x0,x1 render px ("" = whole render)')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    arr = zarr.open(a.render, mode='r')['0']; D, H, Wd = arr.shape
    if not os.path.exists(f'{a.out}/ct{a.c}.png'): cv2.imwrite(f'{a.out}/ct{a.c}.png', np.asarray(arr[a.c]))
    todo = [m for m in a.models.split(',') if not os.path.exists(f'{a.out}/fwd_{m}.npy')]
    if not todo: print('done already', a.out); sys.exit(0)
    if torch.cuda.is_available():
        tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram / tot))
        print(a.render, arr.shape, todo, f'vram cap {a.vram} GB of {tot:.1f}', flush=True)
    tim = json.load(open(f'{a.out}/timing.json')) if os.path.exists(f'{a.out}/timing.json') else {}
    cy0, cy1, cx0, cx1 = (int(x) for x in a.crop.split(',')) if a.crop else (0, H, 0, Wd)
    for m in todo:
        t = time.time(); out = np.zeros((H, Wd), np.float32) if not a.crop else np.full((H, Wd), np.nan, np.float32)
        q0, q1 = max(0, cx0 - CTX), min(Wd, cx1 + CTX)
        for y0 in range(cy0, cy1, a.tile):
            y1 = min(cy1, y0 + a.tile); r0 = max(0, y0 - CTX); r1 = min(H, y1 + CTX)
            vol = np.asarray(arr[:, r0:r1, q0:q1])
            if not (vol[a.c] > 0).any(): continue
            p = run_tile(m, vol, a.c, a.um, a.batch)
            out[y0:y1, cx0:cx1] = p[y0 - r0:y1 - r0, cx0 - q0:cx1 - q0]
            mem = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0
            print(f'  {m} rows {y0}-{y1}  {time.time() - t:.0f}s  max mem {mem:.1f} GB', flush=True)
        np.save(f'{a.out}/fwd_{m}.tmp.npy', out.astype(np.float16)); os.replace(f'{a.out}/fwd_{m}.tmp.npy', f'{a.out}/fwd_{m}.npy')
        tim[m] = round(time.time() - t, 1); print(f'  {m} {tim[m]}s', flush=True)
        json.dump(dict(shape=[D, H, Wd], timing=tim, crop=a.crop, recipe='x_deepread/ink_full.py members, tta8, d-1..+1, face fwd, row tiles + 192 px context'),
                  open(f'{a.out}/timing.json', 'w'))
