"""ink_full.py RENDER.zarr OUT_DIR [--c 20] [--um 8.64] [--faces fwd,rev] [--crop y0,y1,x0,x1]

Max-quality ink (x_zoom recipe, validated on the 0343 stroke labels: AP .693 vs .669 for the lane fusion) on a WHOLE
surface render: Hecate stride 16 (bf16) + Reader v2 and TRACE (VillaFast, overlap 0.5, 8-fold dihedral TTA), every
model averaged over d -1/0/+1, z-fused over valid pixels.  Writes per face: <face>_hec.npy, <face>_rv2.npy, <face>_tr.npy
(probabilities, float16), <face>_fused.npy (z, float16); plus ct<c>.png (CT layer c) and timing.json.
No super-resolution or generative step: every pixel is a model probability on measured CT.
"""
import os, sys, json, time, argparse
import numpy as np, cv2, zarr
cv2.setNumThreads(int(os.environ.get('OMP_NUM_THREADS', '4')))
W = '/home/liam/projects/vesuvius-work'
sys.path.insert(0, f'{W}/x_zoom')
import zoomlib as Z                                                   # noqa: E402

DEPTHS = (-1, 0, 1)


def recipe_fast(vol, c, um, face, log=print):
    """first-pass screen for large surfaces (~10x faster): Hecate stride 32, Reader v2 and TRACE without TTA, single depth d0"""
    valid = vol[c] > 0; out = {}; t = time.time()
    out['hec'] = Z.hecate_map(vol, um, c, 0, face, stride=32); log(f'  hec {time.time() - t:.0f}s'); t = time.time()
    out['rv2'] = Z.villa_map('rv2', vol, c, 0, face, overlap=0.5); log(f'  rv2 {time.time() - t:.0f}s'); t = time.time()
    out['tr'] = Z.villa_map('trace', vol, c, 0, face, overlap=0.5); log(f'  tr {time.time() - t:.0f}s')
    out['fused'] = Z.zfuse([out['hec'], out['rv2'], out['tr']], valid)
    return out


def recipe(vol, c, um, face, log=print):
    """vol (D,H,W) uint8, read layer c; face 'forward'|'reverse' -> dict of float32 maps (hec, rv2, tr, fused)"""
    valid = vol[c] > 0
    out = {}
    t = time.time()
    out['hec'] = np.mean([Z.hecate_map(vol, um, c, d, face, stride=16) for d in DEPTHS], 0); log(f'  hec {time.time() - t:.0f}s')
    t = time.time()
    out['rv2'] = np.mean([Z.villa_map('rv2', vol, c, d, face, overlap=0.5, tta8=True) for d in DEPTHS], 0); log(f'  rv2 {time.time() - t:.0f}s')
    t = time.time()
    out['tr'] = np.mean([Z.villa_map('trace', vol, c, d, face, overlap=0.5, tta8=True) for d in DEPTHS], 0); log(f'  tr {time.time() - t:.0f}s')
    out['fused'] = Z.zfuse([out['hec'], out['rv2'], out['tr']], valid)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('render'); ap.add_argument('out'); ap.add_argument('--c', type=int, default=20)
    ap.add_argument('--um', type=float, default=8.64); ap.add_argument('--faces', default='fwd')
    ap.add_argument('--crop', default='')
    ap.add_argument('--fast', action='store_true', help='first-pass recipe (Hecate s32, no TTA, d0) for large surfaces')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    arr = zarr.open(a.render, mode='r')['0']
    if a.crop:
        y0, y1, x0, x1 = map(int, a.crop.split(',')); vol = np.asarray(arr[:, y0:y1, x0:x1])
    else:
        vol = np.asarray(arr[:])
    print(a.render, vol.shape, flush=True)
    cv2.imwrite(f'{a.out}/ct{a.c}.png', vol[a.c])
    timing = {}
    for f in a.faces.split(','):
        if os.path.exists(f'{a.out}/{f}_fused.npy'):
            continue
        t0 = time.time(); print(f'face {f}', flush=True)
        m = (recipe_fast if a.fast else recipe)(vol, a.c, a.um, 'forward' if f == 'fwd' else 'reverse', log=lambda s: print(s, flush=True))
        for k, v in m.items():
            np.save(f'{a.out}/{f}_{k}.npy', v.astype(np.float16))
        timing[f] = round(time.time() - t0, 1); print(f'face {f} done {timing[f]}s', flush=True)
    json.dump(dict(shape=list(vol.shape), timing=timing), open(f'{a.out}/timing.json', 'w'))
