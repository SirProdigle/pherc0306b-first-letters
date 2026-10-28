"""smoke_v7.py : a tiny CPU re-inference of the version-7 model, checked against the shipped claim map.

  smoke_v7.py make RENDER.zarr OUT.npz [--y0 2110 --x0 4495 --n 5]     (host: cut the CT crop that ships in the Docker image)
  smoke_v7.py run CROP.npz --weights W.safetensors --sha SHA --map fwd_fin.npy --tta 1|4 [--out result.json]

The crop is the 24 layers v8-in reads (render layers 8..31, stored order, clipped at 200) of a window of (n + 1) x (n + 1) stitch-grid
tiles inside the claim box: tile tops on the claim map's own grid (origin y 115, x 694, stride 21), 64 + 21 n px square.  'run' loads the
weights (sha256-checked) into v8-in's InkDetector, runs every grid tile of the window on the CPU in fp32 with the same flips, window
(+-3 sigma Gaussian) and weight-sum stitch as code/ink_v8.py --nsig 3, and compares the pixels whose every covering tile lies inside
the window (so their tile set equals the full run's) with the shipped map (made on a GPU in fp16).  PASS = max |diff| <= 0.02 and r >= 0.999."""
import os, sys, json, time, argparse, hashlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
GY, GX, S, TILE = 115, 694, 21, 64


def make(render, out, y0, x0, n):
    import zarr
    assert (y0 - GY) % S == 0 and (x0 - GX) % S == 0, 'crop origin must be on the claim map grid'
    sz = TILE + S * n
    vol = np.asarray(zarr.open(render, mode='r')['0'][8:32, y0:y0 + sz, x0:x0 + sz])
    st = np.ascontiguousarray(np.minimum(vol, 200).transpose(1, 2, 0)).astype(np.uint8)
    np.savez_compressed(out, st=st, y0=y0, x0=x0, n=n, render=render, layers='8..31 stored order (asc), clip 200')
    print(out, st.shape, 'CT coverage on layer 20:', float((vol[12] > 0).mean()))


def run(crop, weights, want, mapf, tta, out):
    import torch
    sys.path.insert(0, HERE); os.environ.setdefault('INK_DEVICE', 'cpu')
    import ink_v8 as K                                                               # the claim recipe's own functions (positions, part_sums)
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    h = hashlib.sha256(open(weights, 'rb').read()).hexdigest()
    if h != want: raise SystemExit(f'weights sha256 {h} != {want}')
    if K.sha256(f'{K.V8}/model.safetensors') != K.V8_SHA: raise SystemExit('v8-in base weights sha256 mismatch')
    from safetensors.torch import load_file
    model = K.InkDetector.from_pretrained(K.V8); model.load_state_dict(load_file(weights), strict=True); model = model.eval()
    z = np.load(crop); st = z['st']; y0, x0, n = int(z['y0']), int(z['x0']), int(z['n'])
    tops = np.arange(0, S * n + 1, S); pts = np.array([(a, b) for a in tops for b in tops])
    flips = ['id'] if tta == 1 else list(K.FLIPS)
    t = time.time(); H, Wd = st.shape[:2]; num = np.zeros((H, Wd), np.float32); den = np.zeros((H, Wd), np.float32)
    for fl in flips:
        yb, nu, de = K.part_sums(model, st, pts, fl, batch=4, device='cpu', nsig=3)
        num[yb:yb + nu.shape[0]] += nu; den[yb:yb + de.shape[0]] += de
    m = num / np.maximum(den, 1e-9)
    c0, c1 = TILE - 1, S * n + 1                                                     # pixels covered only by tiles inside the window
    mine = m[c0:c1, c0:c1]
    ref = np.load(mapf, mmap_mode='r')[y0 + c0:y0 + c1, x0 + c0:x0 + c1].astype(np.float32)
    ok = np.isfinite(ref); d = np.abs(mine - ref)[ok]
    r = float(np.corrcoef(mine[ok], ref[ok])[0, 1]) if ok.sum() > 2 else float('nan')
    res = dict(weights_sha256=h, tiles=int(len(pts)), flips=flips, compared_px=int(ok.sum()), window_render_px=[y0 + c0, y0 + c1, x0 + c0, x0 + c1],
               max_abs_diff=round(float(d.max()), 5), mean_abs_diff=round(float(d.mean()), 6), pearson_r=round(r, 6),
               map_range=[round(float(ref[ok].min()), 4), round(float(ref[ok].max()), 4)], cpu_seconds=round(time.time() - t, 1))
    res['verdict'] = 'PASS' if res['max_abs_diff'] <= 0.02 and r >= 0.999 else 'FAIL'
    print(json.dumps(res, indent=1), flush=True)
    if out: json.dump(res, open(out, 'w'), indent=1)
    return res['verdict'] == 'PASS'


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=('make', 'run')); ap.add_argument('a'); ap.add_argument('b', nargs='?')
    ap.add_argument('--y0', type=int, default=2110); ap.add_argument('--x0', type=int, default=4495); ap.add_argument('--n', type=int, default=5)
    ap.add_argument('--weights'); ap.add_argument('--sha'); ap.add_argument('--map'); ap.add_argument('--tta', type=int, default=4)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    if a.mode == 'make': make(a.a, a.b, a.y0, a.x0, a.n)
    else: sys.exit(0 if run(a.a, a.weights, a.sha, a.map, a.tta, a.out) else 1)
