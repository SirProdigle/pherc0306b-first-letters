"""v8q_known_text.py OUT_DIR [SET ...] : v8Q on the known-text windows of I1's control panels (lane I2).  GPU, under the lock.
Protocol = lane P2's no-harm protocol for v8-in models: stride 21, d0, identity flip only (no TTA), weight-normalised Gaussian stitch,
tiles admitted at >= 25 % CT coverage (as ink_v8.py), 24 layers c-12..c+11 fed in the training orientation of each set (desc for
these renders: P2 eval/depth_profile.json).  Each window carries a 96 px margin so the window itself has full tile support.
Sets (window = I1's panel window in its source frame; Reader v2 / TRACE maps of the same pixels are I1's):
  w058   PHerc1447 w058 positive-control render x_gates/.../pc1447_w058.zarr (= the crop y 2576:4512, x 389:4748), c 15, all of it (v8Q TRAINED on w058)
  p0343  PHerc0343 fl/0343/concat_66.zarr, c 33, the traced PANTOGINON crop y 450:2350, x 2400:5300            (v8Q TRAINED on 0343 tracings)
  lane   PHerc0343 lane render fl/PHerc0343/screen2/renders/pc343_w048.zarr, c 15, y 575:2311, x 230:4859    (not in v8Q's training crop)
  w060   PHerc1447 w060 HF layer stack (O1 download), c 12, I1's window y 1558:3294, x 1248:4321                 (held out from v8Q)
Writes OUT_DIR/v8q_<set>.npy (window + margin, float16) and v8q_<set>.json."""
import os, sys, json, time
import numpy as np, zarr, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ink_v8 as V                                                                   # noqa: E402
W = os.environ.get('FL_ROOT', '/home/liam/projects/vesuvius-work')
SETS = {'w058': (f'{W}/x_gates/PHerc1447/renders/pc1447_w058.zarr', 15, (0, 1936, 0, 4359)),       # this render IS the val1447 crop (2576:4512, 389:4748)
        'p0343': (f'{W}/fl/0343/concat_66.zarr', 33, (450, 2350, 2400, 5300)),
        'lane': (f'{W}/fl/PHerc0343/screen2/renders/pc343_w048.zarr', 15, (575, 2311, 230, 4859)),
        'w060': (f'HF:{W}/x_refine/trials/O1_overlabel/data/hf/w060', 12, (1558, 3294, 1248, 4321))}   # HF layer stack 00..23 (held out from v8Q)
CKPT = f'{W}/x_refine/trials/P2_inkft_0306B/ckpt/v8Q/final.safetensors'
MARGIN = 96


def main():
    out = sys.argv[1]; sets = sys.argv[2:] or list(SETS); os.makedirs(out, exist_ok=True)
    dev = os.environ.get('INK_DEVICE', 'cuda') if torch.cuda.is_available() else 'cpu'; tiny = os.environ.get('KT_TINY') == '1'   # tiny: smoke test
    if dev == 'cuda':
        tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 30; torch.cuda.set_per_process_memory_fraction(min(1.0, 7.5 / tot))
    from safetensors.torch import load_file
    s = V.sha256(CKPT); assert s == V.V8Q_SHA, s
    model = V.InkDetector.from_pretrained(V.V8); model.load_state_dict(load_file(CKPT), strict=True); model = model.to(dev).eval()
    for name in sets:
        if os.path.exists(f'{out}/v8q_{name}.npy'): print('have', name); continue
        path, c, (y0, y1, x0, x1) = SETS[name]; t = time.time()
        if tiny: y1, x1 = y0 + 40, x0 + 40
        if path.startswith('HF:'):                                                  # a stack of layer tifs (HF download), 24 layers = c-12..c+11
            import tifffile
            src = path[3:]; H, Wd = tifffile.imread(f'{src}/layers/00.tif').shape
            Y0, Y1, X0, X1 = max(0, y0 - MARGIN), min(H, y1 + MARGIN), max(0, x0 - MARGIN), min(Wd, x1 + MARGIN)
            full = np.stack([tifffile.imread(f'{src}/layers/{i:02d}.tif')[Y0:Y1, X0:X1] for i in range(24)])
            vol = full[::-1]; cov = full[c] > 0
        else:
            a = zarr.open(path, mode='r')['0']; D, H, Wd = a.shape
            Y0, Y1, X0, X1 = max(0, y0 - MARGIN), min(H, y1 + MARGIN), max(0, x0 - MARGIN), min(Wd, x1 + MARGIN)
            vol = np.asarray(a[c - 12:c + 12, Y0:Y1, X0:X1])[::-1]                 # desc: training orientation of these renders
            cov = np.asarray(a[c, Y0:Y1, X0:X1]) > 0
        st = np.ascontiguousarray(np.minimum(vol, 200).transpose(1, 2, 0)).astype(np.uint8)
        pts = V.positions(cov, 0.25); pts = pts[np.argsort(pts[:, 0], kind='stable')]
        yb, num, den = V.part_sums(model, st, pts, 'id', 4, dev)
        nu = np.zeros(cov.shape, np.float32); de = np.zeros(cov.shape, np.float32); nu[yb:yb + num.shape[0]] += num; de[yb:yb + den.shape[0]] += den
        m = np.where(de > 1e-9, nu / np.maximum(de, 1e-9), np.nan).astype(np.float16)
        np.save(f'{out}/v8q_{name}.npy', m)
        json.dump(dict(set=name, render=path, c=c, order='desc', window=[y0, y1, x0, x1], crop_with_margin=[Y0, Y1, X0, X1], tiles=int(len(pts)),
                       recipe='v8Q, stride 21, d0, identity flip (no TTA), weight-normalised g1 stitch, tiles >= 25 % CT', weights_sha256=s,
                       seconds=round(time.time() - t, 1)), open(f'{out}/v8q_{name}.json', 'w'), indent=1)
        print(name, len(pts), 'tiles', round(time.time() - t), 's', flush=True)


if __name__ == '__main__':
    main()
