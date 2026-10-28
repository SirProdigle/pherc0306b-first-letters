"""infer.py : the A-click shared v8-in inference runner (lane BENCH).  Every A-click lane uses it, so all maps
are comparable.  Interface (aclick/PLAN.md "Inference runner"):

  infer.py --ckpt PATH|base --zarr Z --layer0 L [--reverse] [--roi y0 y1 x0 x1] [--tta 8] --out OUT.npy

Recipe = version 5's plaid-free v8-in map (x_push3/v5/code/ink_v8.py, --nsig 3, P2's ctx_maps arithmetic):
  * input = 24 layers L..L+23 of the render volume (axis order layer, y, x; uint8), depth-reversed with --reverse, clipped at 200, / 255;
  * 64 px tiles on a stride-21 grid anchored at --grid-origin (default 0 0 = the render's origin, so a --roi map equals the same
    pixels of a full map); a tile is admitted when >= 25 % of it has CT on the read layer L+12 (--cov-layer), as v5 / X5 / P2;
  * fp16 autocast, sigmoid, each tile weighted by a +-3 sigma Gaussian window (peak 1) and the stitch divided by the per-pixel WEIGHT
    SUM (the release divides by the tile count: 20 % ripple at the stride; the +-1 sigma window leaves 21 px seams; +-3 sigma does not);
  * TTA: 1 = identity; 4 = mirror4 (id / h / v / hv, v5's set); 8 = mirror4 x transpose (the dihedral group).  Default 4 (= v5).
Output: OUT.npy float32 probabilities on the render grid (the --roi crop if given, else the whole render), NaN where no tile was admitted;
OUT.json = metadata (ckpt + sha256, recipe, roi, tiles, minutes).

Checkpoints: 'base' = released v8-in (W/third_party/v8in/model.safetensors, sha256-checked); PATH = a safetensors state dict loadable by
InkDetector (strict).  --alpha A gives WiSE weights base + A (ckpt - base) (float tensors only; BN running stats included; ints kept).
Render volumes: a zarr (array, or group with array '0'; --array to choose) or a directory of numbered layer tifs (NN.tif, or layers/NN.tif).

Resumable: the tile list is cut into row-band parts of ~--part-tiles tiles; each finished part's weighted sums are saved under OUT.parts/
(with a config fingerprint; a mismatch refuses to mix); --max-minutes stops before a part that would overrun and exits 3 (gpu_run.sh
retries).  SIGTERM (gaming-mode daemon) exits 3 too; the part in flight is lost, nothing else.  Parts are deleted after the final map is
written unless --keep-parts.  VRAM is capped with set_per_process_memory_fraction (--vram, default 9 GB).

Library use (cut-gap runners, scorers): load_model(ckpt, alpha) and predict_array(model, vol24, cov, tta=..., roi_mask=...).
"""
import os, sys, json, time, hashlib, argparse, signal, shutil
import numpy as np
import torch

W = os.environ.get('FL_ROOT', '/home/liam/projects/vesuvius-work')
V8 = os.environ.get('V8_DIR', f'{W}/third_party/v8in')
sys.path.insert(0, V8)
from ink8um import InkDetector                                                     # noqa: E402
from ink8um.inference import gkern                                                 # noqa: E402

V8_SHA = '3b94548d7f9ba9f99bc4b0ad8cafcec6d42585c06304e1bd11e47b5d0f88c4cd'
TILE, DEPTH, CLIP = 64, 24, 200
VERSION = 'aclick-infer-1'


# ------------------------------------------------------------------------------------------------------------------------- model
def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()


def vram_cap(gb):
    if torch.cuda.is_available():
        tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        torch.cuda.set_per_process_memory_fraction(min(1.0, gb / tot))


def load_model(ckpt='base', alpha=1.0, device='cuda'):
    """-> (model in eval mode on device, info dict).  ckpt 'base' or a safetensors path; alpha = WiSE interpolation toward base."""
    s = sha256(f'{V8}/model.safetensors')
    if s != V8_SHA: raise SystemExit(f'v8-in base weights sha256 mismatch: {s}')
    model = InkDetector.from_pretrained(V8)
    info = dict(ckpt='base', ckpt_sha256=V8_SHA, alpha=1.0)
    if ckpt != 'base':
        from safetensors.torch import load_file
        sd = load_file(ckpt)
        if alpha != 1.0:
            base = {k: v.detach().clone() for k, v in model.state_dict().items()}
            sd = {k: (base[k] + alpha * (v.to(base[k].dtype) - base[k]) if base[k].is_floating_point() else v) for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
        info = dict(ckpt=os.path.abspath(ckpt), ckpt_sha256=sha256(ckpt), alpha=float(alpha))
    return model.to(device).eval(), info


# ------------------------------------------------------------------------------------------------------------------------- tiles
def window(nsig):
    b = gkern(TILE, nsig)
    return (b / b.max()).astype(np.float32)


def positions(cov, min_cov=0.25, stride=21, origin=(0, 0), roi=None):
    """tile top-left (y, x) in cov's frame on the grid origin + k * stride, fully inside cov's frame, admitted when >= min_cov of the
    tile is True in cov; with roi = (y0, y1, x0, x1) only tiles whose 64 px window overlaps it."""
    H, Wd = cov.shape
    gy = origin[0] % stride; gx = origin[1] % stride
    I = np.pad(cov.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    Y, X = np.meshgrid(np.arange(gy, H - TILE + 1, stride), np.arange(gx, Wd - TILE + 1, stride), indexing='ij')
    Y, X = Y.ravel(), X.ravel()
    if roi is not None:
        y0, y1, x0, x1 = roi
        k = (Y + TILE > y0) & (Y < y1) & (X + TILE > x0) & (X < x1); Y, X = Y[k], X[k]
    s = I[Y + TILE, X + TILE] - I[Y, X + TILE] - I[Y + TILE, X] + I[Y, X]
    ok = s >= min_cov * TILE * TILE
    return np.stack([Y[ok], X[ok]], 1)


def tta_set(n):
    """list of (flip dims, transpose) for TTA 1 / 4 / 8 (order: v5's mirror4, then the transposed four)"""
    m4 = [(), (3,), (2,), (2, 3)]
    if n == 1: return [((), False)]
    if n == 4: return [(f, False) for f in m4]
    if n == 8: return [(f, False) for f in m4] + [(f, True) for f in m4]
    raise ValueError('tta must be 1, 4 or 8')


@torch.inference_mode()
def band_sums(model, st, pts, tta=4, nsig=3, batch=8, device='cuda'):
    """st: (24, h, w) uint8 torch tensor ON device, already ordered and clipped; pts: (n, 2) tile top-lefts in st's frame.
    -> (num, den) float32 numpy (h, w): sum of prob x window and sum of window over admitted tiles and TTA transforms"""
    h, w = st.shape[1:]
    b = torch.from_numpy(window(nsig)).to(device)
    num = torch.zeros((h, w), dtype=torch.float32, device=device); den = torch.zeros_like(num)
    ar = torch.arange(TILE, device=device)
    P = torch.from_numpy(np.ascontiguousarray(pts)).to(device)
    T = tta_set(tta)
    for i in range(0, len(pts), batch):
        q = P[i:i + batch]
        iy = q[:, 0:1] + ar[None]; ix = q[:, 1:2] + ar[None]
        x = st[:, iy[:, :, None], ix[:, None, :]].permute(1, 0, 2, 3).float().mul_(1 / 255.)       # B, 24, 64, 64
        acc = torch.zeros((len(q), TILE, TILE), dtype=torch.float32, device=device)
        for fl, tr in T:
            xi = x.transpose(2, 3) if tr else x
            xi = torch.flip(xi, fl) if fl else xi
            with torch.autocast('cuda', dtype=torch.float16, enabled=device.startswith('cuda')):
                lg = model(xi)
            p = torch.sigmoid(lg.float())
            if p.shape[-2:] != (TILE, TILE): p = torch.nn.functional.interpolate(p, size=(TILE, TILE), mode='bilinear')
            p = torch.flip(p, fl) if fl else p
            p = p.transpose(2, 3) if tr else p
            acc += p[:, 0]
        acc *= b
        for k in range(len(q)):
            y0, x0 = int(q[k, 0]), int(q[k, 1])
            num[y0:y0 + TILE, x0:x0 + TILE] += acc[k]; den[y0:y0 + TILE, x0:x0 + TILE] += b * len(T)
    return num.cpu().numpy(), den.cpu().numpy()


def prep_stack(vol24, reverse=False):
    """(24, H, W) uint8 numpy (stored order) -> clipped, ordered contiguous uint8"""
    assert vol24.shape[0] == DEPTH and vol24.dtype == np.uint8, (vol24.shape, vol24.dtype)
    st = vol24[::-1] if reverse else vol24
    return np.ascontiguousarray(np.minimum(st, CLIP))


def predict_array(model, vol24, cov=None, *, reverse=False, tta=4, nsig=3, min_cov=0.25, stride=21, origin=(0, 0), roi=None,
                  roi_mask=None, batch=8, device='cuda'):
    """In-memory map: vol24 (24, H, W) uint8 in stored order (the 24 layers v8-in reads); cov (H, W) bool = CT on the read layer
    (default vol24[12] > 0).  roi = (y0, y1, x0, x1) or roi_mask (H, W) bool: only tiles overlapping it are run (values there equal a
    full run).  -> (H, W) float32, NaN where no tile."""
    cov = (vol24[12] > 0) if cov is None else cov
    pts = positions(cov, min_cov, stride, origin, roi)
    if roi_mask is not None and len(pts):
        I = np.pad(roi_mask.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0))); Y, X = pts[:, 0], pts[:, 1]
        pts = pts[(I[Y + TILE, X + TILE] - I[Y, X + TILE] - I[Y + TILE, X] + I[Y, X]) > 0]
    H, Wd = vol24.shape[1:]; out = np.full((H, Wd), np.nan, np.float32)
    if not len(pts): return out
    # upload only the bounding box of the selected tiles (identical values; far less VRAM / transfer for roi-limited runs)
    y0, x0 = int(pts[:, 0].min()), int(pts[:, 1].min()); y1, x1 = int(pts[:, 0].max()) + TILE, int(pts[:, 1].max()) + TILE
    batch = int(os.environ.get('BENCH_BATCH', batch))
    st = torch.from_numpy(prep_stack(np.ascontiguousarray(vol24[:, y0:y1, x0:x1]), reverse)).to(device)
    num, den = band_sums(model, st, pts - np.array([y0, x0]), tta, nsig, batch, device)
    del st
    out[y0:y1, x0:x1] = np.where(den > 1e-9, num / np.maximum(den, 1e-9), np.nan)
    return out


# ------------------------------------------------------------------------------------------------------------------------- volumes
class Volume:
    """read-only access to a render volume (layer, y, x) uint8: zarr array / group['0'] / directory of numbered layer tifs"""
    def __init__(self, path, array=None):
        self.path = os.path.abspath(path)
        tifdir = None
        if os.path.isdir(path) and not os.path.exists(f'{path}/.zarray') and not os.path.exists(f'{path}/zarr.json') \
                and not os.path.exists(f'{path}/.zgroup'):
            for d in (path, f'{path}/layers'):
                if os.path.isdir(d) and any(f.split('.')[0].isdigit() and f.lower().endswith(('.tif', '.tiff')) for f in os.listdir(d)):
                    tifdir = d; break
        if tifdir:
            import tifffile
            fs = sorted((f for f in os.listdir(tifdir) if f.split('.')[0].isdigit() and f.lower().endswith(('.tif', '.tiff'))),
                        key=lambda f: int(f.split('.')[0]))
            self.files = [f'{tifdir}/{f}' for f in fs]; self.kind = 'tif'; self._tf = tifffile; self._cache = {}; self.array = None
            t = tifffile.TiffFile(self.files[0]); p = t.pages[0]
            self.shape = (len(self.files),) + tuple(p.shape[:2]); self.dtype = p.dtype; t.close()
        else:
            import zarr
            z = zarr.open(path, mode='r')
            if not hasattr(z, 'shape'): z = z[array or '0']
            elif array: raise SystemExit('--array given but the path is an array')
            self.z = z; self.kind = 'zarr'; self.shape = tuple(z.shape); self.dtype = z.dtype; self.array = array
        if np.dtype(self.dtype) != np.uint8: raise SystemExit(f'render volume must be uint8, got {self.dtype}')

    def read(self, l0, l1, y0, y1, x0, x1):
        if self.kind == 'zarr':
            return np.asarray(self.z[l0:l1, y0:y1, x0:x1])
        out = []
        for i in range(l0, l1):
            if i not in self._cache:
                try: self._cache[i] = self._tf.memmap(self.files[i])
                except Exception: self._cache[i] = self._tf.imread(self.files[i])
            a = self._cache[i]
            out.append(np.asarray(a[y0:y1, x0:x1] if a.ndim == 2 else a[y0:y1, x0:x1, 0]))
        return np.stack(out)


# ------------------------------------------------------------------------------------------------------------------------- CLI
class Stop(Exception):
    pass


def render(model, info, V, layer0, out, *, reverse=False, roi=None, tta=4, nsig=3, min_cov=0.25, cov_layer=None, stride=21,
           grid_origin=(0, 0), batch=8, part_tiles=2500, deadline=None, keep_parts=False, device='cuda', extra_meta=None,
           clip_to_roi=False):
    """Map of render volume V (a Volume) into out (.npy, float32, the roi crop; NaN = no tile) + out.json.  Resumable through
    out.parts/; returns 0 when the map is written, 3 when stopped at the deadline (a time.time() value) or by SIGTERM."""
    T0 = time.time()
    if os.path.exists(out) and os.path.exists(out[:-4] + '.json'): return 0
    D, H, Wd = V.shape
    if layer0 < 0 or layer0 + DEPTH > D: raise SystemExit(f'layers {layer0}..{layer0 + 23} outside 0..{D - 1}')
    cl = layer0 + 12 if cov_layer is None else cov_layer
    roi = tuple(roi) if roi else (0, H, 0, Wd)
    y0, y1, x0, x1 = (int(v) for v in (max(0, roi[0]), min(H, roi[1]), max(0, roi[2]), min(Wd, roi[3])))
    if y1 <= y0 or x1 <= x0: raise SystemExit(f'empty roi {roi} for render {H}x{Wd}')
    # frame read for tile admission: roi padded by one tile, clipped to the render.  clip_to_roi: tiles only inside the roi (= a run
    # on the roi-cropped volume, the X5 / harness convention; pass grid_origin = the roi's top-left to match predict_array on the crop)
    if clip_to_roi:
        fy0, fy1, fx0, fx1 = y0, y1, x0, x1
    else:
        fy0, fy1, fx0, fx1 = max(0, y0 - TILE + 1), min(H, y1 + TILE - 1), max(0, x0 - TILE + 1), min(Wd, x1 + TILE - 1)
    cov = V.read(cl, cl + 1, fy0, fy1, fx0, fx1)[0] > 0
    org = ((grid_origin[0] - fy0) % stride, (grid_origin[1] - fx0) % stride)
    pts = positions(cov, min_cov, stride, org, (y0 - fy0, y1 - fy0, x0 - fx0, x1 - fx0))
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    parts = []                                                       # row-band parts of ~part_tiles tiles, cut between grid rows
    if len(pts):
        rows, cnt = np.unique(pts[:, 0], return_counts=True); cur, n = [], 0
        for r, c in zip(rows, cnt):
            cur.append(int(r)); n += int(c)
            if n >= part_tiles: parts.append((cur[0], cur[-1])); cur, n = [], 0
        if cur: parts.append((cur[0], cur[-1]))
    cfg = dict(version=VERSION, zarr=V.path, array=V.array, layer0=layer0, reverse=bool(reverse), cov_layer=cl,
               roi=[y0, y1, x0, x1], frame=[fy0, fy1, fx0, fx1], tta=tta, nsig=nsig, min_cov=min_cov, stride=stride,
               grid_origin=[int(g) for g in grid_origin], ckpt_sha256=info['ckpt_sha256'], alpha=info['alpha'], part_tiles=part_tiles, clip_to_roi=bool(clip_to_roi))
    PD = out[:-4] + '.parts'; os.makedirs(PD, exist_ok=True)
    fp = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
    if os.path.exists(f'{PD}/config.json'):
        if json.load(open(f'{PD}/config.json')).get('fingerprint') != fp:
            raise SystemExit(f'{PD} holds parts of a different configuration; remove it or change the output path')
    else:
        json.dump(dict(cfg, fingerprint=fp), open(f'{PD}/config.json', 'w'), indent=1)
    print(f'infer: {info["ckpt"]} alpha {info["alpha"]}  {os.path.basename(out)}: {len(pts)} tiles x tta {tta} in {len(parts)} parts',
          flush=True)
    last = 0.0
    try:
        for k, (ra, rb) in enumerate(parts):
            pf = f'{PD}/part{k:04d}.npz'
            if os.path.exists(pf): continue
            if deadline is not None and time.time() + 1.2 * last > deadline:
                print(f'time cap: stopping before part {k}/{len(parts)}', flush=True); return 3
            t = time.time()
            sel = pts[(pts[:, 0] >= ra) & (pts[:, 0] <= rb)]
            by0, by1 = int(ra), int(rb) + TILE; bx0, bx1 = int(sel[:, 1].min()), int(sel[:, 1].max()) + TILE
            vol = V.read(layer0, layer0 + DEPTH, fy0 + by0, fy0 + by1, fx0 + bx0, fx0 + bx1)
            st = torch.from_numpy(prep_stack(vol, reverse)).to(device); del vol
            num, den = band_sums(model, st, sel - np.array([by0, bx0]), tta, nsig, batch, device)
            del st
            np.savez(pf[:-4] + '.tmp.npz', box=np.array([by0, by1, bx0, bx1]), num=num, den=den); os.replace(pf[:-4] + '.tmp.npz', pf)
            last = time.time() - t
            print(f'  part {k + 1}/{len(parts)}: {len(sel)} tiles {last:.0f} s ({len(sel) * tta / max(last, 1e-3):.0f} fwd/s)', flush=True)
    except Stop:
        print('SIGTERM: stopping (finished parts kept)', flush=True); return 3
    nu = np.zeros((fy1 - fy0, fx1 - fx0), np.float32); de = np.zeros_like(nu)
    for k in range(len(parts)):
        z = np.load(f'{PD}/part{k:04d}.npz'); by0, by1, bx0, bx1 = (int(v) for v in z['box'])
        nu[by0:by1, bx0:bx1] += z['num']; de[by0:by1, bx0:bx1] += z['den']
    m = np.where(de > 1e-9, nu / np.maximum(de, 1e-9), np.nan).astype(np.float32)
    m = m[y0 - fy0:y1 - fy0, x0 - fx0:x1 - fx0]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.save(out[:-4] + '.tmp.npy', m); os.replace(out[:-4] + '.tmp.npy', out)
    json.dump(dict(cfg, ckpt=info['ckpt'], tiles=int(len(pts)), render_shape=[D, H, Wd], out_shape=list(m.shape),
                   recipe=f'v8-in 24 layers {layer0}..{layer0 + 23}{" reversed" if reverse else ""}, clip {CLIP}, /255, {TILE} px '
                          f'tiles stride {stride} (grid origin {list(grid_origin)}), admitted at >= {min_cov} CT on layer {cl}, '
                          f'fp16 autocast, sigmoid, +-{nsig} sigma window / weight sum, TTA {tta}; NaN = no tile',
                   minutes_last_session=round((time.time() - T0) / 60, 2), finished=time.strftime('%Y-%m-%d %H:%M:%S'),
                   **(extra_meta or {})),
              open(out[:-4] + '.json', 'w'), indent=1)
    if not keep_parts: shutil.rmtree(PD, ignore_errors=True)
    print('wrote', out, m.shape, flush=True)
    return 0


def install_sigterm():
    def on_term(sig, frm): raise Stop()
    signal.signal(signal.SIGTERM, on_term)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--ckpt', required=True, help="'base' or a safetensors state dict for InkDetector")
    ap.add_argument('--alpha', type=float, default=1.0, help='WiSE: base + alpha (ckpt - base)')
    ap.add_argument('--zarr', required=True, help='render volume: zarr (array or group with "0") or a directory of numbered layer tifs')
    ap.add_argument('--array', default=None, help='array name inside a zarr group (default "0")')
    ap.add_argument('--layer0', type=int, required=True, help='first of the 24 layers v8-in reads')
    ap.add_argument('--reverse', action='store_true', help='flip the 24-layer depth order before v8-in')
    ap.add_argument('--roi', type=int, nargs=4, metavar=('Y0', 'Y1', 'X0', 'X1'), default=None)
    ap.add_argument('--tta', type=int, default=4, choices=(1, 4, 8))
    ap.add_argument('--nsig', type=float, default=3, help='stitch window +- nsig sigma (3 = v5 plaid-free; 1 = release)')
    ap.add_argument('--min-cov', type=float, default=0.25)
    ap.add_argument('--cov-layer', type=int, default=None, help='layer whose CT (> 0) admits tiles (default layer0 + 12)')
    ap.add_argument('--stride', type=int, default=21)
    ap.add_argument('--grid-origin', type=int, nargs=2, default=(0, 0), metavar=('GY', 'GX'))
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--vram', type=float, default=9.0, help='per-process VRAM cap in GB')
    ap.add_argument('--part-tiles', type=int, default=2500)
    ap.add_argument('--max-minutes', type=float, default=26.0)
    ap.add_argument('--keep-parts', action='store_true')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    T0 = time.time()
    if not a.out.endswith('.npy'): raise SystemExit('--out must end with .npy')
    if os.path.exists(a.out) and os.path.exists(a.out[:-4] + '.json'): print('done already', a.out); return 0
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dev == 'cuda': vram_cap(a.vram)
    install_sigterm()
    V = Volume(a.zarr, a.array)
    model, info = load_model(a.ckpt, a.alpha, dev)
    return render(model, info, V, a.layer0, a.out, reverse=a.reverse, roi=a.roi, tta=a.tta, nsig=a.nsig, min_cov=a.min_cov,
                  cov_layer=a.cov_layer, stride=a.stride, grid_origin=tuple(a.grid_origin), batch=a.batch, part_tiles=a.part_tiles,
                  deadline=T0 + a.max_minutes * 60, keep_parts=a.keep_parts, device=dev)


if __name__ == '__main__':
    sys.exit(main())
