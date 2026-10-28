"""ink_v8.py RENDER.zarr OUT_DIR [--region ctx|box|u0,u1,v0,v1] [--chunks 4] [--max-minutes 28] : base v8-in on the v2 mesh's
render, lane P2's recipe (lane I2 copy of P2's ctx_maps.py + v8lib).

Recipe (P2, x_refine/reports/P2_inkft_0306B.md): released base v8-in weights (ResNet3D-50 + 2-D decoder, our ink8um loader,
model.safetensors sha256 3b94548d...c4cd); input = render layers 8..31 (24 layers, window centred on the surface layer 20, delta 0)
in stored order (asc), uint8 clipped at 200, / 255; 64 px tiles on a stride-21 grid over the claim-image context u -30..+30 x
v -20..+15 mm (render px y 115..4166, x 694..7638), a tile admitted when >= 25 % of it has CT data (layer 20 > 0); mirror4 TTA
(identity / h / v / hv, each run as its own pass); fp16 autocast, sigmoid; stitched with a Gaussian (+-1 sigma) weight and divided
by the per-pixel weight SUM (the release divides by the tile count, which leaves a 20 % ripple at the stride period).
The tile list is split into --chunks row-band parts per flip (P2: 4); the map = sum of the parts' weighted sums / sum of their weights,
accumulated flip by flip, part by part, in P2's order, so a re-run reproduces P2's arithmetic.

--region ctx  every tile (P2's full-context map; ~61k tile positions x 4 flips)
--region box  only the tiles that overlap the 4 cm2 claim box (version 4c: u -15..+23 x v -6.84..+2.72); every pixel INSIDE the box then
              gets exactly the same tile set as in the full run, so the box region is comparable with the full map value by value;
              pixels outside the box are written NaN.
Restartable: each (flip, part) is saved under OUT_DIR/v8parts/; --max-minutes stops before a part that would overrun (exit 3).
Writes OUT_DIR/fwd_v8.npy (4900 x 7780 float16, NaN outside the region) and fwd_v8.json.
Version 5: --nsig N stitches with a Gaussian (+-N sigma) window (default 1 = the release's window, versions 2-4c).  --nsig 3 --name v8g3
is P2's plaid-free map of the same pass (V5Rc_ctx_base_dl0_asc_g3_fullraster.npy, the v8-in map of version 5's fusion): the +-1 sigma
window enters and leaves each 64 px tile at 61 % weight, which leaves 21 px tile seams; the +-3 sigma window does not."""
import os, sys, json, time, argparse, hashlib
import numpy as np, zarr, torch
W = os.environ.get('FL_ROOT', '/home/liam/projects/vesuvius-work')
V8 = os.environ.get('V8_DIR', f'{W}/third_party/v8in')
sys.path.insert(0, V8)
from ink8um import InkDetector                                                     # noqa: E402
from ink8um.inference import gkern                                                 # noqa: E402
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
K = 1000.0 / 8.64; C = 20
V8_SHA = '3b94548d7f9ba9f99bc4b0ad8cafcec6d42585c06304e1bd11e47b5d0f88c4cd'
V8Q_SHA = '4cb8e9217e5fba795cc0d7fc53afd998ad82c6c5d1250cdb9e63008f36a8f305'                # v8Q: our fine-tune (P2), no PHerc0306B data
FLIPS = {'id': (), 'h': (3,), 'v': (2,), 'hv': (2, 3)}
BOX = (-15.0, 23.0, -6.84, 2.72)                                  # version 4c box (v4b: -15, 25, -7.35, 2.525; v4: -15, 25, -5, 4.875; v2 / v3: -12.1, 23.5, -8.55, 2.5)


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()


def positions(cov, min_cov, stride=21):                                           # P2 ctx_maps.positions
    H, Wd = cov.shape; I = np.pad(cov.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    Y, X = np.meshgrid(np.arange(0, H - 63, stride), np.arange(0, Wd - 63, stride), indexing='ij'); Y, X = Y.ravel(), X.ravel()
    s = I[Y + 64, X + 64] - I[Y, X + 64] - I[Y + 64, X] + I[Y, X]
    ok = s >= min_cov * 4096
    return np.stack([Y[ok], X[ok]], 1)


@torch.inference_mode()
def part_sums(model, st, pts, flip, batch=4, device='cuda', nsig=1):                # P2 ctx_maps.part_sums; v5: window +-nsig sigma (P2: 1 and 3)
    b = (lambda g: (g / g.max()).astype(np.float32))(gkern(64, nsig))
    y0b, y1b = int(pts[:, 0].min()), int(pts[:, 0].max()) + 64; Wd = st.shape[1]
    num = np.zeros((y1b - y0b, Wd), np.float32); den = np.zeros((y1b - y0b, Wd), np.float32)
    for i in range(0, len(pts), batch):
        P = pts[i:i + batch]
        x = torch.from_numpy(np.stack([st[y:y + 64, xx:xx + 64].transpose(2, 0, 1) for y, xx in P])).to(device).float().mul_(1 / 255.)
        fl = FLIPS[flip]; xi = torch.flip(x, fl) if fl else x
        with torch.autocast(device, dtype=torch.float16, enabled=device == 'cuda'):
            lg = model(xi)
        p = torch.sigmoid(lg.float()); p = torch.flip(p, fl) if fl else p
        p = p[:, 0].cpu().numpy()
        for q, (y, xx) in zip(p, P):
            num[y - y0b:y - y0b + 64, xx:xx + 64] += q * b; den[y - y0b:y - y0b + 64, xx:xx + 64] += b
    return y0b, num, den


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('render'); ap.add_argument('out')
    ap.add_argument('--region', default='box'); ap.add_argument('--min-cov', type=float, default=0.25)
    ap.add_argument('--chunks', type=int, default=4); ap.add_argument('--max-minutes', type=float, default=28)
    ap.add_argument('--vram', type=float, default=7.5); ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--weights', default='', help='a safetensors state dict of the same class (e.g. v8Q) loaded over the base; sha-checked')
    ap.add_argument('--sha', default='', help='expected sha256 of --weights (default: v8Q\'s)'); ap.add_argument('--name', default='v8')
    ap.add_argument('--nsig', type=int, default=1, help='Gaussian stitch window, +- nsig sigma (1: release / versions 2-4c; 3: version 5 v8g3)')
    a = ap.parse_args(); T0 = time.time(); os.makedirs(f'{a.out}/v8parts', exist_ok=True)
    if os.path.exists(f'{a.out}/fwd_{a.name}.npy'): print('done already', a.out); return
    dev = os.environ.get('INK_DEVICE', 'cuda') if torch.cuda.is_available() else 'cpu'
    if dev == 'cuda':
        tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 30; torch.cuda.set_per_process_memory_fraction(min(1.0, a.vram / tot))
    y0, y1, x0, x1 = int((-20 + 21) * K), int((15 + 21) * K), int((-30 + 36) * K), int((30 + 36) * K)
    vol = np.asarray(zarr.open(a.render, mode='r')['0'][C - 12:C + 12, y0:y1, x0:x1])
    st = np.ascontiguousarray(np.minimum(vol, 200).transpose(1, 2, 0)).astype(np.uint8)
    cov = vol[12] > 0; del vol
    pts = positions(cov, a.min_cov); pts = pts[np.argsort(pts[:, 0], kind='stable')]
    chunks = np.array_split(pts, a.chunks)                                          # P2's parts (split of the FULL tile list)
    if a.region == 'ctx': reg = None; tag = 'ctx'
    else:
        u0, u1, v0, v1 = BOX if a.region == 'box' else (float(v) for v in a.region.split(','))
        reg = (int(np.ceil((v0 + 21) * K - 0.5)) - y0, int(np.floor((v1 + 21) * K - 0.5)) - y0 + 1,
               int(np.ceil((u0 + 36) * K - 0.5)) - x0, int(np.floor((u1 + 36) * K - 0.5)) - x0 + 1)   # window px whose centre is inside
        sel = lambda c: c[(c[:, 0] + 64 > reg[0]) & (c[:, 0] < reg[1]) & (c[:, 1] + 64 > reg[2]) & (c[:, 1] < reg[3])]   # noqa: E731
        chunks = [sel(c) for c in chunks]; tag = f'box_{reg[0]}_{reg[1]}_{reg[2]}_{reg[3]}'
    tag = f'{a.name}_{tag}'
    ntiles = int(sum(len(c) for c in chunks))
    print(f'v8-in {a.region}: {ntiles} tile positions x {len(FLIPS)} flips on {dev}', flush=True)
    wf = f'{V8}/model.safetensors'; s = sha256(wf)
    if s != V8_SHA: raise SystemExit(f'v8-in weights sha256 mismatch: {s}')
    model = InkDetector.from_pretrained(V8)
    if a.weights:
        from safetensors.torch import load_file
        sw = sha256(a.weights)
        if sw != (a.sha or V8Q_SHA): raise SystemExit(f'weights sha256 mismatch: {sw}')
        model.load_state_dict(load_file(a.weights), strict=True); s = sw
    model = model.to(dev).eval()
    last = 0.0
    for fl in FLIPS:
        for k, cp in enumerate(chunks):
            pf = f'{a.out}/v8parts/{tag}_{fl}_c{k}of{a.chunks}.npz'
            if os.path.exists(pf) or not len(cp): continue
            if time.time() - T0 + last > a.max_minutes * 60: print('time cap: stopping before', fl, k, flush=True); sys.exit(3)
            t = time.time(); yb, num, den = part_sums(model, st, cp, fl, a.batch, dev, a.nsig)
            np.savez(pf + '.tmp.npz', y0b=yb, num1=num, den1=den); os.replace(pf + '.tmp.npz', pf); last = time.time() - t
            print(f'  {fl} part {k}: {len(cp)} tiles {last:.0f}s', flush=True)
    H, Wd = cov.shape; nu = np.zeros((H, Wd), np.float32); de = np.zeros((H, Wd), np.float32)
    for fl in FLIPS:
        for k, cp in enumerate(chunks):
            if not len(cp): continue
            z = np.load(f'{a.out}/v8parts/{tag}_{fl}_c{k}of{a.chunks}.npz'); yb = int(z['y0b'])
            nu[yb:yb + z['num1'].shape[0]] += z['num1']; de[yb:yb + z['den1'].shape[0]] += z['den1']
    m = np.where(de > 1e-9, nu / np.maximum(de, 1e-9), np.nan)
    if reg is not None:
        keep = np.zeros_like(cov); keep[reg[0]:reg[1], reg[2]:reg[3]] = True; m[~keep] = np.nan
    fr = np.full((4900, 7780), np.nan, np.float16); fr[y0:y1, x0:x1] = m.astype(np.float16)
    np.save(f'{a.out}/fwd_{a.name}.tmp.npy', fr); os.replace(f'{a.out}/fwd_{a.name}.tmp.npy', f'{a.out}/fwd_{a.name}.npy')
    json.dump(dict(render=a.render, region=a.region, region_window_px=reg, window_carrier_px=[y0, y1, x0, x1], tiles=ntiles,
                   recipe=f'base v8-in, delta 0, stride 21, weight-normalised stitch (+-{a.nsig} sigma), mirror4, d0, asc, min tile CT coverage 0.25',
                   weights_sha256=s, weights=os.path.basename(a.weights) if a.weights else 'v8-in base', minutes=round((time.time() - T0) / 60, 1)),
              open(f'{a.out}/fwd_{a.name}.json', 'w'), indent=1)
    print('wrote', f'{a.out}/fwd_{a.name}.npy', flush=True)


if __name__ == '__main__':
    main()
