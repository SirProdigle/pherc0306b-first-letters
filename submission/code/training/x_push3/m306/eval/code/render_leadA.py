"""render_leadA.py : a v8-family ink map of lead A on the V5Rc render with the C3 recipe (M306 EVAL lane).

  render_leadA.py --ckpt base|PATH [--order asc|desc] [--render ZARR] [--region ctx|box|u0,u1,v0,v1] [--tta 4] --out OUT/fwd_NAME.npy

Recipe = version 5's v8g3 map (P3/v5/code/ink_v8.py --nsig 3, P2's ctx_maps arithmetic), run through the A-click shared runner
(P3/aclick/bench/infer.py, verified against ink_v8.py: max |diff| 9.9e-5, bench/verify_infer.json):
  24 render layers 8..31 (window centred on the surface layer 20), stored order ('asc') or reversed ('desc'), clip 200, / 255;
  64 px tiles on a stride-21 grid anchored at the context window's corner (render px y 115, x 694), a tile admitted when >= 25 % of
  it has CT on layer 20; mirror4 TTA; fp16 autocast, sigmoid; +-3 sigma Gaussian window, stitch divided by the weight sum.
  --region ctx = every tile inside the claim-image context u -30..+30 x v -20..+15 mm (render px y 115..4166, x 694..7638), as ink_v8
  --region ctx; --region box (or u0,u1,v0,v1 mm) = only the tiles that overlap the box (pixels inside it then get exactly the tiles of
  the full run; outside = NaN), as ink_v8 --region box (default box: the claim box u -15..+23 x v -6.84..+2.72).
Output: OUT (4900 x 7780 float16 on the render raster, NaN where no tile / outside the region) + OUT.json.  Resumable (OUT.parts/),
exit 3 at --max-minutes or SIGTERM (P3/gpu_run.sh retries).  VRAM capped at --vram GB.
Checkpoints: 'base' = released v8-in (sha-checked); PATH = .safetensors state dict, or a torch file (.pt/.pth/.ckpt: a state dict, or a
dict holding one under 'state_dict' / 'model' / 'model_state_dict' / 'ema'; 'module.' / 'model.' prefixes stripped), loaded strictly
into W/third_party/v8in's InkDetector (torch.load weights_only=True).
"""
import os, sys, json, time, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
W = '/home/liam/projects/vesuvius-work'; P3 = f'{W}/.claude/worktrees/leadA-push3/x_push3'
sys.path.insert(0, f'{P3}/aclick/bench')
import infer as I                                                                 # noqa: E402
import torch                                                                      # noqa: E402

K = 1000.0 / 8.64; RAST = (4900, 7780)
V5RC = f'{W}/x_refine/trials/P1_surface_refine/renders/P1V5Rc.zarr'
SHUF = f'{W}/x_req/20261020/A10/v4c_work/shuf_render.zarr'
CTXWIN = (115, 4166, 694, 7638)                                                   # ink_v8: int((-20+21)K), int((15+21)K), int((-30+36)K), int((30+36)K)
BOX = (-15.0, 23.0, -6.84, 2.72)


def box_px(u0, u1, v0, v1):
    """render px (y0, y1, x0, x1) whose centres lie inside the box (ink_v8.py / controls_v5.boxpx)"""
    return (int(np.ceil((v0 + 21) * K - 0.5)), int(np.floor((v1 + 21) * K - 0.5)) + 1,
            int(np.ceil((u0 + 36) * K - 0.5)), int(np.floor((u1 + 36) * K - 0.5)) + 1)


def load_state(path):
    if path.endswith('.safetensors'):
        from safetensors.torch import load_file
        return load_file(path)
    sd = torch.load(path, map_location='cpu', weights_only=True)
    for k in ('state_dict', 'model_state_dict', 'model', 'ema'):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict): sd = sd[k]; break
    out = {}
    for k, v in sd.items():
        for p in ('module.', '_orig_mod.'):
            if k.startswith(p): k = k[len(p):]
        out[k] = v
    return out


def load_model(ckpt, device):
    if ckpt == 'base' or ckpt.endswith('.safetensors'):
        return I.load_model(ckpt, 1.0, device)
    s = I.sha256(f'{I.V8}/model.safetensors')
    if s != I.V8_SHA: raise SystemExit(f'v8-in base weights sha256 mismatch: {s}')
    model = I.InkDetector.from_pretrained(I.V8)
    sd = load_state(ckpt)
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError:
        sd = {(k[6:] if k.startswith('model.') else k): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
    return model.to(device).eval(), dict(ckpt=os.path.abspath(ckpt), ckpt_sha256=I.sha256(ckpt), alpha=1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--ckpt', required=True); ap.add_argument('--order', default='asc', choices=('asc', 'desc'))
    ap.add_argument('--render', default=V5RC); ap.add_argument('--region', default='ctx')
    ap.add_argument('--tta', type=int, default=4, choices=(1, 4, 8)); ap.add_argument('--nsig', type=float, default=3)
    ap.add_argument('--batch', type=int, default=8); ap.add_argument('--vram', type=float, default=9.0)
    ap.add_argument('--max-minutes', type=float, default=26.0); ap.add_argument('--part-tiles', type=int, default=400)   # small parts survive gaming SIGTERMs
    ap.add_argument('--out', required=True)
    a = ap.parse_args(); T0 = time.time()
    if not a.out.endswith('.npy'): raise SystemExit('--out must end with .npy')
    if os.path.exists(a.out) and os.path.exists(a.out[:-4] + '.json'): print('done already', a.out); return 0
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dev == 'cuda': I.vram_cap(a.vram)
    I.install_sigterm()
    V = I.Volume(a.render)
    if a.region == 'ctx': roi, clip = CTXWIN, True
    else:
        b = BOX if a.region == 'box' else tuple(float(v) for v in a.region.split(','))
        roi, clip = box_px(*b), False
    model, info = load_model(a.ckpt, dev)
    tmp = a.out[:-4] + '.crop.npy'
    rc = I.render(model, info, V, 8, tmp, reverse=a.order == 'desc', roi=roi, tta=a.tta, nsig=a.nsig, min_cov=0.25, cov_layer=20,
                  stride=21, grid_origin=(CTXWIN[0], CTXWIN[2]), batch=a.batch, part_tiles=a.part_tiles,
                  deadline=T0 + a.max_minutes * 60, device=dev, clip_to_roi=clip,
                  extra_meta=dict(region=a.region, order=a.order))
    if rc: return rc
    m = np.load(tmp); meta = json.load(open(tmp[:-4] + '.json'))
    fr = np.full(RAST, np.nan, np.float16); y0, y1, x0, x1 = meta['roi']; fr[y0:y1, x0:x1] = m.astype(np.float16)
    np.save(a.out[:-4] + '.tmp.npy', fr); os.replace(a.out[:-4] + '.tmp.npy', a.out)
    meta.update(full_raster=list(RAST), region_px=[y0, y1, x0, x1], layers='8..31', order=a.order,
                recipe_c3=f'v8-family, 24 layers 8..31 {a.order}, clip 200, stride 21 from (115, 694), >= 25 % CT on layer 20, '
                          f'TTA {a.tta}, +-{a.nsig:g} sigma window / weight sum (= v5 v8g3 when ckpt base, asc, ctx, tta 4)')
    json.dump(meta, open(a.out[:-4] + '.json', 'w'), indent=1)
    os.remove(tmp); os.remove(tmp[:-4] + '.json')
    print('wrote', a.out, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
