"""v8lib.py : v8-in inference for lane P2 with a CORRECT stitcher, plus data access shared by the lane's scripts.

The release's stitcher (third_party/v8in/ink8um/inference.py, "as used for all released predictions") accumulates
prob * gaussian(64) / max and divides by the per-pixel tile COUNT.  The Gaussian weights do not sum to a constant over
overlapping tiles, so a constant prediction comes out modulated by +-10 % (20.05 % relative range) with the 21 px stride
period: the checkerboard of x_submit/v8loo (diag_checker.py measured 20.7 % on three maps).  Here the weighted sum is
divided by the per-pixel weight SUM (a partition of unity), everything else as the release: 64 px tiles fully inside the
coverage mask (max over depth > 0, 7x7 elliptical closing), stride 21, uint8 clipped to 200, x 1/255, fp16 autocast.

Options: depth window (centre shifts d, averaged), mirror TTA (identity / h / v / hv flips; v8-in was trained with flips
and 360 deg rotations), and the release stitcher for comparison (norm='count').

Models: 'base' = released v8-in (third_party/v8in), 'loo' = YoussefMoNader v8in-pherc1447-loo-w062 (w058 + w060
fine-tune), or a path to a safetensors state dict of the same class.  Every weights file is sha256-checked against SHA.
"""
import hashlib, json, os, sys
import numpy as np, cv2, torch, torch.nn.functional as F

cv2.setNumThreads(4)
torch.set_num_threads(4)
W = '/home/liam/projects/vesuvius-work'
T = f'{W}/x_refine/trials/P2_inkft_0306B'
sys.path.insert(0, f'{W}/third_party/v8in')
from ink8um import InkDetector                                                    # noqa: E402
from ink8um.inference import gkern, coverage_mask, tile_positions                 # noqa: E402
from safetensors.torch import load_file                                          # noqa: E402

PATHS = {'base': f'{W}/third_party/v8in', 'loo': f'{W}/models/community/v8in_1447_loo_w062'}
SHA = {'base': None,   # filled on first use and recorded in outputs
       'loo': '4b28e153e0a631679c79b6c9b8e3e159108d75aeddc6fdea648708ffec5026b5'}
BASE_SHA = '3b94548d7f9ba9f99bc4b0ad8cafcec6d42585c06304e1bd11e47b5d0f88c4cd'   # released v8-in, verified at step 0


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()


def load_model(name, device='cuda'):
    """returns (model, sha256 of the weights file)"""
    if name in PATHS:
        wf = f'{PATHS[name]}/model.safetensors'; s = sha256(wf)
        if SHA[name] and s != SHA[name]: raise SystemExit(f'sha mismatch {wf}: {s}')
        m = InkDetector.from_pretrained(PATHS[name])
    elif name.startswith('wise_'):          # PREREG amendment 9: wise_<ARM>_a<ALPHA> = (1 - a) base + a ckpt/<ARM>/final, in memory
        arm, al = name[5:].rsplit('_a', 1); al = float(al)
        wf = f'{T}/ckpt/{arm}/final.safetensors'; s = sha256(wf)
        ref = json.load(open(f'{T}/ckpt/{arm}/done.json'))['sha256']
        if s != ref: raise SystemExit(f'sha mismatch {wf}: {s} != done.json {ref}')
        bs = sha256(f'{PATHS["base"]}/model.safetensors')
        if bs != BASE_SHA: raise SystemExit(f'base sha mismatch {bs}')
        m = InkDetector.from_pretrained(PATHS['base']); b = {k: v.detach().clone() for k, v in m.state_dict().items()}; ft = load_file(wf)
        m.load_state_dict({k: ((1 - al) * b[k].float() + al * ft[k].float()).to(b[k].dtype) if b[k].is_floating_point() else b[k]
                           for k in b}, strict=True)
        s = f'wise a{al:g}: (1-a) base {bs[:12]} + a {arm} {s}'
    else:
        wf = name; s = sha256(wf)
        m = InkDetector.from_pretrained(PATHS['base'])
        m.load_state_dict(load_file(wf), strict=True)
    return m.to(device).eval(), s


_BLEND = None


def blend():
    global _BLEND
    if _BLEND is None:
        g = gkern(64, 1); _BLEND = (g / g.max()).astype(np.float32)
    return _BLEND


@torch.inference_mode()
def predict(model, stack, stride=21, tta="none", norm="weight", batch=4, device='cuda', valid=None, nsigs=None):
    """stack (H, W, 24) uint8 already in MODEL order (clipped at 200).  Returns float32 (H, W) probabilities.
    tta: 'none' | 'mirror4'.  norm: 'weight' (correct) | 'count' (release) | 'both' -> (weight map, count map).
    nsigs: e.g. (1, 3) -> a list of weight-normalised maps, one per Gaussian window width (+-nsig sigma across the tile),
    accumulated from the same forward pass (amendment 6; nsig 1 = the release window)."""
    H, Wd = stack.shape[:2]
    cov = coverage_mask(stack) if valid is None else (valid.astype(np.uint8) * 255)
    pts = tile_positions(cov, 64, stride)
    num = np.zeros((H, Wd), np.float32); den = np.zeros((H, Wd), np.float32); cnt = np.zeros((H, Wd), np.float32)
    if not pts: return [num.copy() for _ in nsigs] if nsigs else ((num, num.copy()) if norm == 'both' else num)
    bl = blend()
    if nsigs:
        bls = [(lambda g: (g / g.max()).astype(np.float32))(gkern(64, ns)) for ns in nsigs]
        nums = [np.zeros((H, Wd), np.float32) for _ in nsigs]; dens = [np.zeros((H, Wd), np.float32) for _ in nsigs]
    flips = {'none': [()], 'mirror4': [(), (3,), (2,), (2, 3)], 'id': [()], 'h': [(3,)], 'v': [(2,)], 'hv': [(2, 3)]}[tta]
    for i in range(0, len(pts), batch):
        P = pts[i:i + batch]
        x = np.stack([stack[y:y + 64, xx:xx + 64].transpose(2, 0, 1) for xx, y in P])
        xt = torch.from_numpy(x).to(device).float().mul_(1.0 / 255.0)
        acc = None
        for fl in flips:
            xi = torch.flip(xt, fl) if fl else xt
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                lg = model(xi)
            p = torch.sigmoid(lg.float())
            if p.shape[-2:] != (64, 64): p = F.interpolate(p, size=(64, 64), mode='bilinear')
            p = torch.flip(p, fl) if fl else p
            acc = p if acc is None else acc + p
        pr = (acc / len(flips))[:, 0].cpu().numpy()
        for p, (xx, y) in zip(pr, P):
            if nsigs:
                for nu, de, b in zip(nums, dens, bls):
                    nu[y:y + 64, xx:xx + 64] += p * b; de[y:y + 64, xx:xx + 64] += b
                continue
            num[y:y + 64, xx:xx + 64] += p * bl
            den[y:y + 64, xx:xx + 64] += bl
            cnt[y:y + 64, xx:xx + 64] += 1.0
    if nsigs:
        return [np.clip(np.nan_to_num(np.divide(nu, de, out=np.zeros_like(nu), where=de > 1e-9)), 0, 1) for nu, de in zip(nums, dens)]
    fin = lambda d: np.clip(np.nan_to_num(np.divide(num, d, out=np.zeros_like(num), where=d > 0)), 0, 1)   # noqa: E731
    if norm == 'both': return fin(den), fin(cnt)
    return fin(den) if norm == 'weight' else fin(cnt)


def stack_from_volume(vol, c, d=0, order='asc'):
    """vol (D, H, W) uint8 (render layers); 24 layers c+d-12 .. c+d+11; order 'asc' = as stored, 'desc' = reversed.
    Returns (H, W, 24) uint8 clipped at 200."""
    s = c + d - 12
    assert s >= 0 and s + 24 <= vol.shape[0], (c, d, vol.shape)
    a = vol[s:s + 24]
    if order == 'desc': a = a[::-1]
    return np.ascontiguousarray(np.minimum(a, 200).transpose(1, 2, 0)).astype(np.uint8)


def predict_depths(model, vol, c, depths=(0,), order='asc', **kw):
    acc = [predict(model, stack_from_volume(vol, c, d, order), **kw) for d in depths]
    return np.mean(acc, 0)


def save_map(path, arr, meta):
    np.save(path, arr.astype(np.float16))
    json.dump(meta, open(path[:-4] + '.json', 'w'), indent=1)
