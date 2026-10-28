"""GPU-resident tiled inference for the three First Letters ink models.

The production paths (models/ink9_ens/hecate.py predict, villa `infer.py`, third_party/v8in predict_stack) read every
patch on the CPU, normalise it there and accumulate the overlap-add in numpy.  Here the whole input window of a patch
(a few hundred MB of uint8) is copied to the GPU once; patch extraction, villa's robust normalisation, the model,
the blend and the accumulation all stay on the GPU.  Each runner reproduces its reference implementation's maths
(same origins, windows, skip rules, uint8 encoding) so its maps can be checked against the reference output
(`parity.py`), and adds the speed knobs being measured: precision, channels_last, torch.compile, batch size,
TTA subsets, stride and a tile skip mask.

    from fastink import HecateFast, VillaFast, V8Fast, hecate_prep
"""
from __future__ import annotations

import importlib.util, math, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import zarr

W = Path('/home/liam/projects/vesuvius-work')
VILLA_SRC = '/home/liam/projects/villa/vesuvius/src'
if VILLA_SRC not in sys.path:
    sys.path.insert(0, VILLA_SRC)

DT = {'fp32': None, 'bf16': torch.bfloat16, 'fp16': torch.float16}


def _compile(model, mode):
    if not mode or mode == 'none':
        return model
    return torch.compile(model, mode=None if mode == 'default' else mode, dynamic=False, fullgraph=False)


def to_channels_last(model):
    """channels_last for every 4D parameter and channels_last_3d for every 5D one (Module.to(memory_format=...) refuses
    models that mix 2D and 3D layers)."""
    for p in model.parameters():
        if p.dim() == 5:
            p.data = p.data.contiguous(memory_format=torch.channels_last_3d)
        elif p.dim() == 4:
            p.data = p.data.contiguous(memory_format=torch.channels_last)
    return model


def axis_starts(length, patch, stride):        # hecate.py's schedule
    return list(range(0, max(0, length - patch), stride)) + [max(0, length - patch)]


def mask_any_per_origin(mask, origins, p):
    """For each (y, x) origin: does mask[y:y+p, x:x+p] contain any True?  (integral image)"""
    m = np.pad(mask.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    H, Wd = mask.shape
    out = np.zeros(len(origins), bool)
    for i, (y, x) in enumerate(origins):
        y1, x1 = min(H, y + p), min(Wd, x + p)
        out[i] = (m[y1, x1] - m[y, x1] - m[y1, x] + m[y, x]) > 0
    return out


def quality_mask(render, layers=None, thr=None, tile=128):
    """fl/scripts/quality.py tile scores (per 128-px tile depth-profile contrast) -> per-pixel on-sheet mask.
    `layers` = slice of render layers to use (default: all, as quality.py does).
    thr default 0.10, or $FL_QUALITY_THR: on hazy scans the absolute 0.10 drops much papyrus (x_recto: tiles passing
    0343 93 %, 1545 76 %, 1218 48 %; block contrast 0.68 / 0.47 / 0.30)."""
    import os
    thr = float(os.environ.get('FL_QUALITY_THR', '0.10')) if thr is None else thr
    a = zarr.open(str(render), mode='r')['0']
    D = a.shape[0]
    sl = layers if layers is not None else slice(0, D)
    v = np.asarray(a[sl])
    Dv = v.shape[0]
    valid = v[Dv // 2] > 0
    med = float(np.median(v[Dv // 2][valid])) if valid.any() else 1.0
    H, Wd = valid.shape
    ny, nx = H // tile, Wd // tile
    sc = np.full((ny, nx), np.nan, np.float32)
    for i in range(ny):
        for j in range(nx):
            s = (slice(i * tile, (i + 1) * tile), slice(j * tile, (j + 1) * tile))
            vv = valid[s]
            if vv.mean() < 0.8:
                continue
            p = v[:, s[0], s[1]][:, vv].astype(np.float32).mean(1)
            sc[i, j] = (p.max() - p.min()) / max(med, 1.0)
    good = np.nan_to_num(sc, nan=-1) >= thr
    m = np.zeros((H, Wd), bool)
    m[:ny * tile, :nx * tile] = np.kron(good, np.ones((tile, tile), bool))
    return m, sc


# ----------------------------------------------------------------------------------------------- Hecate
def hecate_prep(render, native_um, z_center, nz=16, target_um=9.6):
    """In-process copy of third_party/pherc343-first-letters/ink/scripts/hecate_prep.py (same maths, byte-identical
    output) that reads only the layers it needs instead of the whole render."""
    a = zarr.open(str(render), mode='r')['0']
    D, H, Wd = a.shape
    step = target_um / native_um
    pos = z_center + (np.arange(nz) - (nz - 1) / 2.) * step
    if pos.min() < 0 or pos.max() > D - 1:
        raise ValueError(f'z window {pos.min():.2f}..{pos.max():.2f} outside 0..{D-1}')
    lo = np.floor(pos).astype(int); frac = (pos - lo).astype(np.float32); hi = np.minimum(lo + 1, D - 1)
    need = sorted(set(lo.tolist()) | set(hi.tolist()))
    blk = np.asarray(a[need[0]:need[-1] + 1])
    s = native_um / target_um
    nh, nw = max(1, int(round(H * s))), max(1, int(round(Wd * s)))
    interp = cv2.INTER_AREA if s < 0.95 else cv2.INTER_LINEAR
    out = np.empty((nz, nh, nw), np.uint8)
    for i in range(nz):
        zi = blk[lo[i] - need[0]].astype(np.float32) * (1 - frac[i]) + blk[hi[i] - need[0]].astype(np.float32) * frac[i]
        out[i] = np.clip(cv2.resize(zi, (nw, nh), interpolation=interp), 0, 255).astype(np.uint8)
    return out


class HecateFast:
    def __init__(self, precision='bf16', channels_last=False, compile=None, device='cuda'):
        spec = importlib.util.spec_from_file_location('hecate', f'{W}/models/ink9_ens/hecate.py')
        self.H = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.H)
        self.model = self.H.load_model(f'{W}/models/ink9_ens/hecate_9.6um.pth', device)
        self.precision, self.channels_last, self.device = precision, channels_last, device
        if channels_last:
            self.model = to_channels_last(self.model)
        self.fwd = _compile(self.model, compile)

    @torch.inference_mode()
    def predict(self, vol_np, reverse=True, stride=32, batch=8, skip_mask=None, rot=0, return_float=False):
        """vol_np: (16, H, W) uint8 Hecate input (hecate_prep output). Returns HxW uint8 (rint 255*p) like hecate.predict.
        skip_mask: optional HxW bool on the same grid; patches with no True pixel are not run (their area -> 0)."""
        d, p, _ = self.model.patch_size
        vol = torch.from_numpy(np.ascontiguousarray(vol_np)).to(self.device)
        if rot:
            vol = torch.rot90(vol, rot, (1, 2))
        if reverse:
            vol = vol.flip(0)
        D, h, w = vol.shape
        z0 = D // 2 - d // 2
        vol = vol[z0:z0 + d]                                     # nz=16 -> the whole stack
        win = torch.hann_window(p, periodic=False, device=self.device)
        win = win[:, None] * win[None, :]
        win = (win / win.max()).clamp_min(.001)
        origins = [(y, x) for y in axis_starts(h, p, stride) for x in axis_starts(w, p, stride)]
        num = torch.zeros((h, w), dtype=torch.float32, device=self.device)
        den = torch.zeros((h, w), dtype=torch.float32, device=self.device)
        for y, x in origins:
            ph, pw = min(p, h - y), min(p, w - x)
            den[y:y + ph, x:x + pw] += win[:ph, :pw]
        nonempty = mask_any_per_origin((vol.amax(0) > 0).cpu().numpy(), origins, p)
        if skip_mask is not None:
            sm = skip_mask if not rot else np.rot90(skip_mask, rot)
            nonempty &= mask_any_per_origin(np.ascontiguousarray(sm), origins, p)
        run = [o for o, k in zip(origins, nonempty) if k]
        dt = DT[self.precision]
        for b0 in range(0, len(run), batch):
            sel = run[b0:b0 + batch]
            patches = torch.zeros((batch, d, p, p), dtype=torch.uint8, device=self.device)
            for i, (y, x) in enumerate(sel):
                ph, pw = min(p, h - y), min(p, w - x)
                patches[i, :, :ph, :pw] = vol[:, y:y + ph, x:x + pw]
            support = patches.any(1)
            image = (patches.float() / self.model.divisor)[:, None]
            valid = support[:, None, None].expand(-1, 1, d, -1, -1)
            if self.channels_last:
                image = image.contiguous(memory_format=torch.channels_last_3d)
            with torch.autocast('cuda', dtype=dt or torch.float32, enabled=dt is not None):
                logits = self.fwd(image, valid)
            prob = logits.float().sigmoid()[:, 0] * support * win
            for i, (y, x) in enumerate(sel):
                ph, pw = min(p, h - y), min(p, w - x)
                num[y:y + ph, x:x + pw] += prob[i, :ph, :pw]
        out = (num / den).clamp(0, 1)
        if rot:
            out = torch.rot90(out, -rot, (0, 1))
        if return_float:
            return out.cpu().numpy(), len(run)
        return torch.round(out * 255).to(torch.uint8).cpu().numpy(), len(run)


# ----------------------------------------------------------------------------------------------- villa flat models
def _pct_sorted(s, q):
    """numpy 'linear' percentile of each row of an already sorted (B, N) tensor."""
    N = s.shape[1]
    idx = q / 100.0 * (N - 1)
    lo = int(math.floor(idx)); hi = min(lo + 1, N - 1); f = idx - lo
    return s[:, lo] * (1 - f) + s[:, hi] * f


def robust_norm_gpu(x):
    """vesuvius normalize_robust (percentile 1/99 clip, median/MAD*1.4826, fallbacks) on a (B, ...) float32 batch."""
    B = x.shape[0]
    v = x.reshape(B, -1)
    s, _ = torch.sort(v, dim=1)
    lower, upper = _pct_sorted(s, 1.0), _pct_sorted(s, 99.0)
    std_pre = v.std(1, unbiased=False)
    c = torch.maximum(torch.minimum(s, upper[:, None]), lower[:, None])   # clipping keeps the order
    med = _pct_sorted(c, 50.0)
    dev, _ = torch.sort((c - med[:, None]).abs(), dim=1)
    scaled = 1.4826 * _pct_sorted(dev, 50.0)
    span = (upper - lower) / 2.0
    bad = ~torch.isfinite(scaled) | (scaled < 1e-6)
    fb = torch.where(std_pre >= 1e-6, std_pre, torch.where(span >= 1e-6, span, torch.ones_like(span)))
    scaled = torch.where(bad, fb, scaled)
    out = (torch.maximum(torch.minimum(v, upper[:, None]), lower[:, None]) - med[:, None]) / scaled[:, None]
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).reshape(x.shape)


TTA = {'none': [()], 'flip2': [(), (1, 2)], 'mirror4': [(), (1,), (2,), (1, 2)], 'yflip2': [(), (1,)]}


class VillaFast:
    """Villa flat ink model (Reader v2, dense_native, ink_9um) with villa's exact pre/post-processing on the GPU."""

    def __init__(self, ckpt, amp='checkpoint', channels_last=False, compile=None, device='cuda', vectorized=False):
        from vesuvius.ink_detection.inference import infer as I
        self.vectorized = vectorized
        self.I = I
        args = I.normalize_inference_paths(I.parse_args(['/dev/null', str(ckpt), '/dev/null', '--no-compile']))
        cfg = I.configure_model(args)
        self.cfg, self.device = cfg, device
        self.model = cfg.model.to(device).eval()
        self.amp = cfg.amp_dtype if amp == 'checkpoint' else DT[amp]
        self.channels_last = channels_last
        if channels_last:
            self.model = to_channels_last(self.model)
        self.fwd = _compile(self.model, compile)
        assert cfg.preprocessing == 'tifxyz_robust', cfg.preprocessing

    def load_window(self, render, layer_start, direction):
        root = self.I.open_volume_root(str(render))
        vol = self.I.select_volume_level(root, '0', source=str(render))
        depth, H, Wd = vol.shape
        idx = self.I.select_layer_indices(depth, layer_start=layer_start, layer_end=layer_start + self.cfg.input_depth,
                                          output_depth=self.cfg.input_depth, direction=direction)
        a = np.asarray(vol[int(idx.min()):int(idx.max()) + 1])
        a = a[idx - idx.min()]
        return root, a

    @torch.inference_mode()
    def predict(self, render, layer_start, direction='reverse', stride=None, overlap=0.5, batch=16, tta='none',
                skip_mask=None, weight_mask=None, window=None, blend='hann'):
        """Returns (HxW uint8 map encoded like villa's TIFF, n_patches_run)."""
        I, p = self.I, self.cfg.patch_size
        root, win_np = window if window is not None else self.load_window(render, layer_start, direction)
        Z, H, Wd = win_np.shape
        stride = stride or I.resolve_patch_stride(patch_size=p, overlap=overlap, explicit_stride=None)
        occ, occ_scale, _ = I.build_lowres_block_mask(root, height=H, width=Wd, user_mask=weight_mask)
        blocks = I.iter_blocks((H, Wd), p, stride, occ, occ_scale)
        origins = [(b.y0, b.x0) for b in blocks]
        keep = mask_any_per_origin(win_np.max(0) > 0, origins, p)           # villa skips raw-empty patches
        if skip_mask is not None:
            keep &= mask_any_per_origin(skip_mask, origins, p)
        blocks = [b for b, k in zip(blocks, keep) if k]
        vol = torch.from_numpy(np.ascontiguousarray(win_np)).to(self.device)
        wmap = I.compute_importance_map_2d(patch_size=(p, p), mode=blend).to(self.device)
        wm = None if weight_mask is None else torch.from_numpy(weight_mask.astype(np.float32)).to(self.device)
        num = torch.zeros((H, Wd), dtype=torch.float32, device=self.device)
        den = torch.zeros((H, Wd), dtype=torch.float32, device=self.device)
        variants = TTA[tta]
        ar = torch.arange(p, device=self.device)
        for b0 in range(0, len(blocks), batch):
            sel = blocks[b0:b0 + batch]
            if self.vectorized:
                # one gather for the whole batch (padded with copies of the last block, whose weights are dropped below)
                ys = torch.tensor([b.y0 for b in sel] + [sel[-1].y0] * (batch - len(sel)), device=self.device)
                xs = torch.tensor([b.x0 for b in sel] + [sel[-1].x0] * (batch - len(sel)), device=self.device)
                vh = torch.tensor([b.valid_h for b in sel] + [0] * (batch - len(sel)), device=self.device)
                vw = torch.tensor([b.valid_w for b in sel] + [0] * (batch - len(sel)), device=self.device)
                iy = ys[:, None] + ar[None]; ix = xs[:, None] + ar[None]                       # B,p
                my = ar[None] < vh[:, None]; mx = ar[None] < vw[:, None]
                iy = iy.clamp_max(H - 1); ix = ix.clamp_max(Wd - 1)
                patches = vol[:, iy[:, :, None], ix[:, None, :]].permute(1, 0, 2, 3).float()   # B,Z,p,p
                patches = patches * (my[:, None, :, None] & mx[:, None, None, :])
            else:
                patches = torch.zeros((batch, Z, p, p), dtype=torch.float32, device=self.device)
                for i, b in enumerate(sel):
                    patches[i, :, :b.valid_h, :b.valid_w] = vol[:, b.y0:b.y0 + b.valid_h, b.x0:b.x0 + b.valid_w]
            img = robust_norm_gpu(patches)[:, None]                          # B,1,Z,p,p
            with torch.autocast('cuda', dtype=self.amp or torch.float32, enabled=self.amp is not None):
                acc = None
                for axes in variants:
                    x = img if not axes else torch.flip(img, [a + 2 for a in axes])
                    if self.channels_last:
                        x = x.contiguous(memory_format=torch.channels_last_3d)
                    lg = self.fwd(x)
                    pr = lg.float().sigmoid()
                    if pr.shape[-2:] != (p, p):
                        pr = F.interpolate(pr, size=(p, p), mode='bilinear', align_corners=False)
                    dims = [2 if a == 1 else 3 for a in axes if a in (1, 2)]
                    pr = torch.flip(pr, dims) if dims else pr
                    acc = pr if acc is None else acc + pr
                prob = (acc / len(variants))[:, 0]
            if self.vectorized:
                m2 = (my[:, :, None] & mx[:, None, :]).float()                                  # B,p,p valid & real
                wgt = wmap[None] * m2
                if wm is not None:
                    wgt = wgt * wm[iy[:, :, None], ix[:, None, :]]
                flat = (iy[:, :, None] * Wd + ix[:, None, :]).reshape(-1)
                num.view(-1).index_put_((flat,), (prob * wgt).reshape(-1), accumulate=True)
                den.view(-1).index_put_((flat,), wgt.reshape(-1), accumulate=True)
            else:
                for i, b in enumerate(sel):
                    wgt = wmap[:b.valid_h, :b.valid_w]
                    if wm is not None:
                        wgt = wgt * wm[b.y0:b.y0 + b.valid_h, b.x0:b.x0 + b.valid_w]
                    num[b.y0:b.y0 + b.valid_h, b.x0:b.x0 + b.valid_w] += prob[i, :b.valid_h, :b.valid_w] * wgt
                    den[b.y0:b.y0 + b.valid_h, b.x0:b.x0 + b.valid_w] += wgt
        out = torch.where(den > 1e-6, num / den.clamp_min(1e-12), torch.zeros_like(num)).clamp(0, 1) * 255
        return out.to(torch.uint8).cpu().numpy(), len(blocks)


# ----------------------------------------------------------------------------------------------- v8-in
class V8Fast:
    def __init__(self, amp='fp16', channels_last=False, compile=None, device='cuda'):
        sys.path.insert(0, str(W / 'third_party/v8in'))
        from ink8um import InkDetector
        from ink8um.inference import gkern, coverage_mask, tile_positions
        self.gkern, self.coverage_mask, self.tile_positions = gkern, coverage_mask, tile_positions
        self.model = InkDetector.from_pretrained(str(W / 'third_party/v8in')).to(device).eval()
        self.amp, self.device, self.channels_last = DT[amp], device, channels_last
        if channels_last:
            self.model = to_channels_last(self.model)
        self.fwd = _compile(self.model, compile)

    @staticmethod
    def stack(render, d=0):
        a = zarr.open(str(render), mode='r')['0']
        n = a.shape[0]; c = n // 2; s = c - 12 + d
        return np.minimum(np.asarray(a[s:s + 24]), 200).astype(np.uint8)       # (24, H, W)

    @torch.inference_mode()
    def predict(self, st_zyx, reverse=True, stride=21, batch=16, skip_mask=None):
        """st_zyx: (24, H, W) uint8 clipped at 200. Same tile rule/blend as ink8um.predict_stack. Returns (float32 HxW, n)."""
        t = 64
        if reverse:
            st_zyx = st_zyx[::-1]
        st_zyx = np.ascontiguousarray(st_zyx)
        H, Wd = st_zyx.shape[1:]
        cov = self.coverage_mask(st_zyx.transpose(1, 2, 0))
        pts = self.tile_positions(cov, t, stride)                                 # (x, y)
        if skip_mask is not None:
            keep = mask_any_per_origin(skip_mask, [(y, x) for x, y in pts], t)
            pts = [q for q, k in zip(pts, keep) if k]
        vol = torch.from_numpy(st_zyx).to(self.device)
        blend = self.gkern(t, 1); blend = torch.from_numpy((blend / blend.max()).astype(np.float32)).to(self.device)
        num = torch.zeros((H, Wd), dtype=torch.float32, device=self.device)
        cnt = torch.zeros((H, Wd), dtype=torch.float32, device=self.device)
        for b0 in range(0, len(pts), batch):
            sel = pts[b0:b0 + batch]
            tiles = torch.zeros((batch, 24, t, t), dtype=torch.float32, device=self.device)
            for i, (x, y) in enumerate(sel):
                tiles[i] = vol[:, y:y + t, x:x + t]
            tiles = tiles * np.float32(1.0 / 255.0)
            if self.channels_last:
                tiles = tiles[:, None].contiguous(memory_format=torch.channels_last_3d)
            with torch.autocast('cuda', dtype=self.amp or torch.float32, enabled=self.amp is not None):
                lg = self.fwd(tiles)
            pr = torch.sigmoid(lg).float()
            if pr.shape[-2:] != (t, t):
                pr = F.interpolate(pr, size=(t, t), mode='bilinear')
            pr = pr[:, 0]
            for i, (x, y) in enumerate(sel):
                num[y:y + t, x:x + t] += pr[i] * blend
                cnt[y:y + t, x:x + t] += 1.0
        out = torch.where(cnt != 0, num / cnt.clamp_min(1), torch.zeros_like(num)).clamp(0, 1)
        return torch.nan_to_num(out).cpu().numpy(), len(pts)
