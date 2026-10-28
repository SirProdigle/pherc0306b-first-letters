"""fastpipe.py: GPU-resident ink for First Letters renders (Hecate + Reader v2 [+ dn343ft] [+ v8-in]) with fused z-maps.

    fastpipe.py RENDER.zarr [RENDER2.zarr ...] --out OUTDIR [--recipe fast|accurate] [--dirs reverse|both] [--d 0]
                [--um 8.64] [--mask] [--weights hec=1,rv2=0.5,...]

Models are loaded once and every render is processed in turn (like fl/scripts/ink_server.py), so it can be used as a
batch tool or imported (`Pipe(recipe).run(render, out)`).  Per render and side it writes
  OUT/<name>__<model>_d<D>{_reverse}.png    uint8 255*p on the render grid (same encoding as the production maps)
  OUT/<name>_d<D>_<fwd|reverse>_z.npy       fused z (float16), same maths as fl/scripts/fuse_view.py
                                            (weighted mean of per-map z over on-sheet tiles, re-z-scored, 48-px border blanked)
  OUT/<name>_d<D>_<fwd|reverse>_overlay.png fuse_view-style overlay at 1/3 resolution
  OUT/<name>_timing.json                    seconds per model / side
Recipes (weights tuned on PHerc0343 = the test set; PHerc0841 w00 as the independent check):
  fast     : Hecate stride 48 + 0.5 Reader v2 + 0.5 dn343ft (no TTA)          ~ production cost, better d'/letters
  balanced : Hecate stride 32 + 0.5 Reader v2 + 0.5 dn343ft (no TTA)
  accurate : Hecate stride 32 depth ensemble d-1/d/d+1 + 0.5 Reader v2 with 4x mirror TTA
--mask skips Hecate/Reader-v2 patches with no on-sheet quality tile (fl/scripts/quality.py rule): faster on typical
screening patches, but the off-sheet area is then blank (0) -- fine for screening, not for figures.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, cv2, zarr, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fastink as FI

W = Path('/home/liam/projects/vesuvius-work')
CK = {'rv2': W / 'models/community/reader-v2/reader-v2-step040000.pth', 'ft': W / 'models/ink9_ens/dn343ft.pth',
      'dn': W / 'models/ink9_ens/dense_native-016000.rebuilt.pth'}
RECIPES = {
    # PHerc0343 (Erwin) auc / d' / letters plain -> with --mask ; PHerc0841 auc / d'   (x_research2/speed recipes_eval.json)
    'fast':     dict(models=['hec', 'rv2', 'ft'], weights={'hec': 1.0, 'rv2': 0.5, 'ft': 0.5}, hec_stride=48, hec_depths=[0],
                     rv2_windows=[0], tta='none', v8=False),          # .752/1.65/14 -> .780/1.76/14 ; .811/2.08
    'balanced': dict(models=['hec', 'rv2', 'ft'], weights={'hec': 1.0, 'rv2': 0.5, 'ft': 0.5}, hec_stride=32, hec_depths=[0],
                     rv2_windows=[0], tta='none', v8=False),          # .752/1.69/14 -> .780/1.79/14 ; .808/2.08
    'accurate': dict(models=['hec', 'rv2'], weights={'hec': 1.0, 'rv2': 0.5}, hec_stride=32, hec_depths=[-1, 0, 1],
                     rv2_windows=[0], tta='mirror4', v8=False),       # .748/1.82/14 -> .780/1.95/14 ; .816/2.17
}


def zscore_stat(render, layer):
    """fuse_view.py's statistics mask: valid pixels, restricted to on-sheet quality tiles when they cover >20 %."""
    ct = np.asarray(zarr.open(str(render), mode='r')['0'][layer]).astype(np.float32)
    valid = ct > 0
    D = zarr.open(str(render), mode='r')['0'].shape[0]
    q, _ = FI.quality_mask(render, layers=slice(max(0, D // 2 - 15), D // 2 + 16))
    stat = (q & valid) if (q & valid).sum() > 0.2 * valid.sum() else valid
    return ct, valid, stat, q


def fuse(maps, weights, ct, valid, stat):
    acc, ws = 0.0, 0.0
    for k, a in maps.items():
        a = a.astype(np.float32)
        if a.shape != ct.shape:
            a = cv2.resize(a, (ct.shape[1], ct.shape[0]), interpolation=cv2.INTER_LINEAR)
        v = a[stat]
        acc = acc + weights[k] * (a - v.mean()) / (v.std() + 1e-6); ws += weights[k]
    z = acc / ws
    z = (z - z[stat].mean()) / (z[stat].std() + 1e-6); z[~valid] = 0
    B = 48
    z[:B] = 0; z[-B:] = 0; z[:, :B] = 0; z[:, -B:] = 0
    return z


def overlay(ct, z, down=3):
    c = cv2.resize(ct, (ct.shape[1] // down, ct.shape[0] // down), interpolation=cv2.INTER_AREA)
    zz = cv2.resize(z, (c.shape[1], c.shape[0]), interpolation=cv2.INTER_AREA)
    vv = c > 0
    lo, hi = np.percentile(c[vv], [1, 99.5]) if vv.any() else (0, 255)
    g = np.clip((c - lo) / (hi - lo + 1e-6), 0, 1) * 0.45
    a = np.clip((zz - 0.9) / 1.8, 0, 1)
    return (np.stack([g * (1 - a) + a * .35, g * (1 - a) + a * .85, g * (1 - a) + a * .95], -1) * 255).astype(np.uint8)


class Pipe:
    def __init__(self, recipe='balanced', weights=None, vram_gb=10):
        self.cfg = dict(RECIPES[recipe]); self.recipe = recipe
        if weights:
            self.cfg['weights'] = dict(self.cfg['weights'], **weights)
            self.cfg['models'] = [m for m in self.cfg['weights'] if self.cfg['weights'][m] > 0]
        torch.backends.cudnn.benchmark = True
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(min(1.0, vram_gb * 2**30 / torch.cuda.get_device_properties(0).total_memory))
        self.M = {}
        for m in self.cfg['models']:
            self.M[m] = FI.HecateFast('bf16') if m == 'hec' else FI.V8Fast('fp16') if m == 'v8' else FI.VillaFast(CK[m])

    def run(self, render, out, d=0, dirs=('forward', 'reverse'), um=8.64, mask=False):
        render, out = Path(render), Path(out); out.mkdir(parents=True, exist_ok=True)
        name = render.name[:-5] if render.name.endswith('.zarr') else render.name
        D, H, Wd = zarr.open(str(render), mode='r')['0'].shape
        c = D // 2
        ct, valid, stat, q = zscore_stat(render, c + d)
        timing = {}
        maps = {dn: {} for dn in dirs}
        for m in self.cfg['models']:
            t = time.time()
            if m == 'hec':
                vols = [FI.hecate_prep(render, um, (D - 1) / 2 + d + dd) for dd in self.cfg['hec_depths']]
                sm = cv2.resize(q.astype(np.uint8), (vols[0].shape[2], vols[0].shape[1]), interpolation=cv2.INTER_NEAREST) > 0 if mask else None
                for dn in dirs:
                    acc = [self.M[m].predict(v, reverse=dn == 'reverse', stride=self.cfg['hec_stride'], batch=16, skip_mask=sm, return_float=True)[0]
                           for v in vols]
                    p = cv2.resize(np.mean(acc, 0), (Wd, H), interpolation=cv2.INTER_LINEAR)
                    maps[dn][m] = np.clip(np.rint(p * 255), 0, 255).astype(np.uint8)
            elif m == 'v8':
                st = self.M[m].stack(render, d + 2)            # validated v8-in window: d+2, reverse
                for dn in dirs:
                    p, _ = self.M[m].predict(st, reverse=dn == 'reverse', stride=32, batch=8, skip_mask=q if mask else None)
                    maps[dn][m] = (p * 255).astype(np.uint8)
            else:
                wins = self.cfg['rv2_windows'] if m == 'rv2' else [0]
                for dn in dirs:
                    acc = []
                    for w in wins:
                        o, _ = self.M[m].predict(render, c + d - 8 + w, dn, overlap=0.5, batch=32, tta=self.cfg['tta'],
                                                 skip_mask=q if mask else None)
                        acc.append(o.astype(np.float32))
                    maps[dn][m] = np.clip(np.mean(acc, 0), 0, 255).astype(np.uint8)
            torch.cuda.synchronize(); timing[m] = round(time.time() - t, 2)
        for dn in dirs:
            suf = '_reverse' if dn == 'reverse' else ''
            for m, a in maps[dn].items():
                cv2.imwrite(str(out / f'{name}__{m}_d{d}{suf}.png'), a)
            z = fuse(maps[dn], self.cfg['weights'], ct, valid, stat)
            tag = 'reverse' if dn == 'reverse' else 'fwd'
            np.save(out / f'{name}_d{d}_{tag}_z.npy', z.astype(np.float16))
            cv2.imwrite(str(out / f'{name}_d{d}_{tag}_overlay.png'), overlay(ct, z))
        json.dump(dict(recipe=self.recipe, weights=self.cfg['weights'], dirs=list(dirs), mask=mask, seconds=timing,
                       grid_cm2=round(H * Wd * (um * 1e-4) ** 2, 3)), open(out / f'{name}_timing.json', 'w'), indent=1)
        return timing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('renders', nargs='+'); ap.add_argument('--out', required=True)
    ap.add_argument('--recipe', default='balanced', choices=list(RECIPES)); ap.add_argument('--dirs', default='both')
    ap.add_argument('--d', type=int, default=0); ap.add_argument('--um', type=float, default=8.64)
    ap.add_argument('--mask', action='store_true'); ap.add_argument('--weights', default='')
    ap.add_argument('--vram-gb', type=float, default=10)
    a = ap.parse_args()
    w = {k: float(v) for k, v in (p.split('=') for p in a.weights.split(',') if p)}
    pipe = Pipe(a.recipe, w or None, a.vram_gb)
    dirs = ('forward', 'reverse') if a.dirs == 'both' else (a.dirs,)
    for r in a.renders:
        t = time.time()
        tm = pipe.run(r, Path(a.out) / (Path(r).name[:-5] if len(a.renders) > 1 else ''), a.d, dirs, a.um, a.mask)
        print(f'{r}: {time.time() - t:.1f}s {tm}', flush=True)


if __name__ == '__main__':
    main()
