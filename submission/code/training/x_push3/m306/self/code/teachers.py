"""teachers.py [readers|v8] [REGION,..] : cleared teacher maps on the SELF pool regions.

Teachers (cleared only; never Hecate): Reader V3, TRACE v3B, Reader v2 (x_req/20261020/V3/run_maps.py map3 recipe: 17-layer window
centred on c, d -1/0/+1 averaged, mirror4, overlap 0.5; sha256-checked, weights_only load) and base v8-in (aclick/bench/infer.py
predict_array: 24 layers layer0.., stored order, clip 200, 64 px tiles, +-3 sigma stitch, TTA 1; stride V8_STRIDE, 32 used
for the pool to save GPU: each pixel still gets ~4 tiles, ripple at the window midpoint 0.32 of peak).
Zone hygiene: the region volume is read on the bbox of its pool mask (+192 px) and every pixel NOT allowed at pad 0
(check rule: < 2.2 mm from the lead-A zone, or no mesh) is set to 0 on all layers before any teacher sees it, so no CT from the
zone or its 2 mm buffer can influence a teacher value.  v8-in runs only tiles that overlap the pool mask.
Outputs m306/self/teach/<r>_<model>.npy (float16 on the bbox; NaN = no tile for v8) + <r>_teach.json (bbox, recipes).
Resumable: v8 in 1024-row bands (parts), exits 3 after --max-minutes (gpu_run.sh retries).  VRAM cap 9 GB."""
import sys, os, json, time, importlib.util
import numpy as np, torch, zarr
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import pool, masks   # noqa: E402
W = '/home/liam/projects/vesuvius-work'; P3 = pool.P3
TEACH = f'{pool.SELF}/teach'; os.makedirs(TEACH, exist_ok=True)
T0 = time.time(); MAXMIN = float(os.environ.get('MAXMIN', 25))
VRAM_GB = float(os.environ.get('VRAM_GB', 9.0))
torch.cuda.set_per_process_memory_fraction(min(1.0, VRAM_GB * 1024 ** 3 / torch.cuda.get_device_properties(0).total_memory))
R = masks.regions()


def region_vol(r):
    g = R[r]; m = masks.load(r); a0, a7, qf = masks.derived(m)
    pm = a7 & m['cov24'] & qf
    ys, xs = np.nonzero(pm); H, Wd = pm.shape; M = 192
    bb = (max(0, ys.min() - M), min(H, ys.max() + 1 + M), max(0, xs.min() - M), min(Wd, xs.max() + 1 + M))
    y0, y1, x0, x1 = bb
    vol = np.asarray(zarr.open(g['zarr'], mode='r')['0'][:, y0:y1, x0:x1])
    vol[:, ~a0[y0:y1, x0:x1]] = 0
    return g, bb, vol, pm[y0:y1, x0:x1]


def meta(r, bb, extra):
    p = f'{TEACH}/{r}_teach.json'
    j = json.load(open(p)) if os.path.exists(p) else {}
    j.update(dict(region=r, zarr=R[r]['zarr'], bbox=[int(v) for v in bb], c=R[r]['c'], layer0=R[r]['layer0'], face=R[r]['face'],
                  zone_rule='CT zeroed where check pad-0 rule fails (< 2.2 mm from the zone or no mesh)'))
    j.setdefault('models', {}).update(extra)
    json.dump(j, open(p, 'w'), indent=1)


def readers(regions, models=('v3', 'trace', 'rv2')):
    spec = importlib.util.spec_from_file_location('run_maps', f'{W}/x_req/20261020/V3/run_maps.py')
    RM = importlib.util.module_from_spec(spec); spec.loader.exec_module(RM)
    for mname in models:
        vf = None
        for r in regions:
            out = f'{TEACH}/{r}_{mname}.npy'
            if os.path.exists(out): continue
            if time.time() - T0 > MAXMIN * 60: print('chunk end', flush=True); sys.exit(3)
            vf = vf or RM.load(mname)
            t = time.time(); g, bb, vol, _ = region_vol(r)
            face = 'forward' if g['face'] == 'fwd' else 'reverse'
            p = RM.map3(vf, vol, g['c'], face, RM.RECIPE[mname])
            np.save(out + '.tmp.npy', p.astype(np.float16)); os.replace(out + '.tmp.npy', out)
            meta(r, bb, {mname: dict(ckpt=RM.CK[mname][0], depths=list(RM.RECIPE[mname][0]), tta=RM.RECIPE[mname][1],
                                     recipe='x_req/20261020/V3/run_maps.py map3')})
            print(f'{r} {mname} {p.shape} {time.time() - t:.0f}s peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GB', flush=True)
        del vf; torch.cuda.empty_cache()


def v8(regions, tta=1, band=int(os.environ.get('V8_BAND', 1024)), stride=int(os.environ.get('V8_STRIDE', 21))):
    sys.path.insert(0, f'{P3}/aclick/bench'); import infer as I
    model = None
    for r in regions:
        out = f'{TEACH}/{r}_v8.npy'
        if os.path.exists(out): continue
        g, bb, vol, pm = region_vol(r)
        l0 = g['layer0']; v24 = np.ascontiguousarray(vol[l0:l0 + 24]); del vol
        if g['face'] != 'fwd': v24 = np.ascontiguousarray(v24[::-1])
        H, Wd = pm.shape; PD = f'{TEACH}/{r}_v8.parts'; os.makedirs(PD, exist_ok=True)
        for b0 in range(0, H, band):
            pf = f'{PD}/b{b0:05d}.npy'
            if os.path.exists(pf): continue
            if time.time() - T0 > MAXMIN * 60: print('chunk end', flush=True); sys.exit(3)
            if model is None: model, info = I.load_model('base')
            t = time.time(); b1 = min(H, b0 + band)
            m = I.predict_array(model, v24, cov=v24[12] > 0, tta=tta, nsig=3, stride=stride, roi=(b0, b1, 0, Wd), roi_mask=pm, batch=8)
            np.save(pf + '.tmp.npy', m[b0:b1].astype(np.float16)); os.replace(pf + '.tmp.npy', pf)
            print(f'{r} v8 band {b0}-{b1} {time.time() - t:.0f}s peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GB', flush=True)
        full = np.concatenate([np.load(f'{PD}/b{b0:05d}.npy') for b0 in range(0, H, band)], 0)
        np.save(out + '.tmp.npy', full); os.replace(out + '.tmp.npy', out)
        meta(r, bb, {'v8': dict(ckpt='base', sha256=I.V8_SHA, tta=tta, nsig=3, stride=stride, layers=f'{l0}..{l0 + 23} stored order',
                                recipe='aclick/bench/infer.py predict_array, tiles overlapping the pool mask only')})
        import shutil; shutil.rmtree(PD)
        print(f'{r} v8 done {full.shape}', flush=True)


if __name__ == '__main__':
    what = sys.argv[1]; regs = sys.argv[2].split(',') if len(sys.argv) > 2 else list(R)
    if what == 'readers': readers(regs)
    elif what == 'v8': v8(regs)
    print('all done', flush=True)
