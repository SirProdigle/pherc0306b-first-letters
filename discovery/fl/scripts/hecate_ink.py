"""In-process Hecate for the screening ink_server (load once, call per render).

    from hecate_ink import Hecate
    hec = Hecate()                                   # loads models/ink9_ens/hecate_9.6um.pth on cuda
    fwd, rev = hec.run(render_zarr, n_layers=31, d=0, native_um=8.64, out_prefix='.../<name>__hecate_d0', stride=48, tta=1)

Writes <out_prefix>.png (forward) and <out_prefix>_reverse.png on the RENDER grid (uint8 = 255*p), so they can be
z-scored/fused with the villa TIFFs directly. stride 48 + bf16 ~= fp32 stride 32 quality (AUC .759 vs .771 on PHerc0343)
at ~3x less time; tta=4 adds 4 rotations (+.010 to +.014 AUC on PHerc0841 / PHerc0343) at 4x the time.
"""
import importlib.util, subprocess, tempfile, shutil, os
import numpy as np, zarr, cv2

W = '/home/liam/projects/vesuvius-work'


class Hecate:
    def __init__(self, device='cuda'):
        spec = importlib.util.spec_from_file_location('hecate', f'{W}/models/ink9_ens/hecate.py')
        self.H = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.H)
        self.model = self.H.load_model(f'{W}/models/ink9_ens/hecate_9.6um.pth', device)

    def run(self, render, n_layers=31, d=0, native_um=8.64, out_prefix=None, stride=48, tta=1, batch=4):
        zc = (n_layers - 1) / 2 + d
        tmp = tempfile.mkdtemp(dir=f'{W}/fl/tmp')
        try:
            subprocess.run([f'{W}/envs/vcpy/bin/python', f'{W}/third_party/pherc343-first-letters/ink/scripts/hecate_prep.py',
                            '--src', str(render), '--out', f'{tmp}/prep.zarr', '--native-um', str(native_um), '--native-um-z', str(native_um),
                            '--target-um', '9.6', '--z-center', f'{zc:.2f}', '--nz', '16'],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            vol = np.asarray(zarr.open(f'{tmp}/prep.zarr', mode='r')['0'][:])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        full = zarr.open(str(render), mode='r')['0']
        H, Wd = full.shape[-2:]
        outs = []
        for rev in (False, True):
            acc = []
            for k in range(tta):
                v = np.ascontiguousarray(np.rot90(vol, k, axes=(1, 2)))
                o = np.zeros(v.shape[-2:], np.uint8)
                self.H.predict(self.model, v, o, reverse=rev, stride=stride, batch_size=batch, precision='bf16')
                acc.append(np.rot90(o, -k).astype(np.float32))
            p = cv2.resize(np.mean(acc, 0), (Wd, H), interpolation=cv2.INTER_LINEAR)
            p = np.clip(np.rint(p), 0, 255).astype(np.uint8)
            path = f'{out_prefix}{"_reverse" if rev else ""}.png'
            cv2.imwrite(path, p)
            outs.append(path)
        return outs
