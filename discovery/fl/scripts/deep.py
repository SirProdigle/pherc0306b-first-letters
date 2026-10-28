"""deep.py RENDER.zarr OUTDIR [--depths -2,-1,0,1,2] [--tta 1] : second-pass ink for one candidate render.
Per depth: dense_native (villa, --tta-mirror) + Hecate (stride 48, tta) both sides, fused 2:1 with fuse_view.py.
Writes OUTDIR/d{d}_{fwd|reverse}_overlay.png and a montage OUTDIR/montage_{fwd|reverse}.jpg (depth columns)."""
import sys, os, subprocess, json
from pathlib import Path
import numpy as np, cv2, zarr
W = Path('/home/liam/projects/vesuvius-work')
sys.path.insert(0, str(W / 'fl/scripts'))
from hecate_ink import Hecate

r, out = str(Path(sys.argv[1]).resolve()), Path(sys.argv[2]).resolve()
depths = [int(v) for v in (sys.argv[sys.argv.index('--depths') + 1] if '--depths' in sys.argv else '-2,-1,0,1,2').split(',')]
tta = int(sys.argv[sys.argv.index('--tta') + 1]) if '--tta' in sys.argv else 1
um = float(sys.argv[sys.argv.index('--um') + 1]) if '--um' in sys.argv else 8.64
out.mkdir(parents=True, exist_ok=True)
N = zarr.open(r, mode='r')['0'].shape[0]
name = Path(r).name[:-5]
PY = str(W / 'envs/vcpy/bin/python')
hec = Hecate()
for d in depths:
    s = N // 2 + d - 8
    dn = out / f'dn_d{d}.tif'
    if not (out / f'dn_d{d}_reverse.tif').exists():
        subprocess.run([PY, '-m', 'vesuvius.ink_detection.inference.infer', r, str(W / 'models/ink9_ens/dense_native-016000.rebuilt.pth'),
                        str(dn), '--overlap', '0.5', '--blend-mode', 'hann', '--batch-size', '8', '--direction', 'both',
                        '--layer-start', str(s), '--layer-end', str(s + 17), '--tta-mirror', '--num-workers', '2', '--no-compile'],
                       cwd=str(W / 'fl/tmp'), env=dict(os.environ, PYTHONPATH=str(Path('/home/liam/projects/villa/vesuvius/src'))),
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=False)
    ft = W / 'models/ink9_ens/dn343ft.pth'
    if ft.exists() and not (out / f'ft_d{d}_reverse.tif').exists():
        subprocess.run([PY, '-m', 'vesuvius.ink_detection.inference.infer', r, str(ft), str(out / f'ft_d{d}.tif'), '--overlap', '0.5',
                        '--blend-mode', 'hann', '--batch-size', '8', '--direction', 'both', '--layer-start', str(s), '--layer-end', str(s + 17),
                        '--tta-mirror', '--num-workers', '2', '--no-compile'], cwd=str(W / 'fl/tmp'),
                       env=dict(os.environ, PYTHONPATH=str(Path('/home/liam/projects/villa/vesuvius/src'))),
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=False)
    hp = out / f'hec_d{d}'
    if not Path(f'{hp}_reverse.png').exists():
        hec.run(r, n_layers=N, d=d, native_um=um, out_prefix=str(hp), stride=32, tta=tta)   # s32: +.012 AUC, +2 letters vs s48
    for suf, tag in (('', 'fwd'), ('_reverse', 'reverse')):
        items = [f'2:{hp}{suf}.png'] + ([f'1:{out}/dn_d{d}{suf}.tif'] if (out / f'dn_d{d}{suf}.tif').exists() else [])
        subprocess.run([PY, str(W / 'fl/scripts/fuse_view.py'), str(out / f'd{d}_{tag}'), r, str(N // 2 + d), *items, '--down', '3'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    print('depth', d, 'done', flush=True)
for tag in ('fwd', 'reverse'):
    ims = []
    for d in depths:
        a = cv2.imread(str(out / f'd{d}_{tag}_overlay.png'))
        if a is None:
            continue
        cv2.putText(a, f'd={d:+d} {tag}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        ims.append(a)
    if ims:
        cv2.imwrite(str(out / f'montage_{tag}.jpg'), np.hstack(ims), [cv2.IMWRITE_JPEG_QUALITY, 88])
print('ALL DONE', flush=True)
