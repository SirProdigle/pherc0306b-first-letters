"""smooth_umb.py IN.json OUT.json [--sigma 75] [--step 25] : per-slice interpolate the umbilicus over its z range, Gaussian-smooth
x/y with sigma slices (mode nearest), resample every --step slices. Pre-smoothing makes the raw axis the umbilicus loss
reads agree with the sigma-75 axis UmbilicusTransform uses (villa issue #2029; fix PR #2030)."""
import json, sys
import numpy as np
from scipy.ndimage import gaussian_filter1d
a = sys.argv[1:]; sig = float(a[a.index('--sigma') + 1]) if '--sigma' in a else 75.0; st = int(a[a.index('--step') + 1]) if '--step' in a else 25
cp = sorted(json.load(open(a[0]))['control_points'], key=lambda q: q['z'])
z = np.array([q['z'] for q in cp], float); x = np.array([q['x'] for q in cp], float); y = np.array([q['y'] for q in cp], float)
zs = np.arange(int(z.min()), int(z.max()) + 1)
xs = gaussian_filter1d(np.interp(zs, z, x), sig, mode='nearest'); ys = gaussian_filter1d(np.interp(zs, z, y), sig, mode='nearest')
raw_x, raw_y = np.interp(zs, z, x), np.interp(zs, z, y); d = np.hypot(xs - raw_x, ys - raw_y)
print(f'raw vs smoothed: median {np.median(d):.1f} vox, >5 vox on {(d > 5).mean():.0%} of slices, max {d.max():.0f}')
out = [{'x': float(xs[i]), 'y': float(ys[i]), 'z': float(zs[i]), 'score': 100} for i in range(0, len(zs), st)]
json.dump({'control_points': out, 'metadata': {'source': a[0], 'gaussian_sigma_slices': sig, 'step': st}}, open(a[1], 'w'), indent=1)
