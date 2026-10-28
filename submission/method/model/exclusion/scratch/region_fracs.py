import sys, json, numpy as np
sys.path.insert(0, '/home/liam/projects/vesuvius-work/.claude/worktrees/leadA-push3/x_push3/aclick/p306/exclusion')
import check
W = '/home/liam/projects/vesuvius-work'
R = {
 'w050_row': (f'{W}/x_focus306/renders/w050.zarr', (1000, 1650, 3000, 6600)),
 'w050_full': (f'{W}/x_focus306/renders/w050.zarr', None),
 'A_own_r41_row': (f'{W}/x_focus/renders/A_own_r41.zarr', (2400, 3900, 1800, 4500)),
 'A_own_r41_full': (f'{W}/x_focus/renders/A_own_r41.zarr', None),
 'w052_lower': (f'{W}/x_focus306/renders/w052.zarr', (1300, 1750, 3300, 6300)),
 'w053_rows': (f'{W}/x_focus306/renders/w053.zarr', (1150, 1700, 3700, 6500)),
 'w059_band': (f'{W}/x_req/20261020b/F6Y/w059.zarr', (414, 1937, 3252, 6772)),
 'w109_rows': (f'{W}/x_refine/trials/P5_scour_0306B/crops/p3w109_esc.zarr', (450, 1650, 300, 1900)),
}
out = {}
for k, (z, roi) in R.items():
    m = check.allowed_mask(z, roi=roi, step=4)
    if roi is None: roi = (0, m.shape[0], 0, m.shape[1])
    y0, y1, x0, x1 = roi
    YY, XX = np.mgrid[y0:y1:16, x0:x1:16]
    d = check.distance_mm(z, YY, XX)
    out[k] = dict(roi=roi, allowed_frac=round(float(m.mean()), 4), dist_mm_min=round(float(np.nanmin(d)), 3),
                  dist_mm_med=round(float(np.nanmedian(d)), 3), mesh_frac=round(float(np.isfinite(d).mean()), 3))
    print(k, out[k], flush=True)
json.dump(out, open('/home/liam/projects/vesuvius-work/.claude/worktrees/leadA-push3/x_push3/aclick/p306/exclusion/region_fracs.json', 'w'), indent=1)
