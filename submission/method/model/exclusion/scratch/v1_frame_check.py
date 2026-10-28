import sys, json, numpy as np
sys.path.insert(0, '/home/liam/projects/vesuvius-work/.claude/worktrees/leadA-push3/x_push3/aclick/p306/exclusion')
import check, build_zone as B
from scipy.spatial import cKDTree
K = 115.741
P, valid, s = B.load(B.V5RC); pts, py, px = B.densify(P, valid, s)
u = -36 + px / K; v = -21 + py / K
box = (u >= -15) & (u <= 23) & (v >= -6.84) & (v <= 2.72)
T = cKDTree(pts[box])
j = json.load(open(B.V1JSON)); L = j['levelling']; c = np.array(L['centre_render_px']); ld = np.array(L['line_dir']); dd = np.array(L['down_dir'])
a = j['area_mm_render_levelled']
uu, vv = np.meshgrid(np.linspace(a['u0'], a['u1'], 9), np.linspace(a['v0'], a['v1'], 5))
X = c[0] + (uu * ld[0] + vv * dd[0]) * K; Y = c[1] + (uu * ld[1] + vv * dd[1]) * K
xyz = check.px_to_xyz('/home/liam/projects/vesuvius-work/x_focus306/renders/w074own.zarr', Y.ravel(), X.ravel())
d, _ = T.query(xyz[np.isfinite(xyz).all(1)])
print('v1 4cm2 area grid points -> distance to v4c box surface (mm): median', np.median(d) * 0.00864, 'max', d.max() * 0.00864)
# swapped-axes alternative (if centre were (y, x))
X2 = c[1] + (uu * ld[0] + vv * dd[0]) * K; Y2 = c[0] + (uu * ld[1] + vv * dd[1]) * K
xyz2 = check.px_to_xyz('/home/liam/projects/vesuvius-work/x_focus306/renders/w074own.zarr', Y2.ravel(), X2.ravel())
d2, _ = T.query(xyz2[np.isfinite(xyz2).all(1)])
print('swapped hypothesis: median', np.median(d2) * 0.00864)
ctx = (u >= -30) & (u <= 30) & (v >= -20) & (v <= 15)
T2 = cKDTree(pts[ctx])
for cs in (0, 1):
    for ds in (0, 1):
        cc = c[::-1] if cs else c; l = ld[::-1] if ds else ld; dv = dd[::-1] if ds else dd
        X = cc[0] + (uu * l[0] + vv * dv[0]) * K; Y = cc[1] + (uu * l[1] + vv * dv[1]) * K
        q = check.px_to_xyz('/home/liam/projects/vesuvius-work/x_focus306/renders/w074own.zarr', Y.ravel(), X.ravel())
        q = q[np.isfinite(q).all(1)]
        d1, _ = T.query(q); d2, _ = T2.query(q)
        print('centre swapped', cs, 'dirs swapped', ds, 'n', len(q), 'to box med %.3f' % (np.median(d1) * 0.00864),
              'to ctx med %.3f max %.3f' % (np.median(d2) * 0.00864, d2.max() * 0.00864))
