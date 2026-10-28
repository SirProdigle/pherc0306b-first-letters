"""make_registry.py : writes renders.json (render zarr -> tifxyz mesh) for the PHerc0306B renders lanes use, after checking
that each render's canvas equals the mesh grid / scale (vc_render_tifxyz --scale 1)."""
import json, os, sys
import numpy as np, tifffile, zarr
HERE = os.path.dirname(os.path.abspath(__file__))
W = '/home/liam/projects/vesuvius-work'
L = f'{W}/x_vast_plan/results/PHerc0306B_z8400/results/flat'
P3L = f'{W}/x_local_lanes/PHerc0306B_z9900/out/L0306bacw/p3/flat'
R = {
    f'{W}/x_focus306/renders/w050.zarr': (f'{L}/w050/tifxyz/flatten.tifxyz', 'pick 2, z8400 lane w050, 41 layers, surface 20'),
    f'{W}/x_focus306/renders/w052.zarr': (f'{L}/w052/tifxyz/flatten.tifxyz', 'z8400 lane w052, 41 layers, surface 20'),
    f'{W}/x_focus306/renders/w053.zarr': (f'{L}/w053/tifxyz/flatten.tifxyz', 'z8400 lane w053, 41 layers, surface 20'),
    f'{W}/x_focus306/renders/w074.zarr': (f'{L}/w074/tifxyz/flatten.tifxyz', 'LEAD A lane render (zone test only)'),
    f'{W}/x_req/20261020b/F6Y/w059.zarr': (f'{L}/w059/tifxyz/flatten.tifxyz', 'z8400 lane w059 (F6Y), 41 layers, surface 20'),
    f'{W}/x_focus306/renders/w074own.zarr': (f'{W}/fl/sqm/PHerc0306B_f306_w074own/auto_grown_20261012144243662', 'LEAD A v1 mesh render (zone test only)'),
    f'{W}/x_focus/renders/A_own_r41.zarr': (f'{W}/fl/sqm/PHerc0306B_a306_row/auto_grown_20261012141245268', 'pick 2 own patch, 41 layers, surface 20'),
    f'{W}/x_refine/trials/P1_surface_refine/renders/P1V5Rc.zarr': (f'{W}/x_req/20261020/A10/PHerc0306B_w074_claim_v4c/mesh/PHerc0306B_20250521133212_w074own_touchup_V5Rc', 'LEAD A V5Rc render (zone test only)'),
    f'{W}/x_local_lanes/PHerc0306B_z9900/out/L0306bacw/p3/renders/L0306bacw_w109.zarr': (f'{P3L}/w109/tifxyz/flatten.tifxyz', 'z9900 P3 lane w109, 31 layers, surface 15 (level 0 removed; crop p3w109_esc.zarr exists)'),
}
extra = sys.argv[1:]   # optional: more ZARR=MESH pairs
for e in extra:
    z, m = e.split('='); R[z] = (m, 'added by CLI')
reg = {}
for z, (m, note) in R.items():
    x = tifffile.imread(f'{m}/x.tif'); s = json.load(open(f'{m}/meta.json'))['scale'][0]
    exp = (round(x.shape[0] / s), round(x.shape[1] / s))
    try:
        sh = zarr.open(z, mode='r')['0'].shape[-2:]; okc = tuple(sh) == exp
    except Exception:
        sh, okc = None, None
    reg[os.path.realpath(z)] = dict(mesh=os.path.realpath(m), offset_yx=[0, 0], note=note, canvas=list(sh) if sh else None,
                                     grid_over_scale=list(exp), canvas_matches=okc)
    print(z, sh, exp, okc)
json.dump(reg, open(f'{HERE}/renders.json', 'w'), indent=1)
