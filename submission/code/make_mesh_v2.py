"""make_mesh_v2.py : the mesh folder of the package (written for version 2; versions 4, 4b and 4c change only the claim box).

  mesh/<MESH>/                 the claim surface V5Rc as tifxyz: x/y/z.tif copied byte-for-byte from lane P1's
                               meshes05/V5Rf_ctx (= code/surface touchup.py + reflatten.py, re-checked byte-identical by I2),
                               meta.json renamed to <MESH> with repo-relative provenance
  mesh/<MESH>_region/          the same tifxyz cut to the claim box + 2 mm (grid_offset recorded, renders line up)
  mesh/distortion_in_box.json  flattening distortion inside the claim box (edges vs the nominal 20 vx, cell area, angle,
                               stretch / shear as in P1 metric.py)
  mesh/box_area.json           the claim box on the flat grid and on the 3-D mesh (cells counted with their fractional overlap)
  mesh/<MESH>/distortion.json  the same statistics over every valid cell of the mesh

The grid is P1's claim-frame carrier grid: node (row i, col j) <-> claim frame (u, v) = (-36 + j * 0.1728, -21 + i * 0.1728) mm,
20 voxels per cell, so a scale-1 render of the mesh is already levelled: render px (x, y) <-> (u, v) = (-36 + x / K, -21 + y / K),
K = 1000 / 8.64 px per mm.  v points DOWN as rendered; the claim image is the render flipped top-to-bottom (reading orientation).

Usage: make_mesh_v2.py SRC_MESH_DIR PACKAGE_DIR [MESH_NAME]
"""
import os, sys, json, shutil, hashlib
import numpy as np, tifffile

UM = 8.64; K = 1000.0 / UM; G = 20; CELL = G / K; U0, V0 = -36.0, -21.0
BOX = dict(u0=-15.0, u1=23.0, v0=-6.84, v1=2.72)                      # claim box, claim frame mm (version 4c; v4b: u -15..+25 x v -7.35..+2.525; v4: v -5..+4.875; v2 / v3: u -12.1..+23.5 x v -8.55..+2.5)
MESH = 'PHerc0306B_20250521133212_w074own_touchup_V5Rc'


def sha(p):
    h = hashlib.sha256(); f = open(p, 'rb')
    for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def load(d):
    P = np.stack([tifffile.imread(f'{d}/{c}.tif') for c in 'xyz'], -1).astype(np.float64)
    ok = np.all(np.isfinite(P), -1) & np.all(P > 0, -1)
    return P, ok


def cell_stats(P, ok, sel=None):
    """per grid cell: edge lengths (vx), stretch, shear, area ratio (3-D area / nominal 400 vx2), grid angle"""
    a_, b_, c_ = P[:-1, :-1], P[:-1, 1:], P[1:, :-1]
    m = ok[:-1, :-1] & ok[:-1, 1:] & ok[1:, :-1]
    if sel is not None: m &= sel
    du, dv = (b_ - a_)[m], (c_ - a_)[m]
    eu, ev = np.linalg.norm(du, axis=-1), np.linalg.norm(dv, axis=-1)
    ar = np.linalg.norm(np.cross(du, dv), axis=-1) / G ** 2
    cosang = (du * dv).sum(-1) / np.maximum(eu * ev, 1e-9)
    an = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    su, sv = eu / G, ev / G
    pc = lambda x, nd=3: [round(float(np.percentile(x, q)), nd) for q in (5, 50, 95)]          # noqa: E731
    return dict(n_cells=int(m.sum()), edges_u_vx_p5_p50_p95=pc(eu, 2), edges_v_vx_p5_p50_p95=pc(ev, 2),
                stretch_u_p5_p50_p95=pc(su), stretch_v_p5_p50_p95=pc(sv), area_ratio_p5_p50_p95=pc(ar),
                grid_angle_deg_p5_p50_p95=pc(an, 2),
                share_cells_stretch_off_more_than_5pct=round(float(((np.abs(su - 1) > .05) | (np.abs(sv - 1) > .05)).mean()), 4),
                share_cells_area_off_more_than_10pct=round(float((np.abs(ar - 1) > .1).mean()), 4))


def box_cells(shape, box):
    """fractional overlap of each grid cell (rows i..i+1, cols j..j+1) with the box"""
    H, Wd = shape; j = np.arange(Wd - 1); i = np.arange(H - 1)
    u0c, u1c = U0 + j * CELL, U0 + (j + 1) * CELL; v0c, v1c = V0 + i * CELL, V0 + (i + 1) * CELL
    fu = np.clip(np.minimum(u1c, box['u1']) - np.maximum(u0c, box['u0']), 0, None) / CELL
    fv = np.clip(np.minimum(v1c, box['v1']) - np.maximum(v0c, box['v0']), 0, None) / CELL
    return fv[:, None] * fu[None, :]


def area3d(P, ok, box):
    P00, P01, P10, P11 = P[:-1, :-1], P[:-1, 1:], P[1:, :-1], P[1:, 1:]
    a = 0.5 * np.linalg.norm(np.cross(P01 - P00, P10 - P00), axis=-1) + 0.5 * np.linalg.norm(np.cross(P01 - P11, P10 - P11), axis=-1)
    cok = ok[:-1, :-1] & ok[:-1, 1:] & ok[1:, :-1] & ok[1:, 1:]
    f = box_cells(P.shape[:2], box)
    return float((np.where(cok, a, 0) * f).sum()) * (UM / 1000) ** 2, float((f * cok).sum() / max(f.sum(), 1e-9))


def main():
    src, pkg = sys.argv[1], sys.argv[2]; name = sys.argv[3] if len(sys.argv) > 3 else MESH
    out = f'{pkg}/mesh/{name}'; os.makedirs(out, exist_ok=True)
    for c in 'xyz': shutil.copy2(f'{src}/{c}.tif', f'{out}/{c}.tif')
    meta = json.load(open(f'{src}/meta.json'))
    meta['uuid'] = meta['name'] = name
    rf = meta.get('p1_reflatten', {}); rf['source'] = 'V5R_rightend = code/surface/touchup.py output (original A + right-end touch-up)'
    meta['p1_reflatten'] = rf
    meta['source'] = 'code/surface: touchup.py (CT-driven right-end touch-up of lead A own-sheet GrowPatch) + reflatten.py (fold-weighted ARAP)'
    meta['derived_from'] = dict(growpatch='PHerc0306B_20250521133212_w074own_auto_grown_20261012144243662 (v1 mesh; seed 3559, 3224, 9348)',
                                frame='claim-frame carrier grid: node (i, j) <-> (u, v) = (-36 + 0.1728 j, -21 + 0.1728 i) mm; 20 vx per cell; '
                                      'scale-1 render px (x, y) <-> (u, v) = (-36 + x / 115.741, -21 + y / 115.741) mm; v down as rendered')
    json.dump(meta, open(f'{out}/meta.json', 'w'), indent=1)
    P, ok = load(out)
    H, Wd = ok.shape
    dist = cell_stats(P, ok); dist['scope'] = 'every valid cell of the mesh'
    json.dump(dist, open(f'{out}/distortion.json', 'w'), indent=1)
    # claim box: cells whose centre lies inside it
    jj, ii = np.meshgrid(np.arange(Wd - 1) + 0.5, np.arange(H - 1) + 0.5)
    Uc, Vc = U0 + jj * CELL, V0 + ii * CELL
    inbox = (Uc >= BOX['u0']) & (Uc <= BOX['u1']) & (Vc >= BOX['v0']) & (Vc <= BOX['v1'])
    db = cell_stats(P, ok, inbox); db['scope'] = 'cells whose centre lies inside the claim box'; db['box_mm'] = BOX
    db['invalid_cells_in_box'] = int((inbox & ~(ok[:-1, :-1] & ok[:-1, 1:] & ok[1:, :-1])).sum())
    json.dump(db, open(f'{pkg}/mesh/distortion_in_box.json', 'w'), indent=1)
    a3, cov = area3d(P, ok, BOX)
    flat = (BOX['u1'] - BOX['u0']) * (BOX['v1'] - BOX['v0'])
    ba = dict(box_mm=BOX, box_size_mm=[round(BOX['u1'] - BOX['u0'], 2), round(BOX['v1'] - BOX['v0'], 2)],
              flat_cm2=round(flat / 100, 4), mesh3d_cm2=round(a3 / 100, 4), flat_covered_by_valid_cells=round(cov, 4),
              method='3-D area of the mesh triangles (2 per grid cell), each cell weighted by its fractional overlap with the box in the flat (u, v) grid')
    # corners: render px of the scale-1 render, grid coords, and 3-D voxel position
    corners = {}
    for nm, (u, v) in dict(top_left=(BOX['u0'], BOX['v0']), top_right=(BOX['u1'], BOX['v0']), bottom_right=(BOX['u1'], BOX['v1']),
                           bottom_left=(BOX['u0'], BOX['v1'])).items():
        gi, gj = (v - V0) / CELL, (u - U0) / CELL; i0, j0 = int(gi), int(gj); fi, fj = gi - i0, gj - j0
        q = P[i0, j0] * (1 - fi) * (1 - fj) + P[i0, j0 + 1] * (1 - fi) * fj + P[i0 + 1, j0] * fi * (1 - fj) + P[i0 + 1, j0 + 1] * fi * fj
        corners[nm] = dict(claim_frame_mm_as_rendered=[u, v], render_px_xy=[round((u - U0) * K, 1), round((v - V0) * K, 1)],
                           tifxyz_grid_row_col=[round(gi, 2), round(gj, 2)], volume_xyz_vx=[round(float(x), 1) for x in q])
    ba['corners'] = corners
    json.dump(ba, open(f'{pkg}/mesh/box_area.json', 'w'), indent=1)
    # region cut: the box + 2 mm
    m = int(np.ceil(2.0 / CELL))
    gy0 = max(0, int((BOX['v0'] - V0) / CELL) - m); gy1 = min(H, int(np.ceil((BOX['v1'] - V0) / CELL)) + m + 1)
    gx0 = max(0, int((BOX['u0'] - U0) / CELL) - m); gx1 = min(Wd, int(np.ceil((BOX['u1'] - U0) / CELL)) + m + 1)
    reg = f'{pkg}/mesh/{name}_region'; os.makedirs(reg, exist_ok=True)
    sub = {}
    for c in 'xyz':
        sub[c] = tifffile.imread(f'{out}/{c}.tif')[gy0:gy1, gx0:gx1]; tifffile.imwrite(f'{reg}/{c}.tif', sub[c])
    okr = (sub['z'] > 0) & (sub['x'] > 0)
    mr = dict(meta); mr['uuid'] = mr['name'] = f'{name}_region'; mr['grid_offset'] = [gy0, gx0]
    mr['bbox'] = [[float(sub[c][okr].min()) for c in 'xyz'], [float(sub[c][okr].max()) for c in 'xyz']]
    mr['note'] = (f'grid rows {gy0}:{gy1}, cols {gx0}:{gx1} of {name} (the claim box + 2 mm); a scale-1 render of this cut lands at '
                  f'render px ({20 * gx0}, {20 * gy0}) of the whole mesh')
    json.dump(mr, open(f'{reg}/meta.json', 'w'), indent=1)
    dr = cell_stats(*load(reg)); dr['scope'] = 'every valid cell of the region cut'
    json.dump(dr, open(f'{reg}/distortion.json', 'w'), indent=1)
    print(json.dumps(dict(mesh=out, grid=[H, Wd], region=[gy0, gy1, gx0, gx1], box=ba['flat_cm2'], box3d=ba['mesh3d_cm2'],
                          in_box=db, sha={c: sha(f'{out}/{c}.tif') for c in 'xyz'}), indent=1))


if __name__ == '__main__':
    main()
