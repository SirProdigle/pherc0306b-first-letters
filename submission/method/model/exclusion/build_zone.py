"""build_zone.py : the A-click exclusion zone (lane P306).  Writes zone.npz (3-D points of every surface shown in any lead-A
claim image, PHerc0306B volume 20250521133212 voxels, x/y/z) and zone.json (provenance).

Zone surface = union of
  (1) mesh V5Rc (claim versions 2, 3, 4, 4b, 4c and the v5 builds): every mesh point whose render pixel lies in the context /
      main-image window u -30..+30 x v -20..+15 mm (claim frame u = -36 + x/K, v = -21 + y/K, K = 115.741 px/mm; claimlib_v2.CTX).
      All supplementary views of v2-v5 (threshold rungs, single models, big box, blank control R, mirror box+3 mm, letter
      table, side-by-side, form overlays) are crops of this window.  Margin: +2 mm each side on top (u -32..+32, v -22..+17).
  (2) mesh w074own auto_grown_20261012144243662 (claim version 1, 14 Oct): levelled frame of the v1 image json
      (centre, line_dir, down_dir in render px); v1 view u -24..16 x v -11..8 mm and its 60 x 36 mm context jpg; we take the
      superset u -34..+30 x v -20..+18 mm plus the same 2 mm margin.
The 2.0 mm 3-D buffer is applied by check.py (distance of a candidate pixel's 3-D point to the nearest zone point).
Meshes are densified 4x (bilinear inside quads whose 4 corners are valid) so the point spacing is <= 5 render px (0.043 mm).
"""
import json, hashlib, os
import numpy as np, tifffile

W = '/home/liam/projects/vesuvius-work'
HERE = os.path.dirname(os.path.abspath(__file__))
K = 115.741
V5RC = f'{W}/x_req/20261020/A10/PHerc0306B_w074_claim_v4c/mesh/PHerc0306B_20250521133212_w074own_touchup_V5Rc'
V1 = f'{W}/CANDIDATES/SUBMISSION_FINAL_PHerc0306B/mesh/PHerc0306B_20250521133212_w074own_auto_grown_20261012144243662'
V1JSON = f'{W}/CANDIDATES/SUBMISSION_FINAL_PHerc0306B/PHerc0306B_20250521133212_w074own_auto_grown_20261012144243662.json'
MARGIN = 2.0
UP = 4


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def load(mesh):
    P = np.stack([tifffile.imread(f'{mesh}/{a}.tif').astype(np.float64) for a in 'xyz'], -1)
    s = json.load(open(f'{mesh}/meta.json'))['scale']
    valid = np.all(P > 0, -1)
    return P, valid, float(s[0])


def densify(P, valid, s, up=UP):
    """returns (pts (N,3), py, px render pixel coords) for all valid vertices and the up x up sub-samples of valid quads."""
    H, Wd, _ = P.shape
    out, oy, ox = [], [], []
    q = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
    qi, qj = np.nonzero(q)
    for a in range(up):
        for b in range(up):
            fa, fb = a / up, b / up
            p = ((1 - fa) * (1 - fb) * P[qi, qj] + fa * (1 - fb) * P[qi + 1, qj] + (1 - fa) * fb * P[qi, qj + 1]
                 + fa * fb * P[qi + 1, qj + 1])
            out.append(p); oy.append((qi + fa) / s); ox.append((qj + fb) / s)
    vi, vj = np.nonzero(valid)
    out.append(P[vi, vj]); oy.append(vi / s); ox.append(vj / s)
    return np.concatenate(out), np.concatenate(oy), np.concatenate(ox)


def main():
    parts, meta = [], dict(created='2026-10-21', lane='P306', units='PHerc0306B 20250521133212 voxels (8.64 um), columns x,y,z',
                          buffer_mm_applied_by_check=2.0, densify=UP, margin_mm=MARGIN, sources=[])
    # (1) V5Rc
    P, valid, s = load(V5RC)
    pts, py, px = densify(P, valid, s)
    u = -36 + px / K; v = -21 + py / K
    m = (u >= -30 - MARGIN) & (u <= 30 + MARGIN) & (v >= -20 - MARGIN) & (v <= 15 + MARGIN)
    parts.append(pts[m])
    meta['sources'].append(dict(mesh=V5RC, x_tif_sha256=sha(f'{V5RC}/x.tif'), versions='v2, v3, v4, v4b, v4c, v5 (C1/C3/H)',
                                window_mm=dict(u0=-30, u1=30, v0=-20, v1=15), points=int(m.sum()),
                                note='claim frame u=-36+x/K, v=-21+y/K; render px = grid/scale'))
    # (2) v1 mesh in its levelled frame
    j = json.load(open(V1JSON))['levelling']
    c = np.array(j['centre_render_px']); ld = np.array(j['line_dir']); dd = np.array(j['down_dir'])
    P, valid, s = load(V1)
    pts, py, px = densify(P, valid, s)
    dx, dy = px - c[0], py - c[1]
    u = (dx * ld[0] + dy * ld[1]) / K; v = (dx * dd[0] + dy * dd[1]) / K
    m = (u >= -34 - MARGIN) & (u <= 30 + MARGIN) & (v >= -20 - MARGIN) & (v <= 18 + MARGIN)
    parts.append(pts[m])
    meta['sources'].append(dict(mesh=V1, x_tif_sha256=sha(f'{V1}/x.tif'), versions='v1 (14 Oct)',
                                window_mm=dict(u0=-34, u1=30, v0=-20, v1=18), points=int(m.sum()),
                                levelling=j, note='superset of the v1 view (u -24..16, v -11..8) and a 60 x 36 mm context'))
    Z = np.concatenate(parts).astype(np.float32)
    meta['n_points'] = int(len(Z)); meta['bbox_xyz'] = [Z.min(0).tolist(), Z.max(0).tolist()]
    np.savez_compressed(f'{HERE}/zone.npz', pts=Z)
    meta['zone_npz_sha256'] = sha(f'{HERE}/zone.npz')
    json.dump(meta, open(f'{HERE}/zone.json', 'w'), indent=1)
    print(json.dumps({k: meta[k] for k in ('n_points', 'bbox_xyz')}), [d['points'] for d in meta['sources']])


if __name__ == '__main__':
    main()
