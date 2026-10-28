"""own_compare.py LANE WINDING SIDE --poly P.json --patch PATCH_TIFXYZ_DIR --maps OWN_MQ_DIR --um UM --lane-map LANE_FUSED.npy --out DIR

Does a lane line reproduce on its own-sheet GrowPatch?  The general form of x_followup/C_depth/compare.py (13 Oct lesson: the
patch can be MIRRORED relative to the lane and the line can run at any angle on it, so never judge with line_find defaults).
  1. Lane points every 1 mm along the line centre and 3 mm either side -> 3-D (lane tifxyz) -> patch (u,v) by sub-cell least
     squares in the nearest cell; keep points inside a cell with |normal offset| < 15 vx and in-plane residual < 3 vx.
  2. Affine fit lane px -> patch px (RANSAC); report rmse, scale, det (negative = mirrored), normal offsets along the line.
  3. Resample each patch face's fused map into the lane frame around the line; Pearson r and hot (z > 2) overlap against the
     lane's own fused map; a stacked image (lane | patch fwd | patch rev) and a wide view of the better face with its CT.
Writes OUT/{own_compare.json, aligned_faces.jpg, wide_<face>_with_CT.jpg}.
"""
import os, json, argparse
import numpy as np, cv2, tifffile
from scipy.spatial import cKDTree
cv2.setNumThreads(4)
W = '/home/liam/projects/vesuvius-work'; G = 20
ap = argparse.ArgumentParser(); ap.add_argument('lane'); ap.add_argument('winding'); ap.add_argument('side')
ap.add_argument('--poly', required=True); ap.add_argument('--patch', required=True); ap.add_argument('--maps', required=True)
ap.add_argument('--um', type=float, required=True); ap.add_argument('--lane-map', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--pad-mm', type=float, default=6.0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); UM = a.um
LANE = f'{W}/x_vast_plan/results/{a.lane}/results/flat/{a.winding}/tifxyz/flatten.tifxyz'
pj = json.load(open(a.poly)); P = np.array(pj.get('poly_render', pj.get('poly', pj)), float)


def page(b):
    x = np.clip((b - 1.0) / 2.0, 0, 1); im = (255 - x * 230).astype(np.uint8); im[b == 0] = 70
    return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)


def bil(A, gy, gx):
    y0, x0 = int(gy), int(gx); fy, fx = gy - y0, gx - x0
    q = A[y0:y0 + 2, x0:x0 + 2].astype(np.float64)
    return np.nan if q.shape != (2, 2) else q[0, 0] * (1 - fy) * (1 - fx) + q[0, 1] * (1 - fy) * fx + q[1, 0] * fy * (1 - fx) + q[1, 1] * fy * fx


def mapping():
    L = [tifffile.imread(f'{LANE}/{c}.tif').astype(np.float64) for c in 'xyz']
    QP = np.stack([tifffile.imread(f'{a.patch}/{c}.tif').astype(np.float64) for c in 'xyz'], -1); ok = QP[..., 2] > 0
    idx = np.argwhere(ok); tree = cKDTree(QP[ok])
    s, e = (P[0] + P[3]) / 2, (P[1] + P[2]) / 2; dv = (e - s) / np.linalg.norm(e - s); nv = np.array([-dv[1], dv[0]])
    n = int(np.linalg.norm(e - s) * UM / 1000) + 1; out = []
    for t in np.linspace(0, 1, n):
        for off in (-3.0, 0.0, 3.0):
            x, y = s + t * (e - s) + nv * off * 1000 / UM
            p = np.array([bil(A, y / G, x / G) for A in L])
            if not np.isfinite(p).all() or p[2] <= 0: continue
            d, i = tree.query(p); r, c = idx[i]; du = dv_ = -1
            for _ in range(3):
                r = int(np.clip(r, 0, QP.shape[0] - 2)); c = int(np.clip(c, 0, QP.shape[1] - 2))
                if not (ok[r, c] and ok[r, c + 1] and ok[r + 1, c]): break
                tu = QP[r, c + 1] - QP[r, c]; tv = QP[r + 1, c] - QP[r, c]
                (du, dv_), *_ = np.linalg.lstsq(np.stack([tu, tv], 1), p - QP[r, c], rcond=None)
                if 0 <= du < 1 and 0 <= dv_ < 1: break
                r, c = r + int(np.floor(dv_)), c + int(np.floor(du))
            if not (0 <= du <= 1.05 and 0 <= dv_ <= 1.05): continue
            nn = np.cross(tu, tv); nn /= np.linalg.norm(nn) + 1e-9; res = p - QP[r, c] - du * tu - dv_ * tv
            out.append(dict(t=float(t), off_mm=off, lane_px=[float(x), float(y)], patch_px=[float((c + du) * G), float((r + dv_) * G)],
                            normal_offset_vx=float(res @ nn), inplane_resid_vx=float(np.linalg.norm(res - (res @ nn) * nn))))
    return out


def to_lane(z, A, X0, X1, Y0, Y1):
    xs, ys = np.meshgrid(np.arange(X0, X1, dtype=np.float32), np.arange(Y0, Y1, dtype=np.float32))
    px = A[0, 0] * xs + A[0, 1] * ys + A[0, 2]; py = A[1, 0] * xs + A[1, 1] * ys + A[1, 2]
    return cv2.remap(z.astype(np.float32), px.astype(np.float32), py.astype(np.float32), cv2.INTER_LINEAR, borderValue=0)


if __name__ == '__main__':
    mp = mapping(); good = [m for m in mp if abs(m['normal_offset_vx']) < 15 and m['inplane_resid_vx'] < 3]
    if len(good) < 6: raise SystemExit(f'only {len(good)} mapped points: the patch does not cover the line')
    src = np.array([m['lane_px'] for m in good], np.float32); dst = np.array([m['patch_px'] for m in good], np.float32)
    A, inl = cv2.estimateAffine2D(src, dst, ransacReprojThreshold=6.0)
    inl = inl.ravel().astype(bool); pred = src @ A[:, :2].T + A[:, 2]; err = np.sqrt(np.sum((pred - dst) ** 2, 1))
    rmse = float(np.sqrt(np.mean(err[inl] ** 2))); det = float(np.linalg.det(A[:, :2]))           # rmse over RANSAC inliers
    good = [m for m, k in zip(good, inl) if k]
    line_off = [round(m['normal_offset_vx'], 2) for m in good if m['off_mm'] == 0]
    pad = int(a.pad_mm * 1000 / UM)
    X0, X1 = int(P[:, 0].min()) - pad, int(P[:, 0].max()) + pad; Y0, Y1 = int(P[:, 1].min()) - pad, int(P[:, 1].max()) + pad
    lane = np.load(a.lane_map, mmap_mode='r'); ref = np.zeros((Y1 - Y0, X1 - X0), np.float32)
    ys0, xs0 = max(0, Y0), max(0, X0); ref[ys0 - Y0:min(lane.shape[0], Y1) - Y0, xs0 - X0:min(lane.shape[1], X1) - X0] = \
        np.asarray(lane[ys0:min(lane.shape[0], Y1), xs0:min(lane.shape[1], X1)]).astype(np.float32)
    res = dict(lane=a.lane, winding=a.winding, side=a.side, n_points=len(good), n_outliers=int((~inl).sum()), affine=A.tolist(), rmse_px=round(rmse, 2),
               scale=round(float(np.sqrt(abs(det))), 4), mirrored=det < 0, line_normal_offset_vx=line_off, faces={})
    rows = [('LANE ' + a.winding + ' ' + a.side + ': the line', ref)]; best = None
    for face in ('fwd', 'rev'):
        f = f'{a.maps}/{face}_fused.npy'
        if not os.path.exists(f): continue
        z = to_lane(np.load(f), A, X0, X1, Y0, Y1); v = (z != 0) & (ref != 0)
        r = float(np.corrcoef(z[v], ref[v])[0, 1]) if v.sum() > 1000 else float('nan')
        hz, hr = (z > 2) & v, (ref > 2) & v; jac = float((hz & hr).sum() / max(1, (hz | hr).sum()))
        res['faces'][face] = dict(pearson_r=round(r, 3), hot_jaccard=round(jac, 3)); rows.append((f'own patch {face}: r {r:.2f}, hot overlap {jac:.2f}', z))
        if best is None or r > best[1]: best = (face, r)
    out = []
    for lab, b in rows:
        im = page(b); im = np.pad(im, ((40, 8), (0, 0), (0, 0)), constant_values=255); cv2.putText(im, lab, (10, 30), 0, 0.9, (30, 30, 200), 2); out.append(im)
    img = np.concatenate(out, 0); cv2.rectangle(img, (10, img.shape[0] - 6), (10 + int(10000 / UM), img.shape[0] - 2), (30, 30, 200), -1)
    cv2.imwrite(f'{a.out}/aligned_faces.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    # wide view of the better face (+-25 mm around the line centre) beside the patch CT, in the lane frame, 10 um/px
    cx, cy = P.mean(0); R = int(25 * 1000 / UM)
    zw = to_lane(np.load(f'{a.maps}/{best[0]}_fused.npy'), A, int(cx) - R, int(cx) + R, int(cy) - R, int(cy) + R)
    ct = cv2.imread(f'{a.maps}/ct20.png', 0).astype(np.float32); cw = to_lane(ct, A, int(cx) - R, int(cx) + R, int(cy) - R, int(cy) + R)
    vv = cw > 0; lo, hi = np.percentile(cw[vv], [1, 99.5]) if vv.any() else (0, 255); g = np.clip((cw - lo) / (hi - lo + 1e-6), 0, 1); g[~vv] = 0
    g = cv2.cvtColor(cv2.createCLAHE(2.0, (8, 8)).apply((g * 255).astype(np.uint8)), cv2.COLOR_GRAY2BGR)
    both = np.concatenate([page(zw), g], 1); s = UM / 10.0; both = cv2.resize(both, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    yl = int(R * s)
    cv2.line(both, (0, yl), (40, yl), (30, 30, 200), 4); cv2.rectangle(both, (10, both.shape[0] - 26), (1010, both.shape[0] - 14), (30, 30, 200), -1)
    cv2.imwrite(f'{a.out}/wide_{best[0]}_with_CT.jpg', both, [cv2.IMWRITE_JPEG_QUALITY, 85])
    json.dump(res, open(f'{a.out}/own_compare.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ('affine', 'line_normal_offset_vx')}), 'line offsets median', float(np.median(line_off)) if line_off else None)
