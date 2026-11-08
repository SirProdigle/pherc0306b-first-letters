"""make_submission_A.py : DRAFT First Letters submission package for lead A (PHerc0306B, own-sheet GrowPatch around the w074 line).
Builds everything the prize page asks for that does not depend on the reading:

  <MESH>.jpg / <MESH>_2x.jpg        the claim image: the 4 cm2 rectangle (28.5 x 14 mm) on the unwrapped surface, generated
                                    programmatically from the CT volume and the submitted mesh (no manual marks), scroll named,
                                    1 cm bar, ink = mean of three per-model z-scores (dark = ink)
  <MESH>_rows.jpg                   the same rectangle with row boxes drawn in the margins only (ink above, bare CT below)
  context_with_rectangle.jpg        60 x 36 mm around it with the rectangle outlined
  region_each_model_and_CT.jpg      Hecate / Reader v2 / TRACE / fused / bare CT over the rectangle
  region.json                       rectangle corners in levelled mm, render px, tifxyz grid and 3-D volume voxels; row centres
  mesh/<MESH>/                      the full own-sheet tifxyz (x/y/z.tif, meta.json): GrowPatch output, quasi-isometric
  mesh/<MESH>_region/               the same tifxyz cut to the rectangle's bounding box + 2 mm
  mesh/distortion_in_rectangle.json flattening distortion inside the rectangle (edge lengths, cell areas, grid angles)
  controls/                         hallucination control panel on this map, depth sweep, independent-surface reproduction,
                                    neighbour windings
Letter dimensions (px and mm for a few letters) need the reading and are left as a template in README.md.
Re-run: envs/vcpy/bin/python x_submit/make_submission_A.py
"""
import os, json, shutil, subprocess
import numpy as np, cv2, tifffile, zarr
cv2.setNumThreads(4)
W = os.environ.get('FL_ROOT', '/home/liam/projects/vesuvius-work')          # FL_ROOT: repo root override (Docker: /opt/fl); unset = this machine
OUT = f'{W}/CANDIDATES/SUBMISSION_DRAFT_A_owner_only'
SCROLL, VOL = 'PHerc0306B', '20250521133212'
UM = 8.64
PATCH = f'{W}/fl/sqm/PHerc0306B_f306_w074own/auto_grown_20261012144243662'
MESH = f'{SCROLL}_{VOL}_w074own_auto_grown_20261012144243662'
DR = f'{W}/x_deepread/results/b1/out/patchW074'                      # max-quality maps of the whole patch render (layer 20)
RENDER = f'{W}/x_focus306/renders/w074own.zarr'                      # 41 layers, surface = layer 20, 20 render px per tifxyz cell
RECT = dict(u0=-19.9, u1=12.3, v0=-7.8, v1=4.6)                       # mm in the levelled frame (u along the line, v down); 14 Oct 11:50: moved left and flattened; was u -16.2..12.3, v -8.85..5.15 (RECT1 backup)
GRID = 20                                                             # render px per tifxyz grid cell (scale 0.05)


def zs(a, v):
    x = a[v]; return ((a - x.mean()) / (x.std() + 1e-6)) * v


def frame():
    P = np.array(json.load(open(f'{W}/x_focus306/loops/own_row.json'))['poly_render'], float)
    c = P.mean(0); d = (P[1] - P[0]) / np.linalg.norm(P[1] - P[0]); n = np.array([-d[1], d[0]])
    if n[1] < 0: n = -n                                               # v points down the page, as in the reader pack
    return c, d, n


def to_render(c, d, n, u, v):
    """levelled mm (u, v) -> render px (x, y)"""
    k = 1000.0 / UM
    return c[0] + (u * d[0] + v * n[0]) * k, c[1] + (u * d[1] + v * n[1]) * k


def sample(img, c, d, n, u0, u1, v0, v1, px_um=10.0, nn=False):
    """resample render-grid image onto the levelled frame at px_um per pixel"""
    uu = np.arange(u0, u1, px_um / 1000.0); vv = np.arange(v0, v1, px_um / 1000.0)
    U, V = np.meshgrid(uu, vv)
    X, Y = to_render(c, d, n, U, V)
    return cv2.remap(img, X.astype(np.float32), Y.astype(np.float32), cv2.INTER_NEAREST if nn else cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def page(z, v):
    a = np.clip((z - 1.0) / 2.0, 0, 1); a[~v] = 0; g = np.where(v, 1 - 0.92 * a, 0.12)
    return (np.stack([g * 0.80, g * 0.88, g * 0.93], -1) * 255).astype(np.uint8)


def ct_page(ct, v):
    x = ct[v]; lo, hi = np.percentile(x, [1, 99.5]); g = np.clip((ct - lo) / (hi - lo + 1e-6), 0, 1); g[~v] = 0
    g = cv2.createCLAHE(2.0, (8, 8)).apply((g * 255).astype(np.uint8)); return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def label(img, text, scale_px_per_mm=100, bar_mm=10, top=True):
    img = img.copy(); H = img.shape[0]; L = int(bar_mm * scale_px_per_mm); f = scale_px_per_mm / 100.0
    lines = text.split('\n'); pad = np.full((int((25 + 45 * len(lines)) * f), img.shape[1], 3), 255, np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(pad, t, (int(15 * f), int((48 + 45 * i) * f)), 0, 1.0 * f, (20, 20, 20), max(1, int(2 * f)))
    img = np.concatenate([pad, img], 0) if top else img
    foot = np.full((int(60 * f), img.shape[1], 3), 255, np.uint8)
    cv2.rectangle(foot, (int(15 * f), int(25 * f)), (int(15 * f) + L, int(37 * f)), (30, 30, 200), -1)
    cv2.putText(foot, f'{bar_mm // 10} cm' if bar_mm >= 10 else f'{bar_mm} mm', (int(25 * f) + L, int(40 * f)), 0, 0.9 * f, (30, 30, 200), max(1, int(2 * f)))
    return np.concatenate([img, foot], 0)


def bilinear(a, gy, gx):
    y0, x0 = int(np.floor(gy)), int(np.floor(gx)); fy, fx = gy - y0, gx - x0
    q = a[y0:y0 + 2, x0:x0 + 2].astype(np.float64)
    return float(q[0, 0] * (1 - fy) * (1 - fx) + q[0, 1] * (1 - fy) * fx + q[1, 0] * fy * (1 - fx) + q[1, 1] * fy * fx)


def main():
    os.makedirs(f'{OUT}/controls', exist_ok=True); os.makedirs(f'{OUT}/mesh', exist_ok=True)
    c, d, n = frame()
    per = {m: np.load(f'{DR}/fwd_{f}.npy').astype(np.float32) for m, f in (('Hecate', 'hec'), ('Reader v2', 'rv2'), ('TRACE', 'tr'))}
    ct = np.asarray(zarr.open(RENDER, mode='r')['0'][20]).astype(np.float32)
    valid = np.logical_and.reduce([np.isfinite(a) & (a != 0) for a in per.values()]) & (ct > 0)
    zz = {m: zs(np.nan_to_num(a), valid) for m, a in per.items()}
    fused = np.mean(list(zz.values()), 0) * valid

    # --- the claim image: the rectangle, levelled, 10 um/px
    r = RECT
    S = lambda a, nn=False: sample(a, c, d, n, r['u0'], r['u1'], r['v0'], r['v1'], nn=nn)
    zr = S(fused); vr = S(valid.astype(np.uint8), True) > 0
    area = (r['u1'] - r['u0']) * (r['v1'] - r['v0']) / 100.0
    title = f"{SCROLL} (scan {VOL}, {UM} um voxels)   mesh {MESH}\n{r['u1'] - r['u0']:.1f} x {r['v1'] - r['v0']:.1f} mm = {area:.2f} cm2, 1 px = 10 um, dark = ink"
    img = page(zr, vr)
    cv2.imwrite(f'{OUT}/{MESH}.jpg', label(img, title), [cv2.IMWRITE_JPEG_QUALITY, 95])
    big = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(f'{OUT}/{MESH}_2x.jpg', label(big, title.replace('1 px = 10 um', '1 px = 5 um (2x)'), 200), [cv2.IMWRITE_JPEG_QUALITY, 95])
    # fallback claim image from the two models whose training data is published and excludes PHerc0306B (Reader v2: 0139, 1667,
    # Paris4, 0814, 0009B, 0343P, 0500P2, 0172; TRACE v3B: 0139, 0814, 0009B, 0343P, 0500P2).  Hecate's card names only Paris4 and
    # unnamed "other scrolls" (weak 3-D supervision), so its overlap with 0306B cannot be ruled out from the card.
    f2 = np.mean([zz['Reader v2'], zz['TRACE']], 0) * valid
    img2 = page(S(f2), vr)
    cv2.imwrite(f'{OUT}/{MESH}_ReaderV2_TRACE_only.jpg', label(img2, title.replace('dark = ink', 'dark = ink (Reader v2 + TRACE only)')),
                [cv2.IMWRITE_JPEG_QUALITY, 95])

    # --- rows: hot-pixel profile across v, peaks = row centres; boxes drawn in the margins only (outside the text extent)
    hot = (zr > 2) & vr
    prof = cv2.GaussianBlur(hot.mean(1).astype(np.float32).reshape(-1, 1), (1, 61), 0).ravel()
    vv = np.arange(r['v0'], r['v1'], 0.01)[:len(prof)]
    rows = []
    for i in range(1, len(prof) - 1):
        if prof[i] >= prof[i - 1] and prof[i] > prof[i + 1] and prof[i] > 0.35 * prof.max():
            if not rows or vv[i] - rows[-1][0] > 3.0: rows.append((float(vv[i]), float(prof[i])))
            elif prof[i] > rows[-1][1]: rows[-1] = (float(vv[i]), float(prof[i]))
    ann = np.concatenate([img, ct_page(S(ct), vr)], 0)
    Hh = img.shape[0]
    for k, (vc, _) in enumerate(rows):
        y = int((vc - r['v0']) * 100)
        for yo in (0, Hh):                                            # ink half and CT half
            for x0, x1 in ((0, 60), (ann.shape[1] - 60, ann.shape[1])):
                cv2.rectangle(ann, (x0, yo + y - 130), (x1, yo + y + 130), (30, 30, 200), 3)
            cv2.putText(ann, f'row {k + 1}', (65, yo + y - 135), 0, 0.8, (30, 30, 200), 2)
    cv2.imwrite(f'{OUT}/{MESH}_rows.jpg', label(ann, title + '\nrows marked in the margins only; lower half = bare CT of the same area'), [cv2.IMWRITE_JPEG_QUALITY, 92])

    # --- context with the rectangle
    cu0, cu1, cv0, cv1 = -30, 30, -18, 18
    zc = sample(fused, c, d, n, cu0, cu1, cv0, cv1); vc_ = sample(valid.astype(np.uint8), c, d, n, cu0, cu1, cv0, cv1, nn=True) > 0
    ctx = page(zc, vc_)
    cv2.rectangle(ctx, (int((r['u0'] - cu0) * 100), int((r['v0'] - cv0) * 100)), (int((r['u1'] - cu0) * 100), int((r['v1'] - cv0) * 100)), (30, 30, 200), 4)
    cv2.imwrite(f'{OUT}/context_with_rectangle.jpg', label(ctx, f'{SCROLL} {MESH}: 60 x 36 mm context, the 4 cm2 rectangle in red'), [cv2.IMWRITE_JPEG_QUALITY, 90])

    # --- each model and bare CT
    panels = [label(page(S(z), vr), f'{m} alone (z > 1 shown)') for m, z in zz.items()]
    panels += [label(img, 'fused (mean of the three z-scores)'), label(ct_page(S(ct), vr), 'bare CT, surface layer (no model)')]
    cv2.imwrite(f'{OUT}/region_each_model_and_CT.jpg', np.concatenate(panels, 0), [cv2.IMWRITE_JPEG_QUALITY, 90])

    # --- geometry: corners in render px, tifxyz grid, 3-D
    X, Y, Zt = (tifffile.imread(f'{PATCH}/{a}.tif') for a in 'xyz')
    corners = {}
    for name, (u, v) in dict(top_left=(r['u0'], r['v0']), top_right=(r['u1'], r['v0']), bottom_right=(r['u1'], r['v1']),
                             bottom_left=(r['u0'], r['v1']), centre=((r['u0'] + r['u1']) / 2, (r['v0'] + r['v1']) / 2)).items():
        px, py = to_render(c, d, n, u, v); gx, gy = px / GRID, py / GRID
        corners[name] = dict(levelled_mm=[u, v], render_px=[round(px, 1), round(py, 1)], tifxyz_grid=[round(gy, 2), round(gx, 2)],
                             volume_xyz_vx=[round(bilinear(A, gy, gx), 1) for A in (X, Y, Zt)])
    gxs = [cn['tifxyz_grid'][1] for cn in corners.values()]; gys = [cn['tifxyz_grid'][0] for cn in corners.values()]
    m = int(np.ceil(2000 / UM / GRID))                                # 2 mm margin in grid cells
    gy0, gy1 = max(0, int(min(gys)) - m), min(X.shape[0], int(max(gys)) + m + 2)
    gx0, gx1 = max(0, int(min(gxs)) - m), min(X.shape[1], int(max(gxs)) + m + 2)
    json.dump(dict(scroll=SCROLL, volume=VOL, voxel_um=UM, mesh=MESH, rectangle_mm=RECT, area_cm2=area,
                   levelling=dict(centre_render_px=c.tolist(), line_dir=d.tolist(), down_dir=n.tolist()),
                   corners=corners, row_centres_mm=[round(v, 2) for v, _ in rows], region_tifxyz_slice=[gy0, gy1, gx0, gx1],
                   maps=DR, render=RENDER, render_layer=20),
              open(f'{OUT}/region.json', 'w'), indent=1)

    # --- mesh: full patch + region cut
    full = f'{OUT}/mesh/{MESH}'; os.makedirs(full, exist_ok=True)
    for f in ('x.tif', 'y.tif', 'z.tif', 'meta.json'): shutil.copy2(f'{PATCH}/{f}', f'{full}/{f}')
    reg = f'{OUT}/mesh/{MESH}_region'; os.makedirs(reg, exist_ok=True)
    meta = json.load(open(f'{PATCH}/meta.json'))
    sub = [A[gy0:gy1, gx0:gx1] for A in (X, Y, Zt)]
    for a, A in zip('xyz', sub): tifffile.imwrite(f'{reg}/{a}.tif', A)
    ok = sub[2] > 0
    meta['bbox'] = [[float(A[ok].min()) for A in sub], [float(A[ok].max()) for A in sub]]
    meta['area_vx2'] = float(ok.sum() * GRID * GRID); meta['area_cm2'] = meta['area_vx2'] * (UM * 1e-4) ** 2
    meta['uuid'] = f'{MESH}_region'; meta['note'] = f'grid rows {gy0}:{gy1}, cols {gx0}:{gx1} of {MESH} (the 4 cm2 rectangle + 2 mm)'
    json.dump(meta, open(f'{reg}/meta.json', 'w'), indent=4)

    # --- flattening distortion inside the rectangle (grid edges vs the nominal 20 voxels, cell area, grid angle)
    P3 = np.stack([X, Y, Zt], -1).astype(np.float64); ok3 = Zt > 0
    poly = np.array([corners[k]['tifxyz_grid'][::-1] for k in ('top_left', 'top_right', 'bottom_right', 'bottom_left')], np.float32)
    msk = np.zeros(Zt.shape, np.uint8); cv2.fillPoly(msk, [np.round(poly * 16).astype(np.int32)], 1, shift=4); msk = msk > 0
    a_, b_, c_ = P3[:-1, :-1], P3[:-1, 1:], P3[1:, :-1]; inside = msk[:-1, :-1]; mm = ok3[:-1, :-1] & ok3[:-1, 1:] & ok3[1:, :-1] & inside
    eu = np.linalg.norm(b_ - a_, axis=-1)[mm]; ev = np.linalg.norm(c_ - a_, axis=-1)[mm]
    ar = np.linalg.norm(np.cross(b_ - a_, c_ - a_), axis=-1)[mm] / GRID ** 2
    an = np.degrees(np.arccos(np.clip(((b_ - a_)[mm] * (c_ - a_)[mm]).sum(-1) / eu / ev, -1, 1)))
    pc = lambda x: [round(float(np.percentile(x, q)), 3) for q in (5, 50, 95)]
    json.dump(dict(scope='cells inside the 4 cm2 rectangle', n_cells=int(mm.sum()), invalid_cells=int((inside & ~mm).sum()),
                   edge_u_vx_p5_p50_p95=pc(eu), edge_v_vx_p5_p50_p95=pc(ev), area_ratio_p5_p50_p95=pc(ar), grid_angle_deg_p5_p50_p95=pc(an),
                   frac_cells_area_off_by_more_than_10pct=round(float((np.abs(ar - 1) > 0.1).mean()), 3)),
              open(f'{OUT}/mesh/distortion_in_rectangle.json', 'w'), indent=1)

    # --- controls
    spec = dict(title=f'Hallucination control: {SCROLL} lead A rectangle on the own-sheet patch, max-quality recipe',
                target=dict(kind='deepread', dir='x_deepread/results/b1/out/patchW074', label=f'{SCROLL} lead A, 4 cm2 rectangle (bbox)',
                            roi=[int(min(cn['render_px'][1] for cn in corners.values())), int(max(cn['render_px'][1] for cn in corners.values())),
                                 int(min(cn['render_px'][0] for cn in corners.values())), int(max(cn['render_px'][0] for cn in corners.values()))]),
                blank='auto', artefact=dict(kind='zoom', dir='x_zoom/PHerc0813_z13088_w040_atlas', label='PHerc0813 atlas line = gap-pocket artefact'))
    json.dump(spec, open(f'{OUT}/controls/panel_spec.json', 'w'), indent=1)
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    rc = subprocess.run(['nice', '-n', '15', f'{W}/envs/vcpy/bin/python', f'{W}/x_refine/R2_triage_fp/control_panel.py', f'{OUT}/controls/panel_spec.json',
                         f'{OUT}/controls/hallucination_control_panel.jpg'], cwd=W, env=env, capture_output=True, text=True)
    open(f'{OUT}/controls/hallucination_control_panel.log', 'w').write(rc.stdout + rc.stderr)
    for src, dst in ((f'{W}/x_followup/A_depth/depth_stack.jpg', 'depth_sweep_-3_to_+3_vx.jpg'),
                     (f'{W}/x_followup/A_depth/depth_metrics.json', 'depth_sweep_metrics.json'),
                     (f'{W}/CANDIDATES/1_closest_calls/deepread/w074_CLOSEUP_native.jpg', 'same_line_on_spiral_fit_surface.jpg'),
                     (f'{W}/CANDIDATES/1_closest_calls/deepread/patchW074_CLOSEUP_native.jpg', 'same_line_on_own_sheet_patch.jpg'),
                     (f'{W}/CANDIDATES/1_closest_calls/deepread/NEIGHBOURS_w072-w074.jpg', 'neighbour_windings_w072-w074.jpg'),
                     (f'{W}/x_refine/trials/R2_triage_fp/panels/panel_0343_positive.jpg', 'reference_panel_0343_known_text.jpg')):
        if os.path.exists(src): shutil.copy2(src, f'{OUT}/controls/{dst}')
        else: print('missing', src)
    print(json.dumps(dict(area_cm2=area, rows=rows, corners={k: v['volume_xyz_vx'] for k, v in corners.items()}, slice=[gy0, gy1, gx0, gx1],
                          panel_rc=rc.returncode)), flush=True)


if __name__ == '__main__':
    main()
