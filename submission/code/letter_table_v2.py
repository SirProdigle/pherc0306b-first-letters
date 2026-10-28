"""letter_table_v2.py PACKAGE_DIR MAPS_DIR CT [IMAGE.png [OUT_PREFIX]] : pixel and millimetre sizes of the letters of the line IN THE v2 CLAIM IMAGE
(<MESH>.png, reading orientation, 8.64 um/px).  Lane I2.

Letters are numbered by their position in the line, counted from the left of the image (no reading is submitted).
The papyrologist's tracings only say which blob is one letter (MEASUREMENT ONLY: never drawn on, never used to make or choose
anything).  Her tracings were drawn on the v1 surface (A's own GrowPatch render), so they are carried into the v2 image through
the exact 3-D correspondence: each node of the v2 mesh (V5Rc) is projected onto A's mesh (nearest point, sub-cell least squares),
giving A's grid position, hence A's render pixel; node positions are interpolated bilinearly to pixels.
Each size is the bounding box of the image's displayed ink at half tone or darker (darkness d >= 0.5 under the pre-registered
rule) inside her traced shape widened by 0.15 mm, cut to the letter's stretch along the line (her latest reading, the brief's
erratum, in A's claim-frame u) and to the line's band (|v| <= 2.6 mm about its centre, so a mark in the next row is not counted).
Tracing: label2 (her latest), label1 where label2 has no ink in a letter's stretch (as v1 did).
A box that touches its stretch's cut on a side is flagged: there the ink runs on into the neighbouring letter.
Writes PACKAGE_DIR/letter_dimensions.{json,md}."""
import os, sys, json
import numpy as np, cv2, tifffile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, f'{HERE}/surface')
import claimlib_v2 as L, make_claim_image_v2 as C                                  # noqa: E402
import p1geom as g                                                                  # noqa: E402
cv2.setNumThreads(4)
W = os.environ.get('FL_ROOT', '/home/liam/projects/vesuvius-work')
A_MESH = f'{W}/fl/sqm/PHerc0306B_f306_w074own/auto_grown_20261012144243662'
REGION = f'{W}/CANDIDATES/SUBMISSION_DRAFT_A_owner_only/region.json'               # A's levelling (= the v1 claim frame), read only
LAB = {'label2': f'{W}/x_followup/A_label/papyrologist_label2_2026-10-14_1145.png', 'label1': f'{W}/x_followup/A_label/papyrologist_label_2026-10-14.png'}
X0, Y0, SC = 426, 1145, 4017 / 2000                                                 # tracing frame = crop of A's render at 2000 px wide
STRETCH = [(1, -11.9, -9.45), (2, -9.45, -7.3), (3, -7.3, -4.0), (4, -3.95, -3.05), (5, -3.0, -0.6), (6, -0.6, 2.4),
           (7, 2.4, 4.55), (8, 4.55, 6.9), (9, 7.3, 10.9)]                          # (position, u from, u to) in A's claim frame, mm
REG = (-16.0, 14.0, -5.0, 5.0)                                                      # V5Rc nodes projected (claim frame u0, u1, v0, v1)


def main():
    pkg, maps, ct = sys.argv[1:4]; img = sys.argv[4] if len(sys.argv) > 4 else ''; prefix = sys.argv[5] if len(sys.argv) > 5 else f'{pkg}/letter_dimensions'
    mesh = [d for d in os.listdir(f'{pkg}/mesh') if os.path.isdir(f'{pkg}/mesh/{d}') and not d.endswith('_region')][0]
    img = img or f'{pkg}/{mesh}.png'; side = json.load(open(img[:-4] + '.json')); ip = side['image_px']; dsp = side['display']
    # 1. image-side: displayed darkness (as rendered) on the context grid, through the same build() as the image
    _, info, aux = C.build(maps, ct, side['ink']['pair'], dsp.get('rung') or 'primary', lambda lv: ['', '', ''], lambda lv: [''],
                           display=dsp.get('kind', 'threshold'), bg_mode=dsp.get('bg', 'ct'))
    d = aux['d'][::-1]                                                               # reading orientation, map pixels
    assert info['levels'] == side['display']['levels'], 'levels differ from the image sidecar'
    H, Wd = d.shape
    U, V = L.grid_uv(L.CTX, L.UMX); U, V = U[::-1], V[::-1]                         # claim-frame (u, v) of each image map pixel
    # 2. V5Rc node -> A grid (3-D projection) for the line region
    Pn = g.load(f'{pkg}/mesh/{mesh}'); NR, NC = Pn.shape[:2]; cell = 20 / L.K
    jj, ii = np.meshgrid(np.arange(NC), np.arange(NR)); Un, Vn = L.P1U0 + jj * cell, L.P1V0 + ii * cell
    reg = np.isfinite(Pn).all(-1) & (Un >= REG[0]) & (Un <= REG[1]) & (Vn >= REG[2]) & (Vn <= REG[3])
    SA = g.Surf(A_MESH, 'A', up=1)
    pr = SA.project(Pn[reg], max_d=30)
    good = np.isfinite(pr[:, 0]) & (np.abs(pr[:, 2]) < 6) & (pr[:, 3] < 3)
    ra = np.full((NR, NC), np.nan); ca = np.full((NR, NC), np.nan)
    ri, ci = np.nonzero(reg); ra[ri[good], ci[good]] = pr[good, 0]; ca[ri[good], ci[good]] = pr[good, 1]
    # 3. image pixel -> V5Rc node coords -> A render px (bilinear in the node grid)
    xr = (U - L.P1U0) * L.K - 0.5; yr = (V - L.P1V0) * L.K - 0.5                     # V5Rc render px (as sample_raster)
    gx, gy = (xr / 20).astype(np.float32), (yr / 20).astype(np.float32)
    f = lambda a: np.where(np.isfinite(a), a, -1e4).astype(np.float32)               # noqa: E731
    ax = cv2.remap(f(ca * 20), gx, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1e4)
    ay = cv2.remap(f(ra * 20), gx, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1e4)
    nodes_ok = cv2.remap(np.isfinite(ca).astype(np.float32), gx, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    okA = (ax > 0) & (ay > 0) & (nodes_ok > 0.999)                                 # all 4 surrounding nodes projected (no blend with a hole)
    # A's claim-frame (u, v) of each image pixel (her letter stretches are given there)
    R = json.load(open(REGION))['levelling']; c0 = np.array(R['centre_render_px']); dd = np.array(R['line_dir']); nn = np.array(R['down_dir'])
    uA = ((ax - c0[0]) * dd[0] + (ay - c0[1]) * dd[1]) / L.K; vA = ((ax - c0[0]) * nn[0] + (ay - c0[1]) * nn[1]) / L.K
    # 4. tracings into the image, widened 0.15 mm
    labs = {}
    for k, p in LAB.items():
        Lb = (cv2.imread(p, cv2.IMREAD_UNCHANGED)[..., :3].max(-1) > 127).astype(np.float32)
        full = np.zeros((int(Y0 + Lb.shape[0] * SC) + 10, int(X0 + Lb.shape[1] * SC) + 10), np.float32)
        bb = cv2.resize(Lb, (int(Lb.shape[1] * SC), int(Lb.shape[0] * SC)), interpolation=cv2.INTER_NEAREST); full[Y0:Y0 + bb.shape[0], X0:X0 + bb.shape[1]] = bb
        m = cv2.remap(full, ax.astype(np.float32), ay.astype(np.float32), cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.5
        r = int(round(0.15 * L.PPMX))
        labs[k] = cv2.dilate((m & okA).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    ink = d >= 0.5; band = np.abs(vA) <= 2.6
    rows = []
    for pos, a, b in STRETCH:
        cols = okA & (uA >= a) & (uA < b) & band
        for k in ('label2', 'label1'):
            mm = labs[k] & ink & cols
            if mm.sum() >= 50: break
        ys, xs = np.where(mm)
        if not len(xs): rows.append(dict(position_in_line=pos, measured=False)); continue
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        # touches the stretch cut? (ink of the box's own mask within 2 px of the u = a / u = b boundary)
        ub = uA[ys, xs]; tol = 2.5 / L.PPMX
        rows.append(dict(position_in_line=pos, measured=True, x_px=int(x0 + ip['map_left']), y_px=int(y0 + ip['map_top']), w_px=int(x1 - x0), h_px=int(y1 - y0),
                         w_mm=round((x1 - x0) / L.PPMX, 2), h_mm=round((y1 - y0) / L.PPMX, 2), tracing=k,
                         runs_into_left_neighbour=bool((ub < a + tol).any()) and pos > 1, runs_into_right_neighbour=bool((ub > b - tol).any()) and pos < 9,
                         ink_px=int(mm.sum())))
    proj = dict(nodes=int(reg.sum()), projected=int(good.sum()), median_abs_normal_offset_vx=round(float(np.median(np.abs(pr[good, 2]))), 3),
                median_inplane_residual_vx=round(float(np.median(pr[good, 3])), 3))
    out = dict(image=os.path.basename(img), variant=side.get('variant', {}).get('key'), px_um=8.64, px_per_mm=round(L.PPMX, 3), letters=rows,
               numbering='position in the line, counted from the left, in the image orientation (reading orientation); no reading is submitted',
               method='bounding box of the image\'s displayed ink at half tone or darker (d >= 0.5) inside the papyrologist-traced letter shape '
                      'widened 0.15 mm, cut to the letter\'s stretch along the line and to |v| <= 2.6 mm; tracings carried from the v1 surface '
                      'through the 3-D correspondence (V5Rc node -> nearest point on the v1 mesh)', projection=proj)
    json.dump(out, open(f'{prefix}.json', 'w'), indent=1, ensure_ascii=False)
    with open(f'{prefix}.md', 'w') as fh:
        fh.write('| letter (position in the line, from the left) | box in the image: x, y (px) | width × height (px) | width × height (mm) | note |\n|---|---|---|---|---|\n')
        for r in rows:
            if not r['measured']: fh.write(f"| {r['position_in_line']} | – | – | – | no ink at half tone in the traced shape |\n"); continue
            nb = [s for s, t in (('left', r['runs_into_left_neighbour']), ('right', r['runs_into_right_neighbour'])) if t]
            note = ('joined to its ' + ' and '.join(nb) + ' neighbour; box cut at the letter boundary') if nb else ''
            fh.write(f"| {r['position_in_line']} | {r['x_px']}, {r['y_px']} | {r['w_px']} × {r['h_px']} | {r['w_mm']} × {r['h_mm']} | {note} |\n")
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
