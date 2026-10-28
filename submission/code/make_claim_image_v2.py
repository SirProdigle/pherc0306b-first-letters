"""make_claim_image_v2.py : the claim image of the package (v2 code + Reader V3 pairs, 20 Oct 2026; version 5: config-driven, 21 Oct).

Input: per-model ink maps on the claim-frame raster of the mesh's scale-1 render (fwd_<model>.npy in MAPS; float16 probabilities,
NaN where not inked; a map cropped from the raster carries its crop as 'box' (y0, y1, x0, x1) in fwd_<model>.json; render px (x, y)
<-> claim frame (u, v) = (-36 + x / K, -21 + y / K) mm, K = 115.741 px/mm), and the render's surface CT layer 20 (ct20.png, or the
render zarr whose ['0'][20] is that layer).
Output: <NAME>.png (+ <NAME>.json sidecar with every input sha256, parameter, level and position), at 8.64 um per pixel:
  context u -30..+30 x v -20..+15 mm, READING orientation (the render flipped top-to-bottom), dark ink on the pale surface CT
  layer or on a plain page under the display rule fixed in advance (claimlib_v2: PREREG A1.1, primary rung L = 0.35 x p99 unless
  --rung secondary; or the grey rule A6), the 4 cm2 claim box dashed, text rows marked by short ticks in the side margins only,
  1 cm bar, header naming scroll / volume / mesh / models / rule.  No character is drawn or annotated; nothing is drawn on the ink.
Version 5 (x_push3/v5): --config CFG.json (build_v5.py) supplies the claim box, the model names and recipes, the model pairs
(fusions of any number of models), models whose value 0 is data ('zero_is_data': coverage = finite and inside the crop, instead of
'map != 0'), the mesh note of the header and the sidecar's status line.  Without --config every value is version 4c's, and with the
version 4c config the PNG is byte-identical to version 4c's (x_push3/v5/proof/).  The arithmetic is version 4c's, unchanged.

  make_claim_image_v2.py --maps DIR --ct CT --mesh MESHDIR --out DIR [--config CFG.json] [--pair KEY] [--rung primary|secondary]
                         [--display threshold|grey] [--bg ct|page] [--name NAME]
"""
import os, sys, json, argparse
import numpy as np, cv2, zarr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claimlib_v2 as L                                                            # noqa: E402

SCROLL, VOL = 'PHerc0306B', '20250521133212'
NAMES = {'rv2': 'Reader v2', 'tr': 'TRACE v3B', 'v8': 'v8-in', 'v8q': 'v8Q', 'v3': 'Reader V3'}     # v8-in = released base weights, P2's fixed stitcher; v8Q = our fine-tune
PAIRS = {'rt': ('rv2', 'tr'), 'rv8': ('rv2', 'v8'), 'rv8q': ('rv2', 'v8q'), 'v3': ('v3',), 'v3t': ('v3', 'tr'), 'v3q': ('v3', 'v8q')}   # A_V3 adds V3 pairs
MARGIN = 220
RECIPES = {'rv2': 'Reader v2 (HF domenicor046/reader-v2 step 40000): villa inference, overlap 0.5, 8-fold dihedral TTA, mean of depths -1/0/+1 around layer 20, face fwd',
           'tr': 'TRACE v3B (GitHub Jashann/vesuvius-scrolling v1.0): as Reader v2',
           'v8': 'v8-in base (HF YoussefMoNader/ink-8um-v8in, model.safetensors sha256 3b94548d...c4cd): 24 layers 8..31 (window centred on layer 20, delta 0), '
                 'stored order (asc), clip 200, 64 px tiles at stride 21 admitted at >= 25 % CT coverage, mirror4 TTA, fp16, Gaussian (+-1 sigma) '
                 'stitch divided by the weight SUM',
           'v8q': 'v8Q = v8-in base fine-tuned by lane P2 on PHerc1447 w058 (team labels) + PHerc0343 (jeff-j-chen tracings) 1:1, 300 steps, NO PHerc0306B '
                  'data (ckpt sha256 4cb8e921...f305); inference exactly as v8-in base',
           'v3': 'Reader V3 (HF domenicor046/reader-v3, reader-v3.pth sha256 ea405417...4d8a, MIT; trained on no eligible scroll): as Reader v2'}
DISPLAYS = ('threshold', 'grey')
ZERO_IS_DATA = set()                                                                # v5: models whose 0 is data (coverage = finite, in crop)
RULES = {}                                                                          # v6: a single-map view's ink-rule text (e.g. a product map)
TEXT = dict(mesh_note='own-sheet GrowPatch + CT-driven touch-up at the right end of the line, re-flattened',
            status='version 4c (Reader V3 + TRACE v3B; claim box drawn by the team leader on the model image)',
            script='code/make_claim_image_v2.py + code/claimlib_v2.py',
            footer_note='')                                                         # v5: an optional footer line (empty = v4c's footer)


def configure(cfg):
    """v5: names, recipes, pairs, zero-is-data models, texts and the claim box from a build config (build_v5.py)"""
    global NAMES, RECIPES, PAIRS
    if not cfg: return
    if cfg.get('models'):
        NAMES = {k: m['name'] for k, m in cfg['models'].items()}
        RECIPES = {k: m['recipe'] for k, m in cfg['models'].items()}
        ZERO_IS_DATA.clear(); ZERO_IS_DATA.update(k for k, m in cfg['models'].items() if m.get('zero_is_data'))
        RULES.clear(); RULES.update({k: m['rule'] for k, m in cfg['models'].items() if m.get('rule')})
    if cfg.get('pairs'): PAIRS = {k: tuple(v) for k, v in cfg['pairs'].items()}
    TEXT.update({k: v for k, v in cfg.get('texts', {}).items() if k in TEXT})
    L.configure(cfg)


def load_config(path):
    if not path: return {}
    cfg = json.load(open(path)); configure(cfg); return cfg


def read_ct(path):
    if path.endswith('.png'): return cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return np.asarray(zarr.open(path, mode='r')['0'][20])


def mesh_box_area(mesh, box):
    """3-D area (cm2) of the claim box on the mesh: triangle areas per grid cell, weighted by the cell's overlap with the box"""
    import tifffile
    P = np.stack([tifffile.imread(f'{mesh}/{c}.tif') for c in 'xyz'], -1).astype(np.float64)
    ok = np.all(np.isfinite(P), -1) & np.all(P > 0, -1); H, Wd = ok.shape; cell = 20 / L.K
    P00, P01, P10, P11 = P[:-1, :-1], P[:-1, 1:], P[1:, :-1], P[1:, 1:]
    a = 0.5 * np.linalg.norm(np.cross(P01 - P00, P10 - P00), axis=-1) + 0.5 * np.linalg.norm(np.cross(P01 - P11, P10 - P11), axis=-1)
    cok = ok[:-1, :-1] & ok[:-1, 1:] & ok[1:, :-1] & ok[1:, 1:]
    j = np.arange(Wd - 1); i = np.arange(H - 1)
    fu = np.clip(np.minimum(L.P1U0 + (j + 1) * cell, box['u1']) - np.maximum(L.P1U0 + j * cell, box['u0']), 0, None) / cell
    fv = np.clip(np.minimum(L.P1V0 + (i + 1) * cell, box['v1']) - np.maximum(L.P1V0 + i * cell, box['v0']), 0, None) / cell
    return float((np.where(cok, a, 0) * fv[:, None] * fu[None, :]).sum()) * (L.UM / 1000) ** 2 / 100


def load(maps, ct_path, pair, ctx):
    files = {k: f'{maps}/fwd_{k}.npy' for k in pair}
    ct = read_ct(ct_path).astype(np.float32)
    m = {k: L.sample_raster(L.load_raster_map(f, ct.shape), ctx, L.UMX) for k, f in files.items()}
    has = [(L.sample_raster(L.load_raster_cov(files[k], ct.shape), ctx, L.UMX, nn=True) > 0) if k in ZERO_IS_DATA else (a != 0) for k, a in m.items()]
    valid = (L.sample_raster((ct > 0).astype(np.uint8), ctx, L.UMX, nn=True) > 0) & np.logical_and.reduce(has)
    return m, L.sample_raster(ct, ctx, L.UMX), valid, files


def build(maps, ct_path, pair_key, rung, header, footer, display='threshold', bg_mode='ct'):
    """-> (RGB image, info).  header / footer: callables (levels dict) -> list of text lines (so the levels can be printed).
    display 'threshold' = PREREG A1.1 (L = rung x p99 of the box, full tone at p99, to p99.9); 'grey' = PREREG A6: tone linear from the
    context median (none) to the context p99.9 (full), no threshold.  bg_mode 'ct' = pale CT layer at 25 % contrast; 'page' = no CT."""
    pair = PAIRS[pair_key]
    m, ct, valid, files = load(maps, ct_path, pair, L.CTX)
    U, V = L.grid_uv(L.CTX, L.UMX)
    inbox = L.box_mask(U, V, L.CBOX) & valid
    z = {k: L.zs(m[k], valid) * valid for k in m}                                  # per-model z over the context's on-sheet px
    fused = np.mean([z[k] for k in pair], 0)
    bg = L.light_ct_bg(ct, valid, valid) if bg_mode == 'ct' else np.where(valid[..., None], L.PAGE[None, None], 60).astype(np.float32)
    R = lambda arr: np.ascontiguousarray(arr[::-1])                                 # noqa: E731  reading orientation
    PP = L.PPMX
    bx0 = int((L.CBOX['u0'] - L.CTX['u0']) * PP); bx1 = int((L.CBOX['u1'] - L.CTX['u0']) * PP)
    by0 = int((L.CTX['v1'] - L.CBOX['v1']) * PP); by1 = int((L.CTX['v1'] - L.CBOX['v0']) * PP)
    if display == 'grey':
        lo, hi = float(np.percentile(fused[valid], 50)), float(np.percentile(fused[valid], 99.9))
        lv = dict(lo_median=lo, hi_p999=hi); d = np.where(valid, np.clip((fused - lo) / (hi - lo), 0, 1), 0).astype(np.float32)
    else:
        lv = L.q4_levels(fused, inbox, L.RUNGS[rung]); d = L.q4_tone(fused, lv, valid)
    rgb = R(L.q4_rgb(d, bg, valid)); dR = R(d); inR = R(inbox)
    rows, prof = L.rows_margin(dR, inR, PP)
    img = rgb.copy(); L.dashed_rect(img, bx0, by0, bx1, by1, (30, 90, 220))
    hl, fl = header(lv), footer(lv)
    out = L.frame_image(img, hl, fl, rows)
    head_h = int(26 * L.HDR_SCALE) * len(hl) + 24
    info = dict(levels={k: round(v, 3) for k, v in lv.items()}, rows_map_px=[int(r) for r in rows],
                visible_share_box=round(float((d[inbox] > 0).mean()), 4), full_tone_share_box=round(float((d[inbox] >= 0.85).mean()), 4),
                half_tone_share_box=round(float((d[inbox] >= 0.5).mean()), 4), p99_tone_box=round(float(np.percentile(d[inbox], 99)), 4),
                files=files, map_shape=list(rgb.shape[:2]), box_map_px=dict(x0=bx0, x1=bx1, y0=by0, y1=by1),
                image_px=dict(width=int(out.shape[1]), height=int(out.shape[0]), map_left=MARGIN, map_top=head_h,
                              map_w=int(rgb.shape[1]), map_h=int(rgb.shape[0])))
    return out, info, dict(fused=fused, d=d, valid=valid, inbox=inbox, z=z)


def texts(mesh_name, pair_key, display, rung, bg, area3):
    """the header and footer callables of the claim image (version 4c's wording; models, box and areas from the arguments)"""
    pair = PAIRS[pair_key]; models = ' + '.join(NAMES[k] for k in pair)
    fus = RULES[pair[0]] if len(pair) == 1 and pair[0] in RULES else 'mean of per-model z-scores'   # v6 hook; v2-v5 views unchanged
    flat = (L.CBOX['u1'] - L.CBOX['u0']) * (L.CBOX['v1'] - L.CBOX['v0']) / 100
    bw, bh = L.CBOX['u1'] - L.CBOX['u0'], L.CBOX['v1'] - L.CBOX['v0']
    frac = L.RUNGS[rung]
    bgtxt = 'pale CT surface layer behind (25 % contrast)' if bg == 'ct' else 'no CT layer (ink on a plain page)'
    rule2 = lambda lv: (f'Ink: {models}, {fus}; dark = ink. Display rule fixed in advance: tone from L = {frac:.2f} x p99 '   # noqa: E731
                        f'to p99 of the 4 cm2 area (z {lv["L"]:.2f} .. {lv["H"]:.2f}), continuing to p99.9 ({lv["P"]:.2f}); {bgtxt}.') if display == 'threshold' else \
        (f'Ink: {models}, {fus}; dark = ink. Display rule fixed in advance (grey, no threshold): tone linear from the median of the '
         f'shown area (z {lv["lo_median"]:.2f}, none) to its 99.9th percentile (z {lv["hi_p999"]:.2f}, full); {bgtxt}.')
    header = lambda lv: [                                                         # noqa: E731
        f'{SCROLL} | volume {VOL} (8.64 um, 116 keV) | mesh {mesh_name} ({TEXT["mesh_note"]})',
        rule2(lv),
        f'Reading orientation (render flipped top-to-bottom). Shown: u -30..+30 x v -20..+15 mm. Dashed box: the 4 cm2 area '
        f'({bw:.1f} x {bh:.2f} mm = {flat:.3f} cm2 flat; {area3:.3f} cm2 on the 3-D mesh). Rows: margin ticks only. Nothing drawn on the ink.']
    footer = lambda lv: [f'{L.UMX} um/px. No character is drawn or annotated.' + (f' {TEXT["footer_note"]}' if TEXT['footer_note'] else '')]   # noqa: E731
    return header, footer, flat, bw, bh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--maps', required=True); ap.add_argument('--ct', required=True); ap.add_argument('--mesh', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--pair', default='rt'); ap.add_argument('--config', default='')
    ap.add_argument('--rung', default='primary', choices=('primary', 'secondary')); ap.add_argument('--name', default='')
    ap.add_argument('--display', default='threshold', choices=DISPLAYS); ap.add_argument('--bg', default='ct', choices=('ct', 'page'))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); load_config(a.config)
    if a.pair not in PAIRS: raise SystemExit(f'--pair {a.pair}: not one of {sorted(PAIRS)}')
    mesh = os.path.basename(os.path.normpath(a.mesh)); name = a.name or mesh
    pair = PAIRS[a.pair]
    area3 = mesh_box_area(a.mesh, L.CBOX)
    header, footer, flat, bw, bh = texts(mesh, a.pair, a.display, a.rung, a.bg, area3)
    out, info, _ = build(a.maps, a.ct, a.pair, a.rung, header, footer, a.display, a.bg)
    fn = f'{a.out}/{name}.png'
    cv2.imwrite(fn, cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 6])
    ip = info['image_px']; b = info['box_map_px']
    side = dict(schema='make_claim_image_v2/1', status=TEXT['status'], scroll='PHerc. 0306B', volume_id=VOL, voxel_size_um=8.64,
                mesh=dict(name=mesh, files={f: L.sha(f'{a.mesh}/{f}') for f in ('x.tif', 'y.tif', 'z.tif', 'meta.json')}),
                orientation='reading orientation = the render flipped top-to-bottom (image row r <-> v = +15 - r / K mm)',
                frame='claim frame (u, v) mm, v down as rendered; render px (x, y) = ((u + 36) K, (v + 21) K), K = 115.741 px/mm',
                context_mm=L.CTX, claim_box_mm=L.CBOX, claim_box_size_mm=[round(bw, 2), round(bh, 2)],
                claim_box_area_cm2=dict(flat=round(flat, 4), mesh3d=round(area3, 4)),
                claim_box_image_px=dict(x0=ip['map_left'] + b['x0'], x1=ip['map_left'] + b['x1'], y0=ip['map_top'] + b['y0'], y1=ip['map_top'] + b['y1']),
                image_px=ip, px_per_mm=round(L.PPMX, 3),
                ink=dict(pair=a.pair, models=[NAMES[k] for k in pair],
                         maps={k: dict(file=os.path.realpath(f), sha256=L.sha(f)) for k, f in info['files'].items()},
                         recipes=RECIPES,
                         fusion='mean of per-model z-scores; each model z-scored over the on-sheet pixels of the context view'),
                ct=dict(file=os.path.realpath(a.ct), sha256=L.sha(a.ct) if os.path.isfile(a.ct) else 'render zarr, layer 20', layer=20),
                display=dict(kind=a.display, bg=a.bg,
                             rule=('d = 0.85*clip((z-L)/(H-L),0,1) + 0.15*clip((z-H)/(P-H),0,1); H = p99, P = p99.9 of fused z over the on-sheet '
                                   'pixels of the claim box; L = 0.35*H (primary) or 0.5*H (secondary)') if a.display == 'threshold' else
                                  ('d = clip((z - lo)/(hi - lo), 0, 1); lo = median, hi = p99.9 of fused z over the on-sheet pixels of the shown area '
                                   '(context); no threshold'),
                             pixel='pixel = bg*(1-d) + INK*d; ' + ('bg = PAGE*(0.75 + 0.25*ct), ct = linear 1-99.5 % stretch of the surface CT layer over the '
                                                                   'context on-sheet pixels, no CLAHE' if a.bg == 'ct' else 'bg = PAGE (no CT)') + '; off-sheet blue-grey',
                             rung=a.rung if a.display == 'threshold' else None, levels=info['levels'], page_rgb=L.PAGE.tolist(), ink_rgb=L.INK.tolist(),
                             fixed_in_advance='x_refine/trials/I1_display_read/PREREG.md, ' + ('Amendment 1 (A1.1), 3, 4 and 5 (rung by the blank rule)'
                                              if a.display == 'threshold' else 'Amendment 6 (grey view)') + ', written before the panels'),
                rows=dict(method='row profile of displayed ink (d >= 0.5) inside the box, smoothed 0.6 mm, peaks >= 3 mm apart; ticks 1.6 mm below the '
                                 'row centre, in the side margins only', centre_y_image_px=[ip['map_top'] + r for r in info['rows_map_px']]),
                visible_share_box=info['visible_share_box'], full_tone_share_box=info['full_tone_share_box'],
                half_tone_share_box=info['half_tone_share_box'], p99_tone_box=info['p99_tone_box'],
                image_sha256=L.sha(fn), script=TEXT['script'])
    if ZERO_IS_DATA & set(pair): side['ink']['zero_is_data'] = sorted(ZERO_IS_DATA & set(pair))
    json.dump(side, open(f'{a.out}/{name}.json', 'w'), indent=1, ensure_ascii=False)
    print(json.dumps(dict(image=fn, sha256=side['image_sha256'], levels=info['levels'], rows=side['rows']['centre_y_image_px'],
                          box_px=side['claim_box_image_px'], area=side['claim_box_area_cm2']), indent=1))


if __name__ == '__main__':
    main()
