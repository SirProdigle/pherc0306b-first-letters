"""known_text_panels.py KT_DIR OUTDIR : known text through the rules of views 4 and 5 (Reader v2 + v8Q), each window at its own levels
(lane I2).  Reader v2 maps of each window are the ones behind I1's control panels; v8Q maps come from v8q_known_text.py
(P2's no-harm protocol: s21, d0, no TTA).  Fused z = mean of per-model z over the window's
on-sheet pixels.  View 4's rule: threshold, secondary rung (L = 0.5 x the window's p99), pale CT layer where one exists.  View 5's
rule: grey, the window's median .. p99.9, no CT.  Writes ctrl_known_v8Q_<set>_{view4,view5}.jpg and KT_DIR/known_text_panels.json.
v8Q TRAINED on 1447 w058 and on the 0343 tracings' crop: those two panels show training data, not a test (said in their headers)."""
import os, sys, json
import numpy as np, cv2, zarr, tifffile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claimlib_v2 as L                                                            # noqa: E402
W = os.environ.get('FL_ROOT', '/home/liam/projects/vesuvius-work')
K = 1000 / 8.64


def crop_from(kt, name, win):
    j = json.load(open(f'{kt}/v8q_{name}.json')); Y0, Y1, X0, X1 = j['crop_with_margin']; y0, y1, x0, x1 = win
    return np.load(f'{kt}/v8q_{name}.npy').astype(np.float32)[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0]


def windows(kt):
    """-> {set: (rv2, v8q, ct or None, orient fn, label)} on the same pixels"""
    out = {}
    y0, y1, x0, x1 = 2576, 4512, 389, 4748                                          # 1447 w058 (val1447 crop, x_gates render frame)
    rv = np.load(f'{W}/x_submit/val1447/maps.npz')['rv2'].astype(np.float32)
    ct = tifffile.imread(f'{W}/data/pherc1447/w058/layers/12.tif')[y0:y1, x0:x1].astype(np.float32)
    out['w058'] = (rv, crop_from(kt, 'w058', (0, y1 - y0, 0, x1 - x0)), ct, lambda a: a,     # the w058 render is the crop itself
                   'known text PHerc1447 w058, as stored (upright) - v8Q TRAINING DATA, not a test')
    Y0, Y1, X0, X1 = 450, 2350, 2400, 5300                                          # 0343 PANTOGINON crop (concat_66 frame)
    rv = np.mean([np.load(f'{W}/x_zoom/val343/rv2_o50_tta_d{dd}.npy').astype(np.float32) for dd in (-1, 0, 1)], 0)
    ct = np.asarray(zarr.open(f'{W}/fl/0343/concat_66.zarr', mode='r')['0'][33, Y0:Y1, X0:X1]).astype(np.float32)
    out['p0343'] = (rv, crop_from(kt, 'p0343', (Y0, Y1, X0, X1)), ct, lambda a: np.ascontiguousarray(a[::-1, ::-1]),
                    'known text PHerc0343, the traced PANTOGINON crop, reading orientation - v8Q TRAINING DATA (0343 tracings), not a test')
    MM = f'{W}/x_refine/trials/N8_submission_hygiene/matched343/w048'; yy, xx, H, Wd = 575, 230, int(15 * K), int(40 * K)   # 0343 lane w048
    rv = np.concatenate([np.load(f'{MM}/A/rev_rv2.npy').astype(np.float32)[:, :2752], np.load(f'{MM}/B/rev_rv2.npy').astype(np.float32)[:, 2752 - 2432:]], 1)[yy:yy + H, xx:xx + Wd]
    ct = np.concatenate([cv2.imread(f'{MM}/A/ct15.png', cv2.IMREAD_UNCHANGED)[:, :2752], cv2.imread(f'{MM}/B/ct15.png', cv2.IMREAD_UNCHANGED)[:, 2752 - 2432:]], 1)[yy:yy + H, xx:xx + Wd].astype(np.float32)
    out['lane'] = (rv, crop_from(kt, 'lane', (yy, yy + H, xx, xx + Wd)), ct, lambda a: np.ascontiguousarray(a[::-1]),
                   'known text PHerc0343 winding w048, away from the traced crop (not in v8Q\'s training crop), reading orientation')
    N1 = f'{W}/x_refine/trials/N1_stroke_refiner/inputs/w060'; by0, by1, bx0, bx1 = json.load(open(f'{N1}/meta.json'))['box']   # 1447 w060 (HF frame)
    r0, r1, c0, c1 = 1558, 1558 + int(15 * K), 1248, 4321                            # I1's panel window (HF frame)
    rv = np.load(f'{N1}/rv2.npy').astype(np.float32)[r0 - by0:r1 - by0, c0 - bx0:c1 - bx0]
    v8 = crop_from(kt, 'w060', (r0, r1, c0, c1))
    ct = tifffile.imread(f'{W}/x_refine/trials/O1_overlabel/data/hf/w060/layers/12.tif')[r0:r1, c0:c1].astype(np.float32)
    out['w060'] = (rv, v8, ct, lambda a: a, 'known text PHerc1447 w060 (held out from v8Q), as stored (upright)')
    return out


def main():
    kt, S = sys.argv[1:3]; os.makedirs(S, exist_ok=True); res = {}
    for name, (rv, v8, ct, R, label) in windows(kt).items():
        valid = (ct > 0) & (rv != 0) & np.isfinite(v8) & (v8 != 0)
        z = np.mean([L.zs(np.nan_to_num(m), valid) * valid for m in (rv, v8)], 0)
        bg_ct = L.light_ct_bg(ct, valid, valid); bg_pg = np.where(valid[..., None], L.PAGE[None, None], 60).astype(np.float32)
        lv4 = L.q4_levels(z, valid, 0.5); d4 = L.q4_tone(z, lv4, valid)
        lo, hi = float(np.percentile(z[valid], 50)), float(np.percentile(z[valid], 99.9)); d5 = np.where(valid, np.clip((z - lo) / (hi - lo), 0, 1), 0).astype(np.float32)
        for tag, d, bg, rule in (('view4', d4, bg_ct, f'view 4 rule: threshold, secondary rung, own levels (L {lv4["L"]:.2f}, p99 {lv4["H"]:.2f}, p99.9 {lv4["P"]:.2f}), pale CT'),
                                 ('view5', d5, bg_pg, f'view 5 rule: grey, own median {lo:.2f} .. p99.9 {hi:.2f}, no threshold, no CT')):
            img = L.frame_image(R(L.q4_rgb(d, bg, valid)), [f'CONTROL | Reader v2 + v8Q | {rule}; dark = ink', label], [], margin=40)
            cv2.imwrite(f'{S}/ctrl_known_v8Q_{name}_{tag}.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        res[name] = dict(label=label, px=int(valid.sum()), view4=dict(levels={k: round(v, 3) for k, v in lv4.items()}, painted=round(float((d4[valid] > 0).mean()), 4),
                                                                      half_tone=round(float((d4[valid] >= 0.5).mean()), 4)),
                         view5=dict(lo_median=round(lo, 3), hi_p999=round(hi, 3), p99_tone=round(float(np.percentile(d5[valid], 99)), 4),
                                    half_tone=round(float((d5[valid] >= 0.5).mean()), 4)))
        print(name, json.dumps(res[name]), flush=True)
    json.dump(res, open(f'{kt}/known_text_panels.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
