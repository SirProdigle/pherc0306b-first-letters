"""compare_v2.py : the container's (or any re-run's) maps, CT layer and claim image against the shipped ones (lane I2).

  compare_v2.py --test-maps DIR --ref-maps DIR [--test-ct CT] [--ref-ct CT] [--test-img PNG --ref-img PNG] [--pair rt|rv8] [--models rv2,v8]
                [--crop 116,4167,694,7639] [--out compare_v2.json]

Maps (per model, on the raster window --crop): max / p99 / mean |diff| in 1/255 steps of probability, share identical, Pearson r.
Fused z and darkness (the claim display rule, claimlib_v2, primary rung, on the context grid): max |dz|, r, max |dd| x 255.
CT layer 20: share of identical pixels, max |diff|.
Image: pixels differing, max |diff| per channel (whole image and map body), PNG sha256 of both."""
import os, sys, json, argparse, hashlib
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claimlib_v2 as L, make_claim_image_v2 as C                                   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--test-maps', required=True); ap.add_argument('--ref-maps', required=True)
ap.add_argument('--test-ct', default=''); ap.add_argument('--ref-ct', default='')
ap.add_argument('--test-img', default=''); ap.add_argument('--ref-img', default='')
ap.add_argument('--pair', default='rt'); ap.add_argument('--models', default='', help='models for the per-model stats (default: the pair)'); ap.add_argument('--crop', default='116,4167,694,7639'); ap.add_argument('--out', default='')
a = ap.parse_args()
y0, y1, x0, x1 = (int(v) for v in a.crop.split(','))
pair = C.PAIRS[a.pair]
models = a.models.split(',') if a.models else pair
res = dict(crop=[y0, y1, x0, x1], models={})
for m in models:
    t = np.load(f'{a.test_maps}/fwd_{m}.npy', mmap_mode='r')[y0:y1, x0:x1].astype(np.float32)
    r = np.load(f'{a.ref_maps}/fwd_{m}.npy', mmap_mode='r')[y0:y1, x0:x1].astype(np.float32)
    ok = np.isfinite(t) & np.isfinite(r); d = np.abs(t - r)[ok]
    res['models'][m] = dict(px=int(ok.sum()), max_abs_x255=round(float(d.max() * 255), 3), p99_abs_x255=round(float(np.percentile(d, 99) * 255), 3),
                            mean_abs_x255=round(float(d.mean() * 255), 4), identical_frac=round(float((d == 0).mean()), 5),
                            pearson_r=round(float(np.corrcoef(t[ok], r[ok])[0, 1]), 7))
if a.test_ct and a.ref_ct:
    ct_t, ct_r = C.read_ct(a.test_ct).astype(np.int16), C.read_ct(a.ref_ct).astype(np.int16)
    dc = np.abs(ct_t - ct_r)
    res['ct_layer20'] = dict(identical_frac=round(float((dc == 0).mean()), 6), within1_frac=round(float((dc <= 1).mean()), 6), max_abs=int(dc.max()))
    blank = lambda lv: ['', '', '']                                                 # noqa: E731
    _, it, xt = C.build(a.test_maps, a.test_ct, a.pair, 'primary', blank, lambda lv: [''])
    _, ir, xr = C.build(a.ref_maps, a.ref_ct, a.pair, 'primary', blank, lambda lv: [''])
    v = xt['valid'] & xr['valid']
    dz = np.abs(xt['fused'] - xr['fused'])[v]; dd = np.abs(xt['d'] - xr['d'])[v] * 255
    res['claim_display'] = dict(levels_test=it['levels'], levels_ref=ir['levels'], valid_identical=bool((xt['valid'] == xr['valid']).all()),
                                fused_z_max_abs=round(float(dz.max()), 5), fused_z_r=round(float(np.corrcoef(xt['fused'][v], xr['fused'][v])[0, 1]), 7),
                                darkness_max_abs_x255=round(float(dd.max()), 3), darkness_p99_abs_x255=round(float(np.percentile(dd, 99)), 4),
                                rows_test=it['rows_map_px'], rows_ref=ir['rows_map_px'])
if a.test_img and a.ref_img:
    h = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()                 # noqa: E731
    it_, ir_ = cv2.imread(a.test_img).astype(np.int16), cv2.imread(a.ref_img).astype(np.int16)
    r = dict(sha_test=h(a.test_img), sha_ref=h(a.ref_img), png_identical=h(a.test_img) == h(a.ref_img), shape_test=list(it_.shape), shape_ref=list(ir_.shape))
    if it_.shape == ir_.shape:
        d = np.abs(it_ - ir_).max(-1); sj = json.load(open(a.ref_img[:-4] + '.json'))['image_px'] if os.path.exists(a.ref_img[:-4] + '.json') else None
        r.update(pixels_differing=int((d > 0).sum()), pixels_total=int(d.size), max_abs_diff=int(d.max()))
        if sj:
            b = d[sj['map_top']:sj['map_top'] + sj['map_h'], sj['map_left']:sj['map_left'] + sj['map_w']]
            r.update(body_pixels_differing=int((b > 0).sum()), body_max_abs_diff=int(b.max()), body_p99_9_abs_diff=float(np.percentile(b, 99.9)))
    res['image'] = r
if a.out: json.dump(res, open(a.out, 'w'), indent=1)
print(json.dumps(res, indent=1), flush=True)
