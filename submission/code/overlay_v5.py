"""overlay_v5.py : the First Letters form's 'ink overlaid on the flattened surface' images (version 5 copy of A10's overlay_variants.py,
20 Oct).  Same ink as the package main image (the lead pair's grey-rule darkness d on the render raster's
integer pixels, as box_search.view: no threshold), over the CT layer 20; only the display differs:
  A = CT compressed to a faint pale grey (205..245) under near-black ink; B = the same faint CT, ink in deep ink-brown;
  C = two panels (A over the clean ink-only page) plus a 2x inset of the line (the form image of version 4c).
Parametrised by the build config (maps, lead pair, its name for the subtitles, box, crop, inset); the arithmetic is unchanged, so
with the version 4c config the three JPEGs are version 4c's byte for byte.  Nothing is drawn on the ink."""
import os, sys, json
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import box_count_v5 as S                                                              # noqa: E402  (px, view, load, CTX, K)
F = cv2.FONT_HERSHEY_SIMPLEX
PAGE = np.array([243, 236, 222], np.float32); INK = np.array([22, 20, 18], np.float32)


def overlays(maps_dir, keys, models_txt, box, out_dir, crop=None, inset=None, mesh='PHerc0306B_20250521133212_w074own_touchup_V5Rc', expect_levels=None):
    CTX = S.CTX; K = S.K; BOX = box
    CROP = crop or dict(u0=-21.0, u1=29.0, v0=-13.0, v1=9.0); INS = inset or dict(u0=-15.5, u1=23.5, v0=-4.0, v1=3.2)
    x0, y0 = S.px(CTX['u0'], CTX['v0']); x1, y1 = S.px(CTX['u1'], CTX['v1']); sl = (slice(y0, y1), slice(x0, x1))
    ct = cv2.imread(f'{maps_dir}/ct20.png', -1)[sl].astype(np.float32)
    mm = {k: S.load(f'{maps_dir}/fwd_{k}.npy', sl) for k in keys}
    valid = (ct > 0) & np.logical_and.reduce([np.isfinite(a) for a in mm.values()]) & (mm[keys[0]] != 0)
    d, lv = S.view(maps_dir, keys, sl, valid)
    if expect_levels: assert abs(lv[0] - expect_levels[0]) < 0.01 and abs(lv[1] - expect_levels[1]) < 0.01, lv
    cx0, cy0 = S.px(CROP['u0'], CROP['v0']); cx1, cy1 = S.px(CROP['u1'], CROP['v1']); c = (slice(cy0 - y0, cy1 - y0), slice(cx0 - x0, cx1 - x0))
    ctc, dc, vc = ct[c], d[c], valid[c]
    lo, hi = np.percentile(ctc[vc], 1), np.percentile(ctc[vc], 99.5); t = np.clip((ctc - lo) / (hi - lo), 0, 1)

    def comp(ct_lo, ct_hi, ink_bgr):
        g = ct_lo + (ct_hi - ct_lo) * t; base = np.repeat(g[..., None], 3, -1)
        out = base * (1 - dc[..., None]) + np.array(ink_bgr, np.float32) * dc[..., None]; out[~vc] = (200, 190, 182); return out.astype(np.uint8)

    def dashed(img, b, col, th=5, dash=40, gap=24):
        p0 = S.px(b['u0'], b['v0']); p1 = S.px(b['u1'], b['v1']); (xa, ya), (xb, yb) = (p0[0] - cx0, p0[1] - cy0), (p1[0] - cx0, p1[1] - cy0)
        for (a, e, f, hz) in ((xa, xb, ya, 1), (xa, xb, yb, 1), (ya, yb, xa, 0), (ya, yb, xb, 0)):
            s = min(a, e)
            while s < max(a, e):
                q = min(s + dash, max(a, e)); cv2.line(img, (s, f) if hz else (f, s), (q, f) if hz else (f, q), col, th, cv2.LINE_AA); s = q + gap

    def frame(img, title, sub):
        img = np.flipud(img).copy(); W = img.shape[1]
        top = np.full((150, W, 3), 255, np.uint8); cv2.putText(top, title, (20, 60), F, 1.5, (20, 20, 20), 3, cv2.LINE_AA); cv2.putText(top, sub, (20, 120), F, 1.2, (40, 40, 40), 2, cv2.LINE_AA)
        bot = np.full((120, W, 3), 255, np.uint8); L = int(round(10 * K)); cv2.rectangle(bot, (20, 40), (20 + L, 64), (0, 0, 0), -1)
        for mm_ in range(11): cv2.line(bot, (20 + int(round(mm_ * K)), 30), (20 + int(round(mm_ * K)), 74), (0, 0, 0), 3)
        cv2.putText(bot, '1 cm (ticks every 1 mm)', (40 + L, 64), F, 1.5, (0, 0, 0), 3, cv2.LINE_AA)
        return np.concatenate([top, img, bot])

    HDR = f'PHerc0306B, volume 20250521133212 (8.64 um), surface {mesh}, reading orientation'
    A = comp(205, 245, (18, 20, 22)); dashed(A, BOX, (220, 110, 40))
    B = comp(205, 245, (20, 35, 70)); dashed(B, BOX, (220, 110, 40))
    ink_only = np.where(vc[..., None], PAGE[::-1] * (1 - dc[..., None]) + INK[::-1] * dc[..., None], (200, 190, 182)).astype(np.uint8); dashed(ink_only, BOX, (220, 110, 40))
    fA = frame(A, HDR, f'Ink (dark): {models_txt}, no threshold, over the CT at the papyrus surface shown faint. Dashed: the 4 cm2 box.')
    fB = frame(B, HDR, f'Ink (brown): {models_txt}, no threshold, over the CT at the papyrus surface shown faint. Dashed: the 4 cm2 box.')
    fI = frame(ink_only, HDR, 'Ink only (same pixels, same tones), on a plain page.')
    li = cv2.resize(np.flipud(A[S.px(0, INS['v0'])[1] - cy0 + 0:S.px(0, INS['v1'])[1] - cy0, S.px(INS['u0'], 0)[0] - cx0:S.px(INS['u1'], 0)[0] - cx0]), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    li = cv2.resize(li, (fA.shape[1], int(li.shape[0] * fA.shape[1] / li.shape[1])), interpolation=cv2.INTER_AREA)
    lab = np.full((70, fA.shape[1], 3), 255, np.uint8); cv2.putText(lab, 'The line enlarged (same image, 2x, from inside the box)', (20, 50), F, 1.4, (20, 20, 20), 3, cv2.LINE_AA)
    fC = np.concatenate([fA, np.full((30, fA.shape[1], 3), 255, np.uint8), lab, li, np.full((30, fA.shape[1], 3), 255, np.uint8), fI])
    os.makedirs(out_dir, exist_ok=True); files = {}
    for n, im in (('A_faintCT', fA), ('B_brownink', fB), ('C_twopanel_inset', fC)):
        fn = f'{out_dir}/overlay_variant_{n}.jpg'; cv2.imwrite(fn, im, [cv2.IMWRITE_JPEG_QUALITY, 92]); files[n] = fn
    return dict(levels=[round(lv[0], 4), round(lv[1], 4)], files=files, shape_C=list(fC.shape))


if __name__ == '__main__':
    a = json.load(open(sys.argv[1]))                                                   # dict(maps_dir, keys, models_txt, box, out_dir, ...)
    print(json.dumps(overlays(**a), indent=1))
