"""overlay_variants.py : legibility-first variants of the form's 'ink overlaid on the flattened surface' image (20 Oct ~17:00).
Same ink (the v4c main image's Reader V3 + TRACE darkness d, same levels, no threshold) and the same CT layer 20; only the display
differs: A = CT compressed to a faint pale grey (205..245) under near-black ink; B = same faint CT, ink in deep ink-brown; C = two
panels (A over the clean ink-only page) plus a 2x inset of the line. Drafts for the form image."""
import sys
import numpy as np, cv2
A10 = '/home/liam/projects/vesuvius-work/x_req/20261020/A10'; sys.path.insert(0, A10)
import box_search as S
CTX = S.CTX; K = S.K; F = cv2.FONT_HERSHEY_SIMPLEX
BOX = dict(u0=-15.0, u1=23.0, v0=-6.84, v1=2.72); CROP = dict(u0=-21.0, u1=29.0, v0=-13.0, v1=9.0)
x0, y0 = S.px(CTX['u0'], CTX['v0']); x1, y1 = S.px(CTX['u1'], CTX['v1']); sl = (slice(y0, y1), slice(x0, x1))
ct = cv2.imread(f'{S.M}/ct20.png', -1)[sl].astype(np.float32)
v3 = np.load(f'{S.M}/fwd_v3.npy', mmap_mode='r')[sl].astype(np.float32); tr = np.load(f'{S.M}/fwd_tr.npy', mmap_mode='r')[sl].astype(np.float32)
valid = (ct > 0) & np.isfinite(v3) & np.isfinite(tr) & (v3 != 0)
d, lv = S.view(('v3', 'tr'), sl, valid); assert abs(lv[0] + 0.300) < 0.01 and abs(lv[1] - 6.872) < 0.01, lv
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
    for mm in range(11): cv2.line(bot, (20 + int(round(mm * K)), 30), (20 + int(round(mm * K)), 74), (0, 0, 0), 3)
    cv2.putText(bot, '1 cm (ticks every 1 mm)', (40 + L, 64), F, 1.5, (0, 0, 0), 3, cv2.LINE_AA)
    return np.concatenate([top, img, bot])
HDR = 'PHerc0306B, volume 20250521133212 (8.64 um), surface PHerc0306B_20250521133212_w074own_touchup_V5Rc, reading orientation'
A = comp(205, 245, (18, 20, 22)); dashed(A, BOX, (220, 110, 40))
B = comp(205, 245, (20, 35, 70)); dashed(B, BOX, (220, 110, 40))
ink_only = np.where(vc[..., None], S.PAGE[::-1] * (1 - dc[..., None]) + S.INK[::-1] * dc[..., None], (200, 190, 182)).astype(np.uint8); dashed(ink_only, BOX, (220, 110, 40))
fA = frame(A, HDR, 'Ink (dark): Reader V3 + TRACE v3B, no threshold, over the CT at the papyrus surface shown faint. Dashed: the 4 cm2 box.')
fB = frame(B, HDR, 'Ink (brown): Reader V3 + TRACE v3B, no threshold, over the CT at the papyrus surface shown faint. Dashed: the 4 cm2 box.')
fI = frame(ink_only, HDR, 'Ink only (same pixels, same tones), on a plain page.')
# 2x inset of the line (u -15..+23, v -1..+3 in claim frame -> flipped view)
li = cv2.resize(np.flipud(A[S.px(0, -4)[1] - cy0 + 0:S.px(0, 3.2)[1] - cy0, S.px(-15.5, 0)[0] - cx0:S.px(23.5, 0)[0] - cx0]), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
li = cv2.resize(li, (fA.shape[1], int(li.shape[0] * fA.shape[1] / li.shape[1])), interpolation=cv2.INTER_AREA)
lab = np.full((70, fA.shape[1], 3), 255, np.uint8); cv2.putText(lab, 'The line enlarged (same image, 2x, from inside the box)', (20, 50), F, 1.4, (20, 20, 20), 3, cv2.LINE_AA)
fC = np.concatenate([fA, np.full((30, fA.shape[1], 3), 255, np.uint8), lab, li, np.full((30, fA.shape[1], 3), 255, np.uint8), fI])
for n, im in (('A_faintCT', fA), ('B_brownink', fB), ('C_twopanel_inset', fC)):
    cv2.imwrite(f'{A10}/overlay_variant_{n}.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, 92])
    s = 1600 / im.shape[1]; cv2.imwrite(f'/tmp/claude-1000/-home-liam-projects-vesuvius-work/7d89d68e-474d-4112-ba44-511303204b24/scratchpad/ov_{n}.jpg', cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])
print('ok', fA.shape, fC.shape)
