"""letter_table_v5.py PKG LABEL_PNG [MESH] : letter sizes in the package's main image from the papyrologist's 16 Oct label (measurement
only).  Version 5 copy of letter_table_label3.py (versions 4b-4c): the arithmetic is unchanged; the only change is that the offset
between her pack and the main image is taken from the main image's sidecar (map_left, map_top) instead of being fixed for version
4c's header height, so a header of another height cannot shift the measurement.  For version 4c's image the offset is (1713, 1062)
as before.

The label was drawn on the papyrologist pack of this surface (5278 x 2437 px at 8.64 um, claim frame u -17.1..+28.5, v -13.55..+7.5 mm,
reading orientation) and returned downscaled; it is upsampled (nearest) to that grid. Pack px (x, y) = map px (x + 1493, y + 867) of
the main image (= image px (x + 1713, y + 1062) for version 4c's margins; checked: r 0.992 between the pack's ink map and the main
image at that offset, best of +-3 px).
Each letter's box: the main image's own ink at half tone or darker (d >= 0.5, d recovered from the page/ink colours of the grey rule)
inside her traced shape widened 0.15 mm, cut to the letter's stretch along the line. Stretch boundaries are the minima of her label's
column profile near the letter boundaries of her earlier tracing. The dashed box outline (its rectangle +- 5 px, from <MESH>.json
claim_box_image_px) is not ink: version 3's table counted it where the box edge crossed a traced shape (version 4b fix).
The main image must be a grey view on the plain page (d is read back from its colours).  Nothing is drawn on any image."""
import sys, json
import cv2, numpy as np

PKG, LABEL = sys.argv[1], sys.argv[2]
MESH = sys.argv[3] if len(sys.argv) > 3 else 'PHerc0306B_20250521133212_w074own_touchup_V5Rc'
K = 1000 / 8.64; U0, V1 = -17.1, 7.5; GRID = (2437, 5278)
MX, MY = 1493, 867                                                                   # pack px -> map px of the main image
PAGE_G, INK_G = 236.0, 20.0                                                          # green channel of the page and ink colours
# her conservative reading, 16 Oct: "το δι' αυτο . . (.) τ/γον"
LETTERS = [('τ', -11.67, -9.55, 'read'), ('ο', -9.55, -7.41, 'read'), ('δ', -7.41, -3.99, 'read'), ('ι', -3.99, -2.92, 'read'),
           ('α', -2.92, -0.60, 'read'), ('υ', -0.60, 2.03, 'read'), ('τ', 2.03, 4.58, 'read'), ('ο', 4.58, 7.14, 'read'),
           ('.', 7.14, 9.54, 'traced, not identified'), ('.', 9.54, 12.63, 'traced, not identified'),
           ('τ/γ', 14.20, 16.89, 'read (τ or γ)'), ('ο', 16.89, 19.81, 'read'), ('ν', 19.81, 22.44, 'read')]


def main():
    side = json.load(open(f'{PKG}/{MESH}.json')); ip = side['image_px']
    disp = side.get('display', {})
    if disp.get('kind') != 'grey' or disp.get('bg') != 'page': raise SystemExit('the main image must be a grey view on the plain page')
    DX, DY = MX + ip['map_left'], MY + ip['map_top']
    lab = cv2.imread(LABEL, cv2.IMREAD_UNCHANGED); g = lab[..., :3].max(-1) if lab.ndim == 3 else lab
    L = cv2.resize((g > 127).astype(np.uint8), (GRID[1], GRID[0]), interpolation=cv2.INTER_NEAREST)
    Lw = cv2.dilate(L, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * 17 + 1, 2 * 17 + 1)))      # widened 0.15 mm
    img = cv2.imread(f'{PKG}/{MESH}.png')
    d = np.clip((PAGE_G - img[DY:DY + GRID[0], DX:DX + GRID[1], 1].astype(np.float32)) / (PAGE_G - INK_G), 0, 1)
    ink = (d >= 0.5) & (Lw > 0)
    bx = side['claim_box_image_px']; edge = np.zeros(img.shape[:2], np.uint8)
    cv2.rectangle(edge, (bx['x0'], bx['y0']), (bx['x1'], bx['y1']), 1, 11)        # the dashed outline, +- 5 px
    ink &= edge[DY:DY + GRID[0], DX:DX + GRID[1]] == 0
    X = lambda u: int(round((u - U0) * K - 0.5))
    out = []
    for i, (ch, a, b, cert) in enumerate(LETTERS, 1):
        xa, xb = X(a), X(b); m = np.zeros_like(ink); m[:, xa:xb] = ink[:, xa:xb]
        ys, xs = np.nonzero(m)
        r = dict(position_in_line=i, letter=ch, certainty=cert, measured=bool(len(xs)))
        if len(xs):
            x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            left = bool(ink[:, max(xa - 2, 0):xa + 2].any() and xa > 0 and i > 1)
            right = bool(ink[:, xb - 2:xb + 2].any() and i < len(LETTERS))
            r.update(x_px=int(x0 + DX), y_px=int(y0 + DY), w_px=int(x1 - x0), h_px=int(y1 - y0),
                     w_mm=round((x1 - x0) / K, 2), h_mm=round((y1 - y0) / K, 2), runs_into_left_neighbour=left,
                     runs_into_right_neighbour=right, ink_px=int(m.sum()), stretch_u_mm=[a, b],
                     note='faint: little ink at half tone' if (y1 - y0) / K < 1.2 else '')
        out.append(r)
    meta = dict(image=f'{MESH}.png', px_um=8.64, px_per_mm=round(K, 3),
                numbering='position in the line, counted from the left, in reading orientation',
                reading="το δι' αυτο . . (.) τ/γον (papyrologist, conservative label of 16 Oct 2026; tentative)",
                method="bounding box of the main image's ink at half tone or darker inside the papyrologist's traced shape (16 Oct label, "
                       "conservative version) widened 0.15 mm, cut to the letter's stretch along the line; measurement only",
                letters=out)
    json.dump(meta, open(f'{PKG}/letter_dimensions.json', 'w'), indent=1, ensure_ascii=False)
    rows = ['| letter | reading | box in the image: x, y (px) | width × height (px) | width × height (mm) | note |', '|---|---|---|---|---|---|']
    for r in out:
        if not r['measured']: rows.append(f"| {r['position_in_line']} | {r['letter']} | – | – | – | no ink at half tone |"); continue
        nb = [s for s, t in (('left', r['runs_into_left_neighbour']), ('right', r['runs_into_right_neighbour'])) if t]
        note = (f"joined to its {' and '.join(nb)} neighbour; box cut at the letter boundary" if nb else '')
        rows.append(f"| {r['position_in_line']} | {r['letter']} | {r['x_px']}, {r['y_px']} | {r['w_px']} × {r['h_px']} | "
                    f"{r['w_mm']} × {r['h_mm']} | {note} |")
    open(f'{PKG}/letter_dimensions.md', 'w').write('\n'.join(rows) + '\n')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
