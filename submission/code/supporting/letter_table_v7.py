"""letter_table_v7.py PKG LABEL_PNG [MESH] : letter sizes in the package's main image from the papyrologist's 23 Oct tracing (measurement
only).  Version 7 copy of letter_table_v5.py: the arithmetic is unchanged (same pack -> image offset, same half-tone ink inside her
traced shape widened 0.15 mm, same dashed-outline exclusion).  What changes:
  - the tracing is her 23 Oct one (same 5278 x 2437 pack grid as the 16 Oct label), and the reading is her 23 Oct reading;
  - that tracing also covers the rows below line 1, so each letter is cut to its line's band of rows as well as its stretch;
  - line 1's stretches are the column edges of the letter-aligned evidence grid (comparisons/), i.e. her narrow points between
    letters moved to the nearest gap in the models' ink; line 2's are the narrowest columns of her tracing between the letters
    she reads there.
Nothing is drawn on any image, and the table is not used to choose anything."""
import sys, json
import cv2, numpy as np

PKG, LABEL = sys.argv[1], sys.argv[2]
MESH = sys.argv[3] if len(sys.argv) > 3 else 'PHerc0306B_20250521133212_w074own_touchup_V5Rc'
K = 1000 / 8.64; U0, V1 = -17.1, 7.5; GRID = (2437, 5278)
MX, MY = 1493, 867                                                                   # pack px -> map px of the main image
PAGE_G, INK_G = 236.0, 20.0                                                          # green channel of the page and ink colours
U = lambda x: round(x / K + U0, 2)                                                   # pack px -> u mm
E1 = [627, 883, 1128, 1519, 1645, 1829, 2259, 2490, 2750, 3060, 3270, 3470, 3705, 4024, 4270, 4580]
# her 23 Oct reading: line 1 τὸ̣ δι’ αὑτὸ α̣ἱ̣ρ̣ε̣τ̣όν, line 2 π̣ [ (a dot under a letter: not certain)
R1 = [('τ', ''), ('ο̣', 'dotted'), ('δ', ''), ('ι', ''), ('α', ''), ('υ', ''), ('τ', ''), ('ο', ''),
      ('α̣', 'dotted'), ('ι̣', 'dotted'), ('ρ̣', 'dotted'), ('ε̣', 'dotted'), ('τ̣', 'dotted'), ('ο', ''), ('ν', '')]
LINES = [dict(line=1, rows=(450, 1180), letters=[(ch, U(E1[i]), U(E1[i + 1]), c) for i, (ch, c) in enumerate(R1)]),
         dict(line=2, rows=(1180, 1760), letters=[('π̣', U(458), U(720), 'dotted')])]


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
    ink0 = (d >= 0.5) & (Lw > 0)
    bx = side['claim_box_image_px']; edge = np.zeros(img.shape[:2], np.uint8)
    cv2.rectangle(edge, (bx['x0'], bx['y0']), (bx['x1'], bx['y1']), 1, 11)        # the dashed outline, +- 5 px
    ink0 &= edge[DY:DY + GRID[0], DX:DX + GRID[1]] == 0
    X = lambda u: int(round((u - U0) * K - 0.5))
    out = []
    for ln in LINES:
        ink = np.zeros_like(ink0); ink[ln['rows'][0]:ln['rows'][1]] = ink0[ln['rows'][0]:ln['rows'][1]]
        n = len(ln['letters'])
        for i, (ch, a, b, cert) in enumerate(ln['letters'], 1):
            xa, xb = X(a), X(b); m = np.zeros_like(ink); m[:, xa:xb] = ink[:, xa:xb]
            ys, xs = np.nonzero(m)
            r = dict(line=ln['line'], position_in_line=i, letter=ch, certainty=cert or 'read', measured=bool(len(xs)))
            if len(xs):
                x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
                left = bool(ink[:, max(xa - 2, 0):xa + 2].any() and xa > 0 and i > 1)
                right = bool(ink[:, xb - 2:xb + 2].any() and i < n)
                r.update(x_px=int(x0 + DX), y_px=int(y0 + DY), w_px=int(x1 - x0), h_px=int(y1 - y0),
                         w_mm=round((x1 - x0) / K, 2), h_mm=round((y1 - y0) / K, 2), runs_into_left_neighbour=left,
                         runs_into_right_neighbour=right, ink_px=int(m.sum()), stretch_u_mm=[a, b],
                         note='faint: little ink at half tone' if (y1 - y0) / K < 1.2 else '')
            out.append(r)
    meta = dict(image=f'{MESH}.png', px_um=8.64, px_per_mm=round(K, 3),
                numbering='line, then position in the line counted from the left, in reading orientation',
                reading={'line 1': 'τὸ̣ δι’ αὑτὸ α̣ἱ̣ρ̣ε̣τ̣όν', 'line 2': 'π̣ [',
                         'source': 'papyrologist (Nicola Bodill), 23 Oct 2026; dotted letters are uncertain'},
                method="bounding box of the main image's ink at half tone or darker inside the papyrologist's traced shape (23 Oct tracing) "
                       "widened 0.15 mm, cut to the letter's stretch along the line and to its line's rows; measurement only",
                letters=out)
    json.dump(meta, open(f'{PKG}/method/checks/letter_dimensions.json', 'w'), indent=1, ensure_ascii=False)
    rows = ['# Letter sizes in the claim image', '',
            "8.64 µm per pixel (115.74 px per mm). Each box is the claim image's ink, at half tone or darker, inside our papyrologist's "
            'traced letter (`supporting/reading/`). Letters are numbered from the left of each line. `letter_dimensions.json` has every letter of her '
            'reading.', '',
            '| line | no. | letter | box in the image: x, y (px) | width × height (px) | width × height (mm) | note |', '|---|---|---|---|---|---|---|']
    for r in out:
        if not r['measured'] or r['h_mm'] < 1.2: continue
        nb = [s for s, k in (('previous', r['runs_into_left_neighbour']), ('next', r['runs_into_right_neighbour'])) if k]
        note = f"touches the {' and '.join(nb)} letter; box cut between them" if nb else ''
        rows.append(f"| {r['line']} | {r['position_in_line']} | {r['letter']} | {r['x_px']}, {r['y_px']} | {r['w_px']} × {r['h_px']} | "
                    f"{r['w_mm']} × {r['h_mm']} | {note} |")
    open(f'{PKG}/method/checks/letter_dimensions.md', 'w').write('\n'.join(rows) + '\n')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
