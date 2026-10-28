"""grids_final.py : the evidence grids for the final version 7 package (comparisons/).  Same maps, frame, columns and grey rule as
evidence_grid_line1.py + evidence_grid_dn.py (noise-reduced rows first); only the layout changes: readable row labels, plain row
names (the main image named as such), and a third grid for line 2.
  line1_equal   : line 1 in 15 equal windows across the claim box (no reading used)
  line1_letters : line 1 in columns at the papyrologist's 23 Oct letters (layout only), labelled with them
  line2_equal   : line 2 in 15 equal windows across the claim box (no reading used); a strip above the columns marks where her
                  reading of line 2 begins, π (layout only)
Every map is shown in every column.  Writes the letter grid to PKG/, the other two to PKG/supporting/letter_grids/, and each
.json to PKG/method/checks/letter_grids/.  Usage: grids_final.py PKG [mode ...]"""
import os, sys, json, textwrap
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
COORD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'coord'); sys.path.insert(0, COORD)
import evidence_grid_line1 as G                                                    # noqa: E402
import evidence_grid_dn as DN                                                      # noqa: E402
N, L, E = G.N, G.L, G.E
PKG = sys.argv[1]
FB = '/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf'; FR = '/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf'
ROWS = [
    ('Noise-reduced models (new)', 'MAIN IMAGE: TRACE x A-hn', ('TxAhn',)),
    ('Noise-reduced models (new)', 'A-hn x sqrt(Reader V3)', ('AhnV3',)),
    ('Noise-reduced models (new)', 'C x sqrt(Reader V3)', ('CV3',)),
    ('Noise-reduced models (new)', 'B-hn x sqrt(Reader V3)', ('BhnV3',)),
    ('Noise-reduced models (new)', 'A-hn alone (A retrained on blank papyrus)', ('Ahn',)),
    ('Public models', 'TRACE v3B', ('tr',)), ('Public models', 'Reader V3', ('v3',)), ('Public models', 'Reader v2', ('rv2',)),
    ('Public models', 'v8-in (organisers\' base model)', ('v8',)), ('Public models', 'v8Q (v8-in tuned on PHerc1447 + 0343)', ('v8q',)),
    ('Public models', 'organisers\' PHerc1447 fine-tune (LOO)', ('loo',)),
    ('Our models', 'model A (mean of 3 seeds)', ('A',)), ('Our models', 'model A, seed 0', ('s0',)), ('Our models', 'model A, seed 1', ('s1',)),
    ('Our models', 'model A, seed 2', ('s2',)), ('Our models', 'model B', ('B',)), ('Our models', 'model B, seed 1', ('B1',)),
    ('Our models', 'variant: sc', ('sc',)), ('Our models', 'variant: no pseudo-labels', ('nopl',)),
    ('Our models', 'model C', ('C',)), ('Our models', 'model C without PHerc0139 labels', ('C2',)),
    ('Fixed whole-image rules', 'TRACE + v8-in (version 5 main image)', ('tr', 'v8')),
    ('Fixed whole-image rules', 'TRACE x A (before hard negatives)', ('TxA',)),
    ('Fixed whole-image rules', 'A x sqrt(TRACE)', ('softA',)), ('Fixed whole-image rules', 'TRACE x mean(A, B, C)', ('Tx3',)),
    ('Fixed whole-image rules', 'minimum of the two model families', ('min_fam',)),
]
BOX = [244, 554, 4642, 1660]                                                       # claim box in these pixels (x0, y0, x1, y1)
L2_LETTERS = [('π', 458, 720)]                                                     # where her line-2 reading starts, x stretch (layout only)


def build(mode, page):
    if mode == 'line1_letters':
        edges = np.array(G.LETTER_EDGES); y0, y1 = 520, 1180; labels = G.LETTER_LABELS
    else:
        edges = np.linspace(BOX[0], BOX[2], G.NWIN + 1).round().astype(int); labels = [str(i + 1) for i in range(G.NWIN)]
        if mode == 'line1_equal':
            ref = page(('TxA',))                                                   # line-1 band: darkest row band in the box's top half
            prof = (255 - ref[BOX[1]:(BOX[1] + BOX[3]) // 2, BOX[0]:BOX[2]].astype(np.float32)).mean(1)
            prof = np.convolve(prof, np.ones(61) / 61, 'same'); yc = BOX[1] + int(np.argmax(prof)); y0, y1 = yc - 300, yc + 260
        else:
            y0, y1 = 1180, 1740                                                    # line 2: below line 1's band, to 0.7 mm under the box
    nwin = len(edges) - 1; S = G.SCALE; LABW = 330
    cws = [int(round((edges[i + 1] - edges[i]) * S)) for i in range(nwin)]; ch = int(round((y1 - y0) * S))
    xs = [LABW + sum(cws[:i]) + 4 * i for i in range(nwin)]; W = xs[-1] + cws[-1] + 4
    big, small, lab, labb = (ImageFont.truetype(FB, 30), ImageFont.truetype(FR, 12), ImageFont.truetype(FR, 15), ImageFont.truetype(FB, 15))
    title = {'line1_equal': 'Line 1 in 15 equal windows across the claim box. No reading used.',
             'line1_letters': 'Line 1 in columns at our papyrologist\'s letters (her 23 Oct reading; used only to place the columns).',
             'line2_equal': 'Line 2 in 15 equal windows across the claim box. No reading used; the strip above marks where her reading of line 2 begins (π).'}[mode]
    hh = 112 if mode == 'line2_equal' else 86
    hp = Image.new('L', (W, hh), 255); dr = ImageDraw.Draw(hp)
    dr.text((8, 6), title, fill=0, font=labb)
    dr.text((8, 26), 'Every row is a whole map shown with the same grey rule (median to 99.9th percentile of the image; dark = ink). '
                     'Every map in every column. Numbers: u range in mm.', fill=60, font=small)
    for i in range(nwin):
        u0 = edges[i] / L.K + E.RX[0] / L.K + L.U0; u1 = edges[i + 1] / L.K + E.RX[0] / L.K + L.U0
        w = dr.textlength(labels[i], font=big); top = 44
        dr.text((xs[i] + (cws[i] - w) / 2, top), labels[i], fill=0 if mode == 'line1_letters' else 90, font=big)
        dr.text((xs[i] + 2, hh - 16), f'{u0:+.1f}..{u1:+.1f}', fill=0, font=small)
    if mode == 'line2_equal':
        dr.text((8, hh - 36), 'her reading:', fill=0, font=labb)
        for ch_, a, b in L2_LETTERS:                                               # mark each letter's stretch in grid x
            def gx(x):
                i = int(np.clip(np.searchsorted(edges, x, 'right') - 1, 0, nwin - 1))
                return xs[i] + (x - edges[i]) * S
            xa, xb = gx(a), gx(b)
            dr.line([(xa, hh - 22), (xb, hh - 22)], fill=0, width=3)
            w = dr.textlength(ch_, font=labb); dr.text(((xa + xb) / 2 - w / 2, hh - 42), ch_, fill=0, font=labb)
    rows, grp = [np.array(hp)], None
    for g, name, ms in ROWS:
        if g != grp:
            gb = Image.new('L', (W, 34), 255); ImageDraw.Draw(gb).text((8, 10), g.upper(), fill=70, font=ImageFont.truetype(FB, 17))
            rows.append(np.array(gb)); grp = g
        img = page(ms)
        r = Image.new('L', (W, ch), 255); d = ImageDraw.Draw(r)
        f = labb if name.startswith('MAIN') else lab
        lines = textwrap.wrap(name, 30)
        for k, t in enumerate(lines): d.text((8, ch // 2 - 10 * len(lines) + 20 * k), t, fill=0, font=f)
        r = np.array(r)
        for i in range(nwin):
            r[:, xs[i]:xs[i] + cws[i]] = cv2.resize(img[y0:y1, edges[i]:edges[i + 1]], (cws[i], ch), interpolation=cv2.INTER_AREA)
        if name.startswith('MAIN'): r[:, LABW - 8:LABW - 4] = 0                    # a bar marks the main image's row
        rows += [r, np.full((4, W), 255, np.uint8)]
    sheet = np.concatenate(rows, 0)
    name = {'line1_equal': 'evidence_grid_line1_equal_windows', 'line1_letters': 'evidence_grid_line1_at_papyrologist_letters',
            'line2_equal': 'evidence_grid_line2_equal_windows'}[mode]
    jdir = f'{PKG}/method/checks/letter_grids'; os.makedirs(jdir, exist_ok=True)
    idir = PKG if mode == 'line1_letters' else f'{PKG}/supporting/letter_grids'; os.makedirs(idir, exist_ok=True)
    cv2.imwrite(f'{idir}/{name}.jpg', sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    json.dump(dict(note=title, frame='papyrologist pack pixels (5278 px wide, top row = u -17.1..+28.5, v +7.5 mm; '
                                     'main image px = these + (1713, 1062))', band_y=[int(y0), int(y1)],
                   column_x_edges=[int(e) for e in edges], labels=list(labels),
                   line2_letters=[dict(letter=c, x=[a, b]) for c, a, b in L2_LETTERS] if mode == 'line2_equal' else None,
                   display='each map z-scored over the context, grey rule median -> 99.9th percentile of the shown area, dark = ink',
                   rows=[dict(group=g, label=n, maps={k: (os.path.basename(G.SRC[k]) if k in G.SRC else 'A x sqrt(TRACE)') for k in ms})
                         for g, n, ms in ROWS]),
              open(f'{jdir}/{name}.json', 'w'), indent=1, ensure_ascii=False)
    print(name, sheet.shape, flush=True)


def main():
    G.SRC.update(DN.derive())
    s = L.sl(L.CTX)
    ct = cv2.imread(f'{G.AM}/ct20.png', -1)[s]
    cx0, cy0 = L.px(L.CTX['u0'], L.CTX['v0'])
    sS = tuple(slice(a.start - o, a.stop - o) for a, o in zip(L.sl(L.SHOWN), (cy0, cx0)))
    sR = (slice(E.RY[0] - cy0, E.RY[1] - cy0), slice(E.RX[0] - cx0, E.RX[1] - cx0))
    cache, pages = {}, {}

    def get(k):
        if k not in cache:
            if k == 'softA':
                cache[k] = np.nan_to_num(get('A')) * np.sqrt(np.clip(np.nan_to_num(get('tr')), 0, 1))
                cache[k][~np.isfinite(get('A')) | ~np.isfinite(get('tr'))] = np.nan
            else:
                cache[k] = np.load(G.SRC[k], mmap_mode='r')[s].astype(np.float32)
        return cache[k]

    def page(ms):                                                                  # identical to evidence_grid_line1.page
        if ms not in pages:
            mm = {k: get(k) for k in ms}
            valid = (ct > 0) & np.all([np.isfinite(v) for v in mm.values()], 0)
            F = L.fused(mm, valid, 'z', ms)
            lev = (float(np.percentile(F[sS][valid[sS]], 50)), float(np.percentile(F[sS][valid[sS]], 99.9)))
            d = np.clip((F[sR] - lev[0]) / max(lev[1] - lev[0], 1e-9), 0, 1)
            pages[ms] = np.flipud(np.where(valid[sR], 236 - d * 216, 255).astype(np.uint8))
        return pages[ms]

    for mode in (sys.argv[2:] or ('line1_equal', 'line1_letters', 'line2_equal')): build(mode, page)


if __name__ == '__main__':
    main()
