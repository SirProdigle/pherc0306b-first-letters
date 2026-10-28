"""evidence_grid_line1.py : line 1 of lead A cut into equal windows (no letter positions, no reading), every lead-A map we have shown
in every window, so nothing is selected (owner, 23 Oct; for review, not for the claim).  Frame = the extended mask pack
(papyrologist_PHerc0306B_w074_masks_m306_ext_20261023): each map z-scored over the context, grey rule median -> p99.9 of the shown
area, dark = ink.  Writes CANDIDATES/quick_compare_m306_20261023/evidence_grid_line1.jpg (+ .json with windows and map sources)."""
import os, sys, json
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nicola_masks_m306 as N                                                      # noqa: E402
import nicola_masks_m306_ext as E                                                  # noqa: E402
L = N.L
M306 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AM = L.AM; SL = f'{M306}/self/leadA'; SM = f'{M306}/super/work_ctx/maps'; SC = f'{M306}/eval/maps/screen'
OUT = f'{N.W}/CANDIDATES/quick_compare_m306_20261023'
ROWS = [  # group, label, maps (z-mean of these; 'expr' builds one map)
    ('public models', 'TRACE v3B', ('tr',)), ('public models', 'Reader V3', ('v3',)), ('public models', 'Reader v2', ('rv2',)),
    ('public models', 'v8-in base', ('v8',)), ('public models', 'v8Q (1447+0343 fine-tune)', ('v8q',)),
    ('public models', 'organisers 1447 LOO fine-tune', ('loo',)),
    ('our models', 'model A (3-seed mean)', ('A',)), ('our models', 'A seed 0', ('s0',)), ('our models', 'A seed 1', ('s1',)),
    ('our models', 'A seed 2', ('s2',)), ('our models', 'model B', ('B',)), ('our models', 'B seed 1', ('B1',)),
    ('our models', 'SELF sc variant', ('sc',)), ('our models', 'SELF no-pseudo-label variant', ('nopl',)),
    ('our models', 'model C', ('C',)), ('our models', 'C without 0139 labels', ('C2',)),
    ('fixed whole-image rules', 'TRACE + v8-in (submitted v5)', ('tr', 'v8')), ('fixed whole-image rules', 'TRACE x A (v6 main image)', ('TxA',)),
    ('fixed whole-image rules', 'A x soft TRACE (sqrt)', ('softA',)), ('fixed whole-image rules', 'TRACE x mean(A,B,C)', ('Tx3',)),
    ('fixed whole-image rules', 'family minimum', ('min_fam',)),
]
SRC = {'tr': N.MAPS['tr'], 'v3': f'{AM}/fwd_v3.npy', 'rv2': f'{AM}/fwd_rv2.npy', 'v8': N.MAPS['v8'], 'v8q': f'{AM}/fwd_v8q.npy',
       'loo': f'{M306}/eval/maps/fwd_loo_PROVISIONAL.npy', 'A': N.MAPS['A'], 's0': f'{SL}/self_main_fwd_t1.npy',
       's1': f'{SL}/self_main_s1_fwd_t1.npy', 's2': f'{SL}/self_main_s2_fwd_t1.npy', 'B': N.MAPS['B'], 'B1': f'{SL}/self_343h_s1_fwd_t1.npy',
       'sc': f'{SL}/self_sc_fwd_t1.npy', 'nopl': f'{SL}/self_nopl_fwd_t1.npy', 'C': N.MAPS['C'], 'C2': f'{SC}/v8306_343_s0_fwd_t1.npy',
       'TxA': f'{SM}/fwd_TxA.npy', 'Tx3': f'{SM}/fwd_Tx3.npy', 'min_fam': f'{SM}/fwd_min_fam.npy', 'lift': f'{SM}/fwd_lift.npy',
       'hyst': f'{SM}/fwd_hyst.npy'}
NWIN, SCALE, LABW = 15, 0.45, 300
LETTERS = '--letters' in sys.argv
# column edges (x px in the mask-pack frame), layout only: first set at the narrow points between the letters of the papyrologist's
# 23 Oct tracing of line 1 (~/Downloads/NewSegment2_annotations.png, same pixels), then (23 Oct ~15:00, owner: "slightly off") moved to
# the nearest ink gap of the model images (mean darkness of TRACE x A, A, TRACE + A in rows 600-1250, 31 px smoothing, +-90 px);
# P|E and E|T stay at her positions (empty ink there; her typed labels, registered at r .97, agree), O|A and A|I set mid-gap by eye
LETTER_EDGES = [627, 883, 1128, 1519, 1645, 1829, 2259, 2490, 2750, 3060, 3270, 3470, 3705, 4024, 4270, 4580]
LETTER_LABELS = ['Τ', 'Ο', 'Δ', 'Ι', 'Α', 'Υ', 'Τ', 'Ο', 'Α', 'Ι', 'Ρ', 'Ε', 'Τ', 'Ο', 'Ν']


def main():
    s = L.sl(L.CTX)
    ct = cv2.imread(f'{AM}/ct20.png', -1)[s]
    cx0, cy0 = L.px(L.CTX['u0'], L.CTX['v0'])
    sS = tuple(slice(a.start - o, a.stop - o) for a, o in zip(L.sl(L.SHOWN), (cy0, cx0)))
    sR = (slice(E.RY[0] - cy0, E.RY[1] - cy0), slice(E.RX[0] - cx0, E.RX[1] - cx0))
    cache = {}

    def get(k):
        if k not in cache:
            if k == 'softA':
                cache[k] = np.nan_to_num(get('A')) * np.sqrt(np.clip(np.nan_to_num(get('tr')), 0, 1))
                cache[k][~np.isfinite(get('A')) | ~np.isfinite(get('tr'))] = np.nan
            else:
                cache[k] = np.load(SRC[k], mmap_mode='r')[s].astype(np.float32)
        return cache[k]

    def page(ms):
        mm = {k: get(k) for k in ms}
        valid = (ct > 0) & np.all([np.isfinite(v) for v in mm.values()], 0)
        F = L.fused(mm, valid, 'z', ms)
        lev = (float(np.percentile(F[sS][valid[sS]], 50)), float(np.percentile(F[sS][valid[sS]], 99.9)))
        d = np.clip((F[sR] - lev[0]) / max(lev[1] - lev[0], 1e-9), 0, 1)
        return np.flipud(np.where(valid[sR], 236 - d * 216, 255).astype(np.uint8))

    box = [244, 554, 4642, 1660]                                                   # claim box in these pixels (x0, y0, x1, y1)
    if LETTERS:                                                                    # columns at the papyrologist's letters (layout only)
        edges = np.array(LETTER_EDGES); y0, y1 = 520, 1180; labels = LETTER_LABELS
    else:
        ref = page(('TxA',))                                                       # line-1 band: the darkest row band in the box's top half
        prof = (255 - ref[box[1]:(box[1] + box[3]) // 2, box[0]:box[2]].astype(np.float32)).mean(1)
        prof = np.convolve(prof, np.ones(61) / 61, 'same'); yc = box[1] + int(np.argmax(prof))
        y0, y1 = yc - 300, yc + 260
        edges = np.linspace(box[0], box[2], NWIN + 1).round().astype(int); labels = [str(i + 1) for i in range(NWIN)]
    nwin = len(edges) - 1
    cws = [int(round((edges[i + 1] - edges[i]) * SCALE)) for i in range(nwin)]; ch = int(round((y1 - y0) * SCALE))
    xs = [LABW + sum(cws[:i]) + 4 * i for i in range(nwin)]
    head = np.full((70, xs[-1] + cws[-1] + 4), 255, np.uint8)
    from PIL import Image, ImageDraw, ImageFont
    hp = Image.fromarray(head); dr = ImageDraw.Draw(hp)
    big = ImageFont.truetype('/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf', 30)
    small = ImageFont.truetype('/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf', 11)
    for i in range(nwin):
        u0 = edges[i] / L.K + E.RX[0] / L.K + L.U0; u1 = edges[i + 1] / L.K + E.RX[0] / L.K + L.U0
        w = dr.textlength(labels[i], font=big)
        dr.text((xs[i] + (cws[i] - w) / 2, 4), labels[i], fill=0, font=big)
        dr.text((xs[i] + 2, 48), f'{u0:+.1f}..{u1:+.1f}', fill=0, font=small)
    head = np.array(hp)
    rows, grp = [head], None
    for g, lab, ms in ROWS:
        if g != grp:
            gb = np.full((40, head.shape[1]), 255, np.uint8)
            cv2.putText(gb, g.upper(), (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 90 if not g.startswith('FAIL') else 0, 2, cv2.LINE_AA)
            rows.append(gb); grp = g
        img = page(ms)
        r = np.full((ch, head.shape[1]), 255, np.uint8)
        cv2.putText(r, lab, (6, ch // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1, cv2.LINE_AA)
        for i in range(nwin):
            c = cv2.resize(img[y0:y1, edges[i]:edges[i + 1]], (cws[i], ch), interpolation=cv2.INTER_AREA)
            r[:, xs[i]:xs[i] + cws[i]] = c
        rows += [r, np.full((4, head.shape[1]), 255, np.uint8)]
        print(lab, flush=True)
    sheet = np.concatenate(rows, 0)
    name = 'evidence_grid_line1_letters' if LETTERS else 'evidence_grid_line1'
    cv2.imwrite(f'{OUT}/{name}.jpg', sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    note = ('line 1 cut at the papyrologist\'s 23 Oct letter positions (layout only: column edges at the narrow points between her traced '
            'letters; no model chosen or changed by it); every map shown in every column' if LETTERS else
            'line 1 cut into equal windows across the claim box (no letter positions or reading used); every map shown in every window')
    json.dump(dict(note=note, frame='papyrologist_PHerc0306B_w074_masks_m306_ext_20261023 pixels', band_y=[int(y0), int(y1)],
                   window_x_edges=[int(e) for e in edges], labels=labels,
                   rows=[dict(group=g, label=l, maps={k: os.path.relpath(SRC[k], N.W) if k in SRC else 'A x sqrt(TRACE)' for k in ms}) for g, l, ms in ROWS]),
              open(f'{OUT}/{name}.json', 'w'), indent=1, ensure_ascii=False)
    print('done', sheet.shape, flush=True)


if __name__ == '__main__':
    main()
