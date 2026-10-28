"""tri_rank.py MAPS_ROOT [--tag l1218cw] [--min 4] [--um 8.64] [--flat FLAT_DIR] : review queue for spiral-winding windows.

Calibration (x_gates/, 2026-10-11): text = PHerc0343 positive control through our own lane (pc343 w046/w047 rev x2600,
w048 rev x0/x1300), PHerc1447 w058 labelled text (rev x0/x1300), PHerc0841 w00 labelled crop; artefacts = windows rejected by
hand (1218 fold w065-w076, cavities w049/w050; 1545 cracks w038-w040, mounting material w094).

HARD gates (fail = auto-reject):
  1. row direction: row frequency angle within ROW_TOL of 90 deg (rows run along the winding).  0343 text 0-20 off, 1447 0;
     1218 fold 35 off.  (0841's 14 mm hand-segment crop is 25 off: only one window, slanted segment; noted, not used to tune.)
  2. flat sheet: a fibre family within FIB_TOL of 90 deg.  0343 text 4-6 off, 1447 1 off; 1218 fold 16-39 off.
  3. papyrus present: fibre-scale texture energy (stroke_feats.texture_energy) relative to the median over this lane's
     windows >= REL_TEX_MIN.  Absolute energy is scan-dependent (0343 papyrus 0.25-0.27, clean 1218 papyrus 0.14-0.25), so the
     lane itself is the reference.  0343 text 0.97-1.04 of its lane; 1218 windows >= 0.76; 1545 w094 (smooth mounting
     material, 0.128 absolute) ~0.5 of its lane.
SOFT gates (fail = second tier, REVIEW-2; real text has failed each of these once or nearly):
  4. crack enrichment (gap_gate) <= CRACK_MAX.  0343 text 0.01-0.22 but w048 letters 1-4 2.02; 1447 text 1.73-1.80; cracks 1.5-4.
  5. shape >= SHAPE_MIN: share of hot area in elongated marks (>= 1 mm, >= 2:1).  Text 0.246-0.75; 9/17 artefacts < 0.20.
  6. recur <= RECUR_MAX: along-winding hot profile correlation with the same 3-D spot on w+-2 (needs --flat).  0343 text
     0.14-0.25; 9/17 artefacts > 0.30.  Only 4 text windows, all 0343, so soft.
  (horiz, the share of long marks along the winding, did NOT separate: text up to 0.80, artefacts up to 0.67; not used.)
KNOWN BLIND SPOT: the row-score cut (--min, default 4) misses text with only two rows: 1447 w058 text scores -0.54/0.08 and
0841 -0.88 although both pass gates 1-3.  Use --min 0 to gate every window when a lane is small enough.
--glyph (x_textdet/, 2026-10-12): ALSO admit windows whose learned glyph score (fl/scripts/glyph_win.py -> glyph_win.json,
  x_glyph CNN trained on synthetic letters + real text of other scrolls) is >= --gmin, the p99 of 1276 negative lane windows
  (l1218cw + l1545acw, model gA: 9.06).  Leave-one-dataset-out check: 1447 w058 two-row text ranks 2nd of 1276 (row score:
  630th-990th), 0343 lane text 1st-17th, 0841 crop 7th.  Glyph-admitted windows still go through every gate below.
"""
import sys, os, glob, json, argparse
import numpy as np, zarr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gap_gate import gap_stats
from stroke_feats import shape_horiz, recur, texture_energy

ROW_TOL, FIB_TOL, REL_TEX_MIN = 20.0, 12.0, 0.65
CRACK_MAX, SHAPE_MIN, RECUR_MAX = 1.5, 0.20, 0.30
ap = argparse.ArgumentParser()
ap.add_argument('maps'); ap.add_argument('--tag', default=''); ap.add_argument('--min', type=float, default=4.0)
ap.add_argument('--um', type=float, default=8.64); ap.add_argument('--win', type=int, default=2600)
ap.add_argument('--flat', default='', help='flat/ dir with <wNNN>/tifxyz for the recur gate')
ap.add_argument('--no-crack', action='store_true'); ap.add_argument('--json')
ap.add_argument('--glyph', action='store_true', help='also admit windows with glyph_win.json score >= --gmin')
ap.add_argument('--gmin', type=float, default=9.06, help='glyph admission threshold (p99 of negative lane windows, model gA)')
a = ap.parse_args()
off90 = lambda ang: abs(((ang - 90) + 90) % 180 - 90)

cand = []
for f in sorted(glob.glob(os.path.join(a.maps, (a.tag + '_' if a.tag else '') + '*/spectral_win.json'))):
    d = os.path.dirname(f); n = os.path.basename(d)
    ren = os.path.join(os.path.dirname(os.path.dirname(d)), 'renders', n + '.zarr')
    gf = os.path.join(d, 'glyph_win.json')
    gw = json.load(open(gf)) if a.glyph and os.path.exists(gf) else {}
    gmap = {(sd, x): g for sd in ('fwd', 'rev') for x, g in gw.get(sd, [])}
    for al in json.load(open(f)).get('align') or []:
        side, x0, s, rang, fibs = al[0], al[1], al[2], al[3], al[4]
        g = gmap.get((side, x0))
        admit = (s is not None and s >= a.min) or (g is not None and g >= a.gmin)
        if s is None or not admit or rang is None or not isinstance(fibs, list): continue
        r = dict(score=round(s, 2), name=n, side=side, x0=x0, row_angle=rang, fibres=fibs, row_off=round(off90(rang), 1),
                 fib_off=round(min(off90(x) for x in fibs), 1) if fibs else None, tex=None, rel_tex=None, crack=None, shape=None, recur=None,
                 glyph=g, via='row' if s >= a.min else 'glyph')
        r['_d'], r['_ren'] = d, ren
        if os.path.exists(ren):
            ct = np.asarray(zarr.open(ren, mode='r')['0'][15][:, x0:x0 + a.win]).astype(np.float32)
            t = texture_energy(ct, ct > 0, a.um); r['tex'] = None if t is None else round(t, 3)
        cand.append(r)
texs = [r['tex'] for r in cand if r['tex'] is not None]
lane_tex = float(np.median(texs)) if len(texs) >= 5 else None      # too few windows: the papyrus gate is skipped
rows = []
for r in cand:
    d, ren, n, side, x0 = r.pop('_d'), r.pop('_ren'), r['name'], r['side'], r['x0']
    hard, soft = [], []
    if r['row_off'] > ROW_TOL: hard.append('rows not along winding')
    if r['fib_off'] is None or r['fib_off'] > FIB_TOL: hard.append('sheet not flat (fold/crossing)')
    if lane_tex and r['tex'] is not None:
        r['rel_tex'] = round(r['tex'] / lane_tex, 2)
        if r['rel_tex'] < REL_TEX_MIN: hard.append('no papyrus texture (mounting material/void)')
    zf = glob.glob(os.path.join(d, f"*_d0_{'reverse' if side == 'rev' else 'fwd'}_z.npy"))
    if not hard and os.path.exists(ren) and zf:
        z = np.load(zf[0]).astype(np.float32)
        if not a.no_crack:
            m = np.zeros(z.shape, bool); m[:, x0:x0 + a.win] = True
            try: r['crack'] = round(gap_stats(ren, z, d=0, um=a.um, mask=m)['enrich'], 2)
            except Exception as e: r['crack'] = f'err {e}'
            if not (isinstance(r['crack'], float) and r['crack'] <= CRACK_MAX): soft.append('on cracks')
        ct = np.asarray(zarr.open(ren, mode='r')['0'][15][:, x0:x0 + a.win]).astype(np.float32)
        sh = shape_horiz(z[:, x0:x0 + a.win], ct > 0, a.um)[0]; r['shape'] = None if sh is None else round(sh, 3)
        if sh is not None and sh < SHAPE_MIN: soft.append('round blobs, no strokes')
        if a.flat:
            tag, w = n.rsplit('_', 1)
            rc = recur(a.maps, tag, w, x0, a.flat)[0]; r['recur'] = rc
            if rc is not None and rc > RECUR_MAX: soft.append('repeats on w+-2 (structure)')
    elif not hard:
        soft.append('gates not run (no render/map)')
    r['queue'] = ('REVIEW' if not soft else 'REVIEW-2: ' + ', '.join(soft)) if not hard else 'auto-reject: ' + ', '.join(hard)
    rows.append(r)
rank = lambda q: 0 if q == 'REVIEW' else 1 if q.startswith('REVIEW-2') else 2
rows.sort(key=lambda r: (rank(r['queue']), r['queue'].count(',') if r['queue'].startswith('REVIEW-2') else 0, -r['score']))  # fewer soft fails first
for r in rows:
    print(f"{r['score']:6.2f} {r['name']:<22} {r['side']} x{r['x0']:<6} rows {r['row_off']:5.1f} fib {r['fib_off']} tex {r['rel_tex']} "
          f"crack {r['crack']} shape {r['shape']} recur {r['recur']}" + (f" glyph {r['glyph']} via {r['via']}" if a.glyph else '') + f"  -> {r['queue']}")
if a.json: json.dump(rows, open(a.json, 'w'), indent=1)
