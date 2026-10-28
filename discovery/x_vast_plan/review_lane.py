"""review_lane.py LANE_ID [--top 6] : local review queue for a finished cloud lane (x_vast_plan/results/<lane>/).

1. glyph_win.py on the pulled fused d0 maps (results/<lane>/maps/<tag>_wNNN/*_z.npy) -> glyph_win.json per winding
2. reading face per winding from the lane's flattened tifxyz + the smoothed umbilicus (x_recto rule: ink models read only
   when their layer order runs towards the scroll centre; the wrong face scored AUC .54-.64 vs .86-.87 on 0343)
3. candidates on the reading face: tri_rank REVIEW / REVIEW-2 windows (box-side triage, masked row score) plus every window
   with glyph >= GLYPH_P99 (9.06 = top 1 % of 1276 negative windows; 0343 lane text scores 10.8)
3b. line finder (fl/scripts/line_find.py, x_refine S1): single lines of >= 4 separate letter-sized marks along the fibres (the
   kind the owner's eye found on 0306B w050 / w074).  The lane's top LINE_TOP line candidates and every one scoring >= LINE_MIN
   are admitted into the tri_rank window that holds them (why 'line'; column 'line' = best line score in the window); they
   always get cards, centred on the line.  Output of the finder: results/<lane>/review/line_find.json.
4. cards (ink overlay | ink as a page | bare CT preview) for the top candidates -> results/<lane>/review/, plus review.json
"""
import sys, os, glob, json, argparse, subprocess
import numpy as np, cv2
cv2.setNumThreads(int(os.environ.get('OMP_NUM_THREADS', '4')))
W = '/home/liam/projects/vesuvius-work'; HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f'{W}/CANDIDATES/_tools'); from card import make_card, auto_roi   # noqa: E402
GLYPH_P99, WIN = 9.06, 2600

ap = argparse.ArgumentParser(); ap.add_argument('lane'); ap.add_argument('--top', type=int, default=6)
a = ap.parse_args()
R = f'{HERE}/results/{a.lane}'; res = f'{R}/results'; maps = f'{R}/maps'
env = dict(l.strip().split('=', 1) for l in open(f'{res}/lane.env') if '=' in l)
um = float(env['UM']); scroll = env['SCROLL']; tag = open(f'{res}/best_tag').read().strip()

# 1. glyph scores
if not glob.glob(f'{maps}/*/glyph_win.json'):
    subprocess.run([f'{W}/envs/vcpy/bin/python', f'{W}/fl/scripts/glyph_win.py', maps, '--tag', tag, '--um', str(um)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
glyph = {}
for f in glob.glob(f'{maps}/*/glyph_win.json'):
    n = os.path.basename(os.path.dirname(f)); j = json.load(open(f))
    for side in ('fwd', 'rev'):
        for x0, s in j.get(side, []): glyph[(n, side, x0)] = s

# 2. reading face per winding
src = open(f'{HERE}/ink_lane.py').read(); ns = {}
exec(src[src.index('def reading_face'):src.index('def dirs_for')], {'json': json, 'np': np}, ns)
umb = (glob.glob(f'{HERE}/state/umb_{scroll}_s75.json') or glob.glob(f'{HERE}/state/umb_{scroll}*'))[0]
face = {}
for d in sorted(glob.glob(f'{res}/flat/w[0-9][0-9][0-9]')):
    t = glob.glob(f'{d}/**/*.tifxyz', recursive=True) or [d]
    try:
        f, share = ns['reading_face'](t[0], umb); face[os.path.basename(d)] = ('fwd' if f == 'forward' else 'rev', round(share, 3))
    except Exception as e:
        face[os.path.basename(d)] = (None, repr(e)[:80])

# 3. candidates on the reading face
tri = {(r['name'], r['side'], r['x0']): r for r in json.load(open(f'{res}/tri_rank.json'))}
cands = {}
for k, r in tri.items():
    if r['queue'].startswith('REVIEW'): cands[k] = 'tri:' + ('R' if r['queue'] == 'REVIEW' else 'R2')
# glyph against this lane's own null (x_refine R2: the p99 of negatives ranges 8.15-10.3 between lanes, so a fixed 9.06 under- or
# over-admits; known text sits at z ~2.1, range 0.6-2.4, against a lane's reading-face windows).  Admit z >= GLYPH_Z or >= 9.06.
GLYPH_Z = 1.5
rf_vals = [s for (n, side, x0), s in glyph.items() if s == s and face.get(n.rsplit('_', 1)[-1], (None,))[0] in (side, None)]
g_med = float(np.median(rf_vals)) if rf_vals else 0.0
g_sd = float(1.4826 * np.median(np.abs(np.array(rf_vals) - g_med))) + 1e-6 if rf_vals else 1.0
gz = lambda s: (s - g_med) / g_sd
for k, s in glyph.items():
    if s == s and (s >= GLYPH_P99 or gz(s) >= GLYPH_Z): cands[k] = (cands.get(k, '') + ' glyph').strip()
# 3b. line finder (x_refine S1).  LINE_MIN 6.0: known text lines score 8.3-11.2; reviewed-null lanes give 0.22 (dev) / 1.0
#     (0306B test lanes) false lines per 100 cm2 at >= 6 (S1 report, post-hoc pooled-glyph scorer);
#     the top LINE_TOP of every lane are admitted whatever their score (recall first: the owner's eye is the final filter).
LINE_MIN, LINE_TOP = 6.0, 10
lf = f'{R}/review/line_find.json'
if not os.path.exists(lf):
    os.makedirs(os.path.dirname(lf), exist_ok=True)
    subprocess.run([f'{W}/envs/vcpy/bin/python', f'{W}/fl/scripts/line_find.py', '--lane', a.lane, '--glyph', 'gA', '--device', 'cpu', '--out', lf],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
line = {}
if os.path.exists(lf):
    allc = sorted(((c, m) for m in json.load(open(lf))['maps'] for c in m['candidates']), key=lambda t: -t[0]['score'])
    for i, (c, m) in enumerate(allc):
        if not (i < LINE_TOP or c['score'] >= LINE_MIN): break
        cx = float(np.mean([p[0] for p in c['poly']]))
        xs = sorted({x0 for (nn, sd, x0) in list(tri) + list(glyph) if nn == m['name'] and sd == m['side']}) or [int(max(0, round((cx - WIN / 2) / 1300)) * 1300)]
        k = (m['name'], m['side'], min(xs, key=lambda x0: abs(x0 + WIN / 2 - cx)))      # the existing window centred nearest the line
        if k not in line or c['score'] > line[k]['score']: line[k] = c
        if 'line' not in cands.get(k, ''): cands[k] = (cands.get(k, '') + ' line').strip()
rows = []
for (n, side, x0), why in cands.items():
    w = n.rsplit('_', 1)[-1]; rf = face.get(w, (None,))[0]
    if rf is not None and side != rf: continue                     # wrong face: models can't read there
    t = tri.get((n, side, x0), {})
    g = glyph.get((n, side, x0), float('nan'))
    rows.append(dict(name=n, side=side, x0=x0, why=why, glyph=round(g, 2), glyph_z=round(gz(g), 2) if g == g else None,
                     row=t.get('score'), queue=t.get('queue', 'not gated'), crack=t.get('crack'), shape=t.get('shape'),
                     crack_veto=isinstance(t.get('crack'), (int, float)) and t['crack'] >= 2.0,      # R2: crack >= 2 = 0/23 text bands
                     line=line[(n, side, x0)]['score'] if (n, side, x0) in line else None,
                     line_feats={kk: line[(n, side, x0)]['feats'].get(kk) for kk in ('n_slots', 'len_mm', 'contrast', 'glyph_p', 'crack')} if (n, side, x0) in line else None,
                     line_poly=line[(n, side, x0)]['poly'] if (n, side, x0) in line else None))
rank = lambda r: (r['crack_veto'], -(r['glyph_z'] if r['glyph_z'] is not None else -9) - 1.0 * ('tri:R' in r['why'] and 'R2' not in r['why'])
                  - 0.5 * max(0.0, r['line'] or 0.0), -(r['row'] or -9))
rows.sort(key=rank)

# 4. cards
out = f'{R}/review'; os.makedirs(out, exist_ok=True)
carded = rows[:a.top] + [r for r in rows[a.top:] if 'line' in r['why']][:LINE_TOP]       # line-finder candidates always get cards
for i, r in enumerate(carded):
    w = r['name'].rsplit('_', 1)[-1]
    zf = glob.glob(f"{maps}/{r['name']}/*_d0_{'reverse' if r['side'] == 'rev' else 'fwd'}_z.npy")
    cf = f'{res}/ct/{w}.jpg'
    if not zf or not os.path.exists(cf): r['card'] = None; continue
    z = np.load(zf[0]).astype(np.float32); ct = cv2.imread(cf, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    ct = cv2.resize(ct, (z.shape[1], z.shape[0])); ct[z == 0] = 0; v = ct > 0
    m = np.zeros_like(v); m[:, r['x0']:r['x0'] + WIN] = True
    if r.get('line_poly'):                                     # centre the card on the line
        P = np.array(r['line_poly']); L = np.linalg.norm(P[1] - P[0]); Hh = np.linalg.norm(P[3] - P[0])
        roi = (int(P[:, 1].mean()), int(P[:, 0].mean()), int(Hh / 2 + 1500 / um), int(L / 2 + 1000 / um))
    else:
        roi = auto_roi(ct, z, um, r_mm=5.0, valid=v & m)
    r['card'] = f"{out}/{i + 1:02d}_{r['name']}_{r['side']}_x{r['x0']}.jpg"
    make_card(ct, z, roi, um, f"{scroll} cloud lane {r['name']} {r['side']} (reading face) x{r['x0']}-{r['x0'] + WIN}",
              [f"why: {r['why']}; glyph {r['glyph']} = z {r['glyph_z']} vs this lane's reading face (median {g_med:.2f}, sd {g_sd:.2f}; known text z ~2.1); masked row {r['row']}",
               f"tri_rank: {r['queue']}; crack {r['crack']}; shape {r['shape']}. CT = lane preview (half size, upscaled)."]
              + ([f"line finder: score {r['line']} {r['line_feats']} (x_refine S1; >= {LINE_MIN} admits)"] if r.get('line') is not None else []),
              r['card'], valid=v)
json.dump(dict(lane=a.lane, tag=tag, faces=face, n_windows=len(tri), glyph_null=dict(median=round(g_med, 3), sd=round(g_sd, 3)), candidates=rows),
          open(f'{out}/review.json', 'w'), indent=1)
faces = [v[0] for v in face.values()]
print(f"{a.lane}: {len(tri)} windows; reading face {max(set(faces), key=faces.count) if faces else '?'}; "
      f"{len(rows)} reading-face candidates ({sum('line' in r['why'] for r in rows)} via the line finder); {len(carded)} cards in {out}")
for r in carded:
    print(f"  glyph {r['glyph']:>6} (z {r['glyph_z']}) row {r['row']} line {r['line']} {r['name']} {r['side']} x{r['x0']} [{r['why']}] {r['queue']}{'  CRACK VETO' if r['crack_veto'] else ''}")
