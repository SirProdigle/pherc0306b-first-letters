"""controls_v5.py : every control of the package from one build config (version 5).

One module for what versions 3-4c ran as three scripts, with their arithmetic unchanged:
  views_controls()    = controls_v3_views.py (-> evidence/controls_<tag>_views_same_levels.json, and the 7.14 cm2 context view):
                        per pair, the threshold rule at the claim box's and at the big box's levels (rung by the A5 blank rule), the
                        grey rule; blank strip, 4 cm2 box and big box read-outs; depth shuffle on the box where shuffled maps exist.
  variant_controls()  = variant_controls.py (-> evidence/blank_strip_variants_<tag>.json): version 2's display variants at their own
                        levels, strip and box, and the grey views' background rule (CT vs plain page).
  shuffle_variants()  = shuffle_control.py score_variants (-> evidence/depth_shuffle_variants_<tag>.json): every registry view at its
                        own levels on the depth-shuffled maps.
  blank_views()       NEW in version 5 (-> evidence/blank_control_views_<tag>.json): every registry view on the pre-registered blank
                        control and on every reported region (BLANK_CONTROL_PREREG.md), the A5 rung verdict on the control for
                        threshold views, the background rule's numbers for grey views.
  resolve_rungs()     the A5 rule for registry views whose rung is 'auto' (build_v5.py step rungs).
The 'strip' of the first three is the config's blank control, so with the version 4c config (control = I1's strip) their json is
version 4c's byte for byte.  Maps are cached per pair; every number is computed with the claimlib_v2 / make_claim_image_v2
functions the views use."""
import os, sys, json
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claimlib_v2 as L, make_claim_image_v2 as C                                   # noqa: E402

LUM = np.array([0.299, 0.587, 0.114], np.float32)
RAST = (4900, 7780)


class Ctx:
    """per-build state: maps dir, CT, shuffled maps dir, regions, and a cache of each pair's sampled maps and fused z"""
    def __init__(self, cfg, maps, ct, shuf=''):
        self.cfg, self.maps, self.ct, self.shuf = cfg, maps, ct, shuf
        C.configure(cfg)
        self.U, self.V = L.grid_uv(L.CTX, L.UMX)
        self.big = cfg.get('big_box', dict(u0=-16.0, u1=26.0, v0=-12.0, v1=5.0))
        self.control = {k: float(cfg['blank_control'][k]) for k in ('u0', 'u1', 'v0', 'v1')}
        self._pairs = {}

    def pair(self, pair):
        if pair not in self._pairs:
            mods = C.PAIRS[pair]
            m, ct, valid, files = C.load(self.maps, self.ct, mods, L.CTX)
            z = {k: L.zs(m[k], valid) * valid for k in m}
            fused = np.mean([z[k] for k in mods], 0)
            self._pairs = {pair: dict(mods=mods, m=m, ct=ct, valid=valid, files=files, fused=fused,
                                      inbox=L.box_mask(self.U, self.V, L.CBOX) & valid,
                                      inbig=L.box_mask(self.U, self.V, self.big) & valid)}   # keep one pair in memory
        return self._pairs[pair]

    def tone(self, p, display, rung='primary', ref='claim'):
        """(d, levels) exactly as make_claim_image_v2.build (and controls_v3_views.render for ref 'big')"""
        if display == 'grey':
            lo, hi = float(np.percentile(p['fused'][p['valid']], 50)), float(np.percentile(p['fused'][p['valid']], 99.9))
            return np.where(p['valid'], np.clip((p['fused'] - lo) / (hi - lo), 0, 1), 0).astype(np.float32), dict(lo_median=lo, hi_p999=hi)
        lv = L.q4_levels(p['fused'], p['inbox'] if ref == 'claim' else p['inbig'], L.RUNGS[rung])
        return L.q4_tone(p['fused'], lv, p['valid']), lv

    def region(self, b, p):
        return L.box_mask(self.U, self.V, b) & p['valid']


def stats3(d, m):                                                                   # controls_v3_views.stats
    x = d[m]
    return dict(px=int(m.sum()), painted=round(float((x > 0).mean()), 5), half_tone=round(float((x >= 0.5).mean()), 5),
                p99_tone=round(float(np.percentile(x, 99)), 4))


def stats4(d, m):                                                                   # variant_controls.stats
    x = d[m]
    return dict(px=int(m.sum()), painted=round(float((x > 0).mean()), 5), half_tone=round(float((x >= 0.5).mean()), 5),
                p99_tone=round(float(np.percentile(x, 99)), 4), p999_tone=round(float(np.percentile(x, 99.9)), 4))


def rgb_view(cx, p, d, bg, boxes):
    """controls_v3_views.render's image: reading orientation, the claim box (blue) and optionally the big box (orange) dashed"""
    bgimg = L.light_ct_bg(p['ct'], p['valid'], p['valid']) if bg == 'ct' else np.where(p['valid'][..., None], L.PAGE[None, None], 60).astype(np.float32)
    rgb = np.ascontiguousarray(L.q4_rgb(d, bgimg, p['valid'])[::-1]); PP = L.PPMX
    def rect(b):
        return (int((b['u0'] - L.CTX['u0']) * PP), int((L.CTX['v1'] - b['v1']) * PP), int((b['u1'] - L.CTX['u0']) * PP), int((L.CTX['v1'] - b['v0']) * PP))
    x0, y0, x1, y1 = rect(L.CBOX); L.dashed_rect(rgb, x0, y0, x1, y1, (30, 90, 220))
    if 'big' in boxes:
        x0, y0, x1, y1 = rect(cx.big); L.dashed_rect(rgb, x0, y0, x1, y1, (230, 120, 20))
    return rgb


def save_png(fn, rgb):
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    cv2.imwrite(fn, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 6])


def shuffle_stats(cx, pair, p, d_levels, display):                                  # controls_v3_views.shuffle_stats
    mods = p['mods']
    if not cx.shuf or not all(os.path.exists(f'{cx.shuf}/fwd_{k}.npy') for k in mods): return None
    m, valid, lv = p['m'], p['valid'], d_levels
    st = {k: (float(m[k][valid].astype(np.float64).mean()), float(m[k][valid].astype(np.float64).std())) for k in mods}
    ms = {k: L.sample_raster(L.load_raster_map(f'{cx.shuf}/fwd_{k}.npy', RAST), L.CTX, L.UMX) for k in mods}
    vb = p['inbox'] & np.logical_and.reduce([ms[k] != 0 for k in mods])
    f0 = np.mean([(m[k] - st[k][0]) / (st[k][1] + 1e-9) for k in mods], 0)
    f1 = np.mean([(ms[k] - st[k][0]) / (st[k][1] + 1e-9) for k in mods], 0)
    if display == 'grey':
        tone = lambda f: np.where(vb, np.clip((f - lv['lo_median']) / (lv['hi_p999'] - lv['lo_median']), 0, 1), 0)  # noqa: E731
    else:
        tone = lambda f: L.q4_tone(f, dict(L=lv['L'], H=lv['H'], P=lv['P']), vb)                                    # noqa: E731
    d0, d1 = tone(f0 * valid), tone(f1)
    rd = lambda d: dict(painted=round(float((d[vb] > 0).mean()), 5), half_tone=round(float((d[vb] >= 0.5).mean()), 5),  # noqa: E731
                        p99_tone=round(float(np.percentile(d[vb], 99)), 4))
    return dict(px=int(vb.sum()), original=rd(d0), shuffled=rd(d1), fused_z_r=round(float(np.corrcoef(f0[vb], f1[vb])[0, 1]), 4))


def views_controls(cx, out_json, context=None, views_dir=''):
    """controls_v3_views.main with the config's pairs and blank control; context = dict(pair, file) -> that pair's grey view with
    both boxes (the package's 7.14 cm2 context view); views_dir: also write every control image there (version 4c wrote them all)"""
    cfg = cx.cfg; pairs = cfg.get('controls', {}).get('pairs', ['v3', 'v3t', 'v3q', 'rv8q', 'rt']); BIG = cx.big; STRIP = cx.control
    res = dict(big_box_mm=BIG, strip_mm=STRIP, claim_box_mm=L.CBOX, rule='v2 display rules (PREREG A1.1, A5 rung rule, A6 grey), unchanged')
    res['big_box_cm2_flat'] = round((BIG['u1'] - BIG['u0']) * (BIG['v1'] - BIG['v0']) / 100, 3)
    for pair in pairs:
        if not all(os.path.exists(f'{cx.maps}/fwd_{k}.npy') for k in C.PAIRS[pair]): print('skip, maps missing:', pair); continue
        p = cx.pair(pair); strip = cx.region(STRIP, p); r = {}
        for box_levels in ('claim', 'big'):
            rungs = {}
            for rung in ('primary', 'secondary'):
                d, lv = cx.tone(p, 'threshold', rung, box_levels)
                rungs[rung] = dict(levels={k: round(v, 3) for k, v in lv.items()}, strip=stats3(d, strip), box4=stats3(d, p['inbox']),
                                   big=stats3(d, p['inbig']), _d=d, _lv=lv)
            chosen = 'primary' if rungs['primary']['strip']['painted'] <= 0.01 else 'secondary'
            t = rungs[chosen]
            name = f'{pair}_threshold_{chosen}' + ('' if box_levels == 'claim' else '_BIG7cm2')
            if views_dir: save_png(f'{views_dir}/{name}.png', rgb_view(cx, p, t['_d'], 'ct', ('claim',) if box_levels == 'claim' else ('claim', 'big')))
            entry = dict(rung_rule={k: rungs[k]['strip']['painted'] for k in rungs}, chosen=chosen, file=f'views/{name}.png',
                         levels=t['levels'], strip=t['strip'], box4=t['box4'], big=t['big'], blank_bar_1pct=bool(t['strip']['painted'] <= 0.01))
            if box_levels == 'claim': entry['shuffle_box4'] = shuffle_stats(cx, pair, p, t['_lv'], 'threshold')
            r[f'threshold_{box_levels}'] = entry
        d, lv = cx.tone(p, 'grey'); name = f'{pair}_grey_page'
        if views_dir: save_png(f'{views_dir}/{name}.png', rgb_view(cx, p, d, 'page', ('claim',)))
        r['grey'] = dict(file=f'views/{name}.png', levels={k: round(v, 3) for k, v in lv.items()}, strip=stats3(d, strip),
                         box4=stats3(d, p['inbox']), big=stats3(d, p['inbig']), shuffle_box4=shuffle_stats(cx, pair, p, lv, 'grey'))
        if views_dir or (context and context['pair'] == pair):
            rgb = rgb_view(cx, p, d, 'page', ('claim', 'big'))
            if views_dir: save_png(f'{views_dir}/{pair}_grey_page_BIG7cm2.png', rgb)
            if context and context['pair'] == pair: save_png(context['out'], rgb)
        res[pair] = r
        print(pair, json.dumps({k: (v.get('chosen'), v['strip'], v['box4']) for k, v in r.items() if k.startswith('threshold')}), flush=True)
    json.dump(res, open(out_json, 'w'), indent=1)
    return res


def variant_controls(cx, out_json):
    """variant_controls.main with the config's variant list and blank control"""
    cfg = cx.cfg
    VARIANTS = [tuple(v) for v in cfg.get('controls', {}).get('variant_list', [
        ['rt', 'threshold', 'primary'], ['rt', 'threshold', 'secondary'], ['rv8', 'threshold', 'primary'], ['rv8', 'threshold', 'secondary'],
        ['rv8q', 'threshold', 'primary'], ['rv8q', 'threshold', 'secondary'], ['rt', 'grey', None], ['rv8q', 'grey', None]])]
    S = cx.control
    res = dict(strip_mm=dict(S), box_mm=L.CBOX)
    for pair, disp, rung in VARIANTS:
        if not all(os.path.exists(f'{cx.maps}/fwd_{k}.npy') for k in C.PAIRS[pair]): print('skip, maps missing:', pair); continue
        key = f'{pair}_{disp}' + (f'_{rung}' if rung else '')
        p = cx.pair(pair); d, lv = cx.tone(p, disp, rung or 'primary'); v, inbox = p['valid'], p['inbox']
        strip = L.box_mask(cx.U, cx.V, S)
        r = dict(levels={k: round(x, 3) for k, x in lv.items()}, strip=stats4(d, strip & v), box=stats4(d, inbox))
        if disp == 'grey': r.update(bg_rule(p, d, strip & v, inbox))
        res[key] = r
        print(key, json.dumps(r), flush=True)
    json.dump(res, open(out_json, 'w'), indent=1)
    return res


def bg_rule(p, d, strip, inbox):
    """I2's background rule for a grey view: keep the CT layer iff the share of the blank region displayed >= 50 % dark with CT is
    <= 1.25 x the share without it (variant_controls.py)"""
    r = {}; lp = float(L.PAGE @ LUM); v = p['valid']
    for bgm in ('ct', 'page'):
        bg = L.light_ct_bg(p['ct'], v, v) if bgm == 'ct' else np.broadcast_to(L.PAGE[None, None], d.shape + (3,))
        pix = bg * (1 - d[..., None]) + L.INK[None, None] * d[..., None]
        dark = 1 - (pix @ LUM) / lp
        r[f'displayed_dark_ge50_{bgm}'] = dict(strip=round(float((dark[strip] >= 0.5).mean()), 5), box=round(float((dark[inbox] >= 0.5).mean()), 5),
                                               strip_p99=round(float(np.percentile(dark[strip], 99)), 4))
    a, b = r['displayed_dark_ge50_ct']['strip'], r['displayed_dark_ge50_page']['strip']
    r['bg_decision'] = dict(rule='keep CT iff strip share (displayed darkness >= 0.5) with CT <= 1.25 x without CT', ratio=round(a / max(b, 1e-9), 3),
                            keep_ct=bool(a <= 1.25 * b))
    return r


def boxpx(b):
    """render px (y0, y1, x0, x1) whose centres lie in the box (run.sh BOXPX, shuffle_control.BOXPX)"""
    K = L.K
    return (int(np.ceil((b['v0'] + 21) * K - 0.5)), int(np.floor((b['v1'] + 21) * K - 0.5)) + 1,
            int(np.ceil((b['u0'] + 36) * K - 0.5)), int(np.floor((b['u1'] + 36) * K - 0.5)) + 1)


def shuffle_variants(cx, views, out_json):
    """shuffle_control.score_variants over the registry views (list of dicts with key, view, pair, display, rung, bg)"""
    box = L.box_mask(cx.U, cx.V, L.CBOX); res = dict(box_px=boxpx(L.CBOX), shuffle='layers 8..31 permuted, seed 1 (prep)'); missing = []
    for v in views:
        key = v['key']; mods = C.PAIRS[v['pair']]
        if not cx.shuf or not all(os.path.exists(f'{cx.shuf}/fwd_{k}.npy') for k in mods): missing.append(key); continue
        p = cx.pair(v['pair']); m, valid = p['m'], p['valid']
        d0, lv_exact = cx.tone(p, v['display'], v['rung']); lv = {k: round(x, 3) for k, x in lv_exact.items()}
        st = {k: (float(m[k][valid].astype(np.float64).mean()), float(m[k][valid].astype(np.float64).std())) for k in mods}
        ms = {k: L.sample_raster(L.load_raster_map(f'{cx.shuf}/fwd_{k}.npy', RAST), L.CTX, L.UMX) for k in mods}
        vb = valid & box & np.logical_and.reduce([ms[k] != 0 for k in mods])
        f1 = np.mean([(ms[k] - st[k][0]) / (st[k][1] + 1e-9) for k in mods], 0)
        if v['display'] == 'grey':
            d1 = np.where(vb, np.clip((f1 - lv['lo_median']) / (lv['hi_p999'] - lv['lo_median']), 0, 1), 0)
        else:
            d1 = L.q4_tone(f1, dict(L=lv['L'], H=lv['H'], P=lv['P']), vb)
        rd = lambda d: dict(painted=round(float((d[vb] > 0).mean()), 5), half_tone=round(float((d[vb] >= 0.5).mean()), 5),   # noqa: E731
                            p99_tone=round(float(np.percentile(d[vb], 99)), 4))
        res[key] = dict(view=v['view'], display=v['display'], levels=lv, original=rd(d0), shuffled=rd(d1),
                        fused_z_r=round(float(np.corrcoef(np.mean([(m[k] - st[k][0]) / (st[k][1] + 1e-9) for k in mods], 0)[vb], f1[vb])[0, 1]), 4))
        print(key, json.dumps(res[key]), flush=True)
    for a, b in (('rv8q_secondary', 'rt_secondary'), ('rv8q_grey', 'rt_grey')):
        if a in res and b in res:
            x_, y_ = res[a]['shuffled']['half_tone'], res[b]['shuffled']['half_tone']
            res[f'q4_test_{a}'] = dict(vs=b, metric='shuffled half-tone share in the box', ratio=round(x_ / max(y_, 1e-9), 3) if y_ > 0 else None,
                                       bar=1.25, passes=bool(x_ <= 1.25 * y_ or x_ <= 1e-4))
    json.dump(res, open(out_json, 'w'), indent=1)
    return res, missing


def a5(cx, p, region):
    """the A5 rung rule on a region: primary unless the primary rung paints > 1 % of it at the view's own (box) levels"""
    d, lv = cx.tone(p, 'threshold', 'primary'); s = stats3(d, cx.region(region, p))
    return dict(primary_painted=s['painted'], primary_half_tone=s['half_tone'], rung='primary' if s['painted'] <= 0.01 else 'secondary',
                levels_primary={k: round(x, 3) for k, x in lv.items()})


def resolve_rungs(cx, views):
    """rung 'auto' -> the A5 rule on the pre-registered blank control; returns {key: decision}"""
    out = {}
    for v in views:
        if v['display'] != 'threshold' or v['rung'] != 'auto': continue
        p = cx.pair(v['pair']); r = a5(cx, p, cx.control); out[v['key']] = dict(r, control=cx.control)
        print('rung', v['key'], json.dumps(out[v['key']]), flush=True)
    return out


def blank_views(cx, views, out_json):
    """every registry view on the blank control and on each reported region; A5 verdicts; the grey views' background rule"""
    regions = dict(control=cx.control); regions.update({k: {x: float(b[x]) for x in ('u0', 'u1', 'v0', 'v1')} for k, b in cx.cfg.get('blank_reported', {}).items()})
    res = dict(control=dict(cx.cfg['blank_control']), reported_regions={k: b for k, b in regions.items() if k != 'control'},
               claim_box_mm=L.CBOX, note='painted = d > 0; half_tone = d >= 0.5; p99_tone = 99th percentile of d; each view at its own levels',
               views={})
    for v in views:
        p = cx.pair(v['pair']); d, lv = cx.tone(p, v['display'], v['rung'])
        r = dict(view=v['view'], pair=v['pair'], display=v['display'], rung=v['rung'] if v['display'] == 'threshold' else None,
                 levels={k: round(x, 3) for k, x in lv.items()}, box=stats4(d, p['inbox']),
                 regions={k: stats4(d, cx.region(b, p)) for k, b in regions.items()})
        if v['display'] == 'threshold':
            r['a5_on_control'] = a5(cx, p, cx.control)
            r['a5_on_reported'] = {k: a5(cx, p, b)['rung'] for k, b in regions.items() if k != 'control'}
        else:
            r['bg_rule_on_control'] = bg_rule(p, d, cx.region(cx.control, p), p['inbox'])
        res['views'][v['key']] = r
        print('blank', v['key'], json.dumps({k: r['regions'][k]['half_tone' if v['display'] == 'grey' else 'painted'] for k in regions}), flush=True)
    json.dump(res, open(out_json, 'w'), indent=1, ensure_ascii=False)
    return res
