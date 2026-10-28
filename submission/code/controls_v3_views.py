"""views_v3.py [--previews-only] : lead A with Reader V3 under v2's display rules fixed in advance (version 3 package).

Uses copies of the v2 package code (code/: claimlib_v2.py, make_claim_image_v2.py with three V3 pairs added) on the same carrier raster,
CT layer and context as the v2 claim image (maps/: v2's fwd_rv2 / fwd_tr / fwd_v8q + our fwd_v3).  Pairs: v3 (Reader V3 alone),
v3t (V3 + TRACE v3B), v3q (V3 + v8Q), and v2's rv8q (Reader v2 + v8Q, the v2 main image) and rt (Reader v2 + TRACE).
Rules, unchanged from v2:
  threshold display (A1.1): L = rung x p99 of the box, full tone at p99, to p99.9; pale CT background; rung = primary (0.35) unless its
    blank strip (u -11.2..28.8, v 6..15 mm) is painted on more than 1 % of its px at its own levels, then secondary (0.5) (v2's A5 rule);
  grey display (A6): tone linear from the context median to its p99.9, no threshold; no CT (v2's background rule).
Controls at each view's own levels: blank strip (painted, half tone, p99 tone) and box; depth shuffle on the 4 cm2 box where the shuffled
maps exist (z with the original maps' context mean / std, toned with the original levels).
Big box option (team advice 18 Oct, larger window): u -16..+26 x v -12..+5 mm = 7.14 cm2, levels from the big box for the threshold
display; both boxes drawn (4 cm2 dashed blue, big box dashed orange); it stays clear of the blank strip (v >= 6).
Writes views/*.png (full size), views/*_prev.jpg, controls.json.  The papyrologist's tracing is not used here."""
import os, sys, json
import numpy as np, cv2
A = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, f'{A}/code')
import claimlib_v2 as L, make_claim_image_v2 as C                                   # noqa: E402

MAPS, CT = f'{A}/maps', f'{A}/maps/ct20.png'; SHUF = f'{A}/shuf_maps'; OUT = f'{A}/views'; os.makedirs(OUT, exist_ok=True)
BIG = dict(u0=-16.0, u1=26.0, v0=-12.0, v1=5.0)
PAIRS = ['v3', 'v3t', 'v3q', 'rv8q', 'rt']
STRIP = dict(u0=-11.2, u1=28.8, v0=6.0, v1=15.0)


def stats(d, m):
    x = d[m]
    return dict(px=int(m.sum()), painted=round(float((x > 0).mean()), 5), half_tone=round(float((x >= 0.5).mean()), 5),
                p99_tone=round(float(np.percentile(x, 99)), 4))


def render(pair, display, rung, bg, box_levels):
    """C.build with the threshold levels taken from box_levels (4 cm2 claim box or BIG); returns rgb (reading orientation), d, masks"""
    mods = C.PAIRS[pair]
    m, ct, valid, files = C.load(MAPS, CT, mods, L.CTX)
    U, V = L.grid_uv(L.CTX, L.UMX)
    inbox = L.box_mask(U, V, L.CBOX) & valid; inbig = L.box_mask(U, V, BIG) & valid; strip = L.box_mask(U, V, STRIP) & valid
    z = {k: L.zs(m[k], valid) * valid for k in m}
    fused = np.mean([z[k] for k in mods], 0)
    bgimg = L.light_ct_bg(ct, valid, valid) if bg == 'ct' else np.where(valid[..., None], L.PAGE[None, None], 60).astype(np.float32)
    if display == 'grey':
        lo, hi = float(np.percentile(fused[valid], 50)), float(np.percentile(fused[valid], 99.9))
        lv = dict(lo_median=lo, hi_p999=hi); d = np.where(valid, np.clip((fused - lo) / (hi - lo), 0, 1), 0).astype(np.float32)
    else:
        ref = inbox if box_levels == 'claim' else inbig
        lv = L.q4_levels(fused, ref, L.RUNGS[rung]); d = L.q4_tone(fused, lv, valid)
    rgb = np.ascontiguousarray(L.q4_rgb(d, bgimg, valid)[::-1])
    PP = L.PPMX
    def rect(b):
        return (int((b['u0'] - L.CTX['u0']) * PP), int((L.CTX['v1'] - b['v1']) * PP), int((b['u1'] - L.CTX['u0']) * PP), int((L.CTX['v1'] - b['v0']) * PP))
    x0, y0, x1, y1 = rect(L.CBOX); L.dashed_rect(rgb, x0, y0, x1, y1, (30, 90, 220))
    if box_levels == 'big':
        x0, y0, x1, y1 = rect(BIG); L.dashed_rect(rgb, x0, y0, x1, y1, (230, 120, 20))
    return rgb, d, dict(valid=valid, inbox=inbox, inbig=inbig, strip=strip, fused=fused, m=m, levels=lv)


def shuffle_stats(pair, x, display):
    mods = C.PAIRS[pair]
    if not all(os.path.exists(f'{SHUF}/fwd_{k}.npy') for k in mods): return None
    m, valid, lv = x['m'], x['valid'], x['levels']
    st = {k: (float(m[k][valid].astype(np.float64).mean()), float(m[k][valid].astype(np.float64).std())) for k in mods}
    ms = {k: L.sample_raster(L.load_raster_map(f'{SHUF}/fwd_{k}.npy', (4900, 7780)), L.CTX, L.UMX) for k in mods}
    vb = x['inbox'] & np.logical_and.reduce([ms[k] != 0 for k in mods])
    f0 = np.mean([(m[k] - st[k][0]) / (st[k][1] + 1e-9) for k in mods], 0)
    f1 = np.mean([(ms[k] - st[k][0]) / (st[k][1] + 1e-9) for k in mods], 0)
    if display == 'grey':
        tone = lambda f: np.where(vb, np.clip((f - lv['lo_median']) / (lv['hi_p999'] - lv['lo_median']), 0, 1), 0)  # noqa: E731
    else:
        tone = lambda f: L.q4_tone(f, dict(L=lv['L'], H=lv['H'], P=lv['P']), vb)                                    # noqa: E731
    d0, d1 = tone(f0 * x['valid']), tone(f1)
    rd = lambda d: dict(painted=round(float((d[vb] > 0).mean()), 5), half_tone=round(float((d[vb] >= 0.5).mean()), 5),  # noqa: E731
                        p99_tone=round(float(np.percentile(d[vb], 99)), 4))
    return dict(px=int(vb.sum()), original=rd(d0), shuffled=rd(d1), fused_z_r=round(float(np.corrcoef(f0[vb], f1[vb])[0, 1]), 4))


def save(name, rgb):
    cv2.imwrite(f'{OUT}/{name}.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 6])
    p = cv2.resize(rgb, (rgb.shape[1] // 3, rgb.shape[0] // 3), interpolation=cv2.INTER_AREA)
    cv2.imwrite(f'{OUT}/{name}_prev.jpg', cv2.cvtColor(p, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    res = dict(big_box_mm=BIG, strip_mm=STRIP, claim_box_mm=L.CBOX, rule='v2 display rules (PREREG A1.1, A5 rung rule, A6 grey), unchanged')
    U, V = L.grid_uv(L.CTX, L.UMX)
    res['big_box_cm2_flat'] = round((BIG['u1'] - BIG['u0']) * (BIG['v1'] - BIG['v0']) / 100, 3)
    for pair in PAIRS:
        mods = C.PAIRS[pair]
        if not all(os.path.exists(f'{MAPS}/fwd_{k}.npy') for k in mods): print('skip, maps missing:', pair); continue
        r = {}
        for box_levels in ('claim', 'big'):
            rungs = {}
            for rung in ('primary', 'secondary'):
                rgb, d, x = render(pair, 'threshold', rung, 'ct', box_levels)
                rungs[rung] = dict(levels={k: round(v, 3) for k, v in x['levels'].items()}, strip=stats(d, x['strip']),
                                   box4=stats(d, x['inbox']), big=stats(d, x['inbig']), _rgb=rgb, _x=x, _d=d)
            chosen = 'primary' if rungs['primary']['strip']['painted'] <= 0.01 else 'secondary'
            t = rungs[chosen]
            name = f'{pair}_threshold_{chosen}' + ('' if box_levels == 'claim' else '_BIG7cm2')
            save(name, t['_rgb'])
            entry = dict(rung_rule={k: rungs[k]['strip']['painted'] for k in rungs}, chosen=chosen, file=f'views/{name}.png',
                         levels=t['levels'], strip=t['strip'], box4=t['box4'], big=t['big'],
                         blank_bar_1pct=bool(t['strip']['painted'] <= 0.01))
            if box_levels == 'claim': entry['shuffle_box4'] = shuffle_stats(pair, t['_x'], 'threshold')
            r[f'threshold_{box_levels}'] = entry
        rgb, d, x = render(pair, 'grey', None, 'page', 'claim'); name = f'{pair}_grey_page'; save(name, rgb)
        r['grey'] = dict(file=f'views/{name}.png', levels={k: round(v, 3) for k, v in x['levels'].items()}, strip=stats(d, x['strip']),
                         box4=stats(d, x['inbox']), big=stats(d, x['inbig']), shuffle_box4=shuffle_stats(pair, x, 'grey'))
        rgb, d, x = render(pair, 'grey', None, 'page', 'big'); save(f'{pair}_grey_page_BIG7cm2', rgb)
        res[pair] = r
        print(pair, json.dumps({k: (v.get('chosen'), v['strip'], v['box4']) for k, v in r.items() if k.startswith('threshold')}), flush=True)
    json.dump(res, open(f'{A}/controls.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
