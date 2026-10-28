"""variant_controls.py MAPS CT OUT.json : blank strip and box read-outs of every display variant at its OWN levels (lane I2).
Label-free.  Strip = I1's blank strip of the same context, u -11.2..+28.8 x v +6..+15 mm (claim frame, as rendered); box = the 4 cm2 box.
Per variant: painted share (d > 0), half-tone share (d >= 0.5), p99 of d, in the strip and in the box; for the grey variant also the
DISPLAYED darkness (1 - luminance / luminance(PAGE)) >= 0.5 share with the pale CT layer and without it (ink-as-page), which decides
its background by the rule written in PROGRESS before this ran: keep CT iff share_ct <= 1.25 x share_page on the strip."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claimlib_v2 as L, make_claim_image_v2 as C                                   # noqa: E402

VARIANTS = [('rt', 'threshold', 'primary'), ('rt', 'threshold', 'secondary'), ('rv8', 'threshold', 'primary'), ('rv8', 'threshold', 'secondary'),
            ('rv8q', 'threshold', 'primary'), ('rv8q', 'threshold', 'secondary'), ('rt', 'grey', None), ('rv8q', 'grey', None)]
LUM = np.array([0.299, 0.587, 0.114], np.float32)


def stats(d, m):
    x = d[m]
    return dict(px=int(m.sum()), painted=round(float((x > 0).mean()), 5), half_tone=round(float((x >= 0.5).mean()), 5),
                p99_tone=round(float(np.percentile(x, 99)), 4), p999_tone=round(float(np.percentile(x, 99.9)), 4))


def main():
    maps, ct, out = sys.argv[1:4]
    U, V = L.grid_uv(L.CTX, L.UMX); strip = (U >= -11.2) & (U <= 28.8) & (V >= 6) & (V <= 15)
    res = dict(strip_mm=dict(u0=-11.2, u1=28.8, v0=6.0, v1=15.0), box_mm=L.CBOX)
    for pair, disp, rung in VARIANTS:
        key = f'{pair}_{disp}' + (f'_{rung}' if rung else '')
        _, info, x = C.build(maps, ct, pair, rung or 'primary', lambda lv: [''], lambda lv: [''], display=disp, bg_mode='page')
        d, v, inbox = x['d'], x['valid'], x['inbox']
        r = dict(levels=info['levels'], strip=stats(d, strip & v), box=stats(d, inbox))
        if disp == 'grey':
            m2 = C.load(maps, ct, C.PAIRS[pair], L.CTX); ctg = m2[1]
            lp = float(L.PAGE @ LUM)
            for bgm in ('ct', 'page'):
                bg = L.light_ct_bg(ctg, v, v) if bgm == 'ct' else np.broadcast_to(L.PAGE[None, None], d.shape + (3,))
                pix = bg * (1 - d[..., None]) + L.INK[None, None] * d[..., None]
                dark = 1 - (pix @ LUM) / lp
                r[f'displayed_dark_ge50_{bgm}'] = dict(strip=round(float((dark[strip & v] >= 0.5).mean()), 5), box=round(float((dark[inbox] >= 0.5).mean()), 5),
                                                       strip_p99=round(float(np.percentile(dark[strip & v], 99)), 4))
            a, b = r['displayed_dark_ge50_ct']['strip'], r['displayed_dark_ge50_page']['strip']
            r['bg_decision'] = dict(rule='keep CT iff strip share (displayed darkness >= 0.5) with CT <= 1.25 x without CT', ratio=round(a / max(b, 1e-9), 3),
                                    keep_ct=bool(a <= 1.25 * b))
        res[key] = r
        print(key, json.dumps(r), flush=True)
    json.dump(res, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main()
