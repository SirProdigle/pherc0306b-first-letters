"""controls_panel_v7.py : one sheet of the main image's false-ink controls (version 7).

panel(cx, rows, out_jpg, blank) with cx = controls_v5.Ctx and rows = [(label, pair, kind)], kind 'view' (the pair's grey view at its
own levels) or 'shuffle' (the depth-shuffled maps of that pair at the REAL view's levels: controls_v5.shuffle_stats arithmetic, as
m306/eval/code/evlib.shuffle_view).  Each row is the claim box u -15..+23 x v -6.84..+2.72 mm in reading orientation, captioned with
its share of the box and of blank control R darker than half tone.  Returns the numbers."""
import json
import numpy as np, cv2
import claimlib_v2 as L
import controls_v5 as CV

BOX = dict(u0=-15.0, u1=23.0, v0=-6.84, v1=2.72)


def crop_rc(b):
    PP = L.PPMX
    return (int(round((L.CTX['v1'] - b['v1']) * PP)), int(round((L.CTX['v1'] - b['v0']) * PP)),
            int(round((b['u0'] - L.CTX['u0']) * PP)), int(round((b['u1'] - L.CTX['u0']) * PP)))


def grey_view(cx, pair, blank):
    p = cx.pair(pair); d, lv = cx.tone(p, 'grey')
    bg = np.where(p['valid'][..., None], L.PAGE[None, None], 60).astype(np.float32)
    rgb = np.ascontiguousarray(L.q4_rgb(d, bg, p['valid'])[::-1])
    st = dict(levels={k: round(v, 4) for k, v in lv.items()}, box=CV.stats4(d, p['inbox']), blank=CV.stats4(d, cx.region(blank, p)))
    return rgb, p, lv, st


def shuffle_rgb(cx, pair, p, lv):
    mods = p['mods']; m, valid = p['m'], p['valid']
    st = CV.shuffle_stats(cx, pair, p, lv, 'grey')
    s0 = {k: (float(m[k][valid].astype(np.float64).mean()), float(m[k][valid].astype(np.float64).std())) for k in mods}
    ms = {k: L.sample_raster(L.load_raster_map(f'{cx.shuf}/fwd_{k}.npy', CV.RAST), L.CTX, L.UMX) for k in mods}
    vb = p['inbox'] & np.logical_and.reduce([ms[k] != 0 for k in mods])
    f1 = np.mean([(ms[k] - s0[k][0]) / (s0[k][1] + 1e-9) for k in mods], 0)
    d1 = np.where(vb, np.clip((f1 - lv['lo_median']) / (lv['hi_p999'] - lv['lo_median']), 0, 1), 0).astype(np.float32)
    bg = np.where(vb[..., None], L.PAGE[None, None], np.where(valid[..., None], L.PAGE[None, None] * 0.93, 60)).astype(np.float32)
    return np.ascontiguousarray(np.clip(bg * (1 - d1[..., None]) + L.INK[None, None] * d1[..., None], 0, 255).astype(np.uint8)[::-1]), st


def strip(width, text, h=60, scale=1.15):
    s = np.full((h, width, 3), 255, np.uint8)
    cv2.putText(s, text, (12, int(h * 0.68)), cv2.FONT_HERSHEY_SIMPLEX, scale, (15, 15, 15), 2, cv2.LINE_AA)
    return s


def panel(cx, rows, out_jpg, blank, width=2400, title=''):
    r0, r1, c0, c1 = crop_rc(BOX); res = {}; ims = []
    if title: ims.append(strip(width, title, 70, 1.3))
    for lab, pair, kind in rows:
        rgb, p, lv, st = grey_view(cx, pair, blank)
        if kind == 'shuffle':
            rgb, sst = shuffle_rgb(cx, pair, p, lv)
            res[lab] = dict(pair=pair, kind=kind, box_half_tone_shuffled=sst['shuffled']['half_tone'], box_painted_shuffled=sst['shuffled']['painted'],
                            p99_shuffled=sst['shuffled']['p99_tone'], box_half_tone_real=sst['original']['half_tone'])
            cap = (f'{lab}: DEPTH-SHUFFLED layers, at the real view\'s levels. Box darker than half tone: {100 * sst["original"]["half_tone"]:.1f} % '
                   f'-> {100 * sst["shuffled"]["half_tone"]:.2f} %')
        else:
            res[lab] = dict(pair=pair, kind=kind, levels=st['levels'], box_half_tone=st['box']['half_tone'], blank_half_tone=st['blank']['half_tone'])
            cap = (f'{lab}: own grey levels. Darker than half tone: box {100 * st["box"]["half_tone"]:.1f} %, blank control R '
                   f'{100 * st["blank"]["half_tone"]:.2f} %')
        im = rgb[r0:r1, c0:c1]; f = width / im.shape[1]
        im = cv2.resize(im, (width, int(round(im.shape[0] * f))), interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_NEAREST)
        ims += [strip(width, cap), im, np.full((14, width, 3), 255, np.uint8)]
    out = np.vstack(ims)
    cv2.imwrite(out_jpg, cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    json.dump(res, open(out_jpg[:-4] + '.json', 'w'), indent=1)
    return res
