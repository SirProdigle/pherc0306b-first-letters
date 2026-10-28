"""blank_strip.py MAPS CT OUT.json : label-free false-positive check of the claim display (lane I2).
Share of a blank strip of the SAME context (u -11.2..+28.8 x v +6..+15 mm, claim frame as rendered; I1's strip) that the claim
image paints at the claim box's own levels (L, H, P from the box), for each pair and rung, against the share painted in the box."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claimlib_v2 as L, make_claim_image_v2 as C                                   # noqa: E402
maps, ct, out = sys.argv[1:4]
U, V = L.grid_uv(L.CTX, L.UMX); strip = (U >= -11.2) & (U <= 28.8) & (V >= 6) & (V <= 15)
res = dict(strip_mm=dict(u0=-11.2, u1=28.8, v0=6.0, v1=15.0))
for pair in ('rv8', 'rt'):
    for rung in ('primary', 'secondary'):
        _, info, x = C.build(maps, ct, pair, rung, lambda lv: [''], lambda lv: [''])
        s = strip & x['valid']
        res[f'{pair}_{rung}'] = dict(levels=info['levels'], painted_share_box=info['visible_share_box'],
                                     painted_share_blank_strip=round(float((x['d'][s] > 0).mean()), 4),
                                     half_tone_share_blank_strip=round(float((x['d'][s] >= 0.5).mean()), 5), strip_px=int(s.sum()))
json.dump(res, open(out, 'w'), indent=1); print(json.dumps(res, indent=1))
