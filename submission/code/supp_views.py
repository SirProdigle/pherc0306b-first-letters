"""supp_views.py MAPS CT MESH OUTDIR : supplementary views of the v2 draft with neutral headers (lane I2).
The same views as lane I1's build_claim_v2.py 'supp' block (code copied, I1's old-display helpers verbatim), for both pairs:
  model_{rv2,tr,v8}_primary.jpg              each model alone under the rule (own p99 in the box)
  display_ladder_L020_L035_L050_{pair}.jpg   the claim box + 3 mm at L = 0.20 / 0.35 / 0.50 x p99
  old_display_yellow_{pair}.jpg              the same fused map drawn as on 14 Oct (yellow, z 0.9-2.7, over CLAHE CT at 45 %)
  bare_ct.jpg                                the surface CT layer, linear 1-99.5 % stretch, no ink"""
import os, sys
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claimlib_v2 as L, make_claim_image_v2 as C                                   # noqa: E402
YEL = np.array([250, 232, 120], np.float32)
TAG = {'rt': 'Rv2_TRACE', 'rv8': 'Rv2_v8in'}


def ct_bg(ct, valid):                                                               # I1 i1lib.ct_bg, verbatim
    x = ct[valid]; lo, hi = np.percentile(x, [1, 99.5])
    g = np.clip((ct - lo) / (hi - lo + 1e-6), 0, 1); g[~valid] = 0
    g = cv2.createCLAHE(2.0, (8, 8)).apply((g * 255).astype(np.uint8)).astype(np.float32) * 0.45
    return np.repeat(g[..., None], 3, -1)


def claim_rgb(d, valid, ct):                                                        # I1 i1lib.claim_rgb, verbatim
    """D5: yellow at opacity d over the dimmed CLAHE CT (RGB uint8)."""
    bg = ct_bg(ct, valid)
    out = bg * (1 - d[..., None]) + YEL[None, None] * d[..., None]
    out[~valid] = 0
    return np.clip(out, 0, 255).astype(np.uint8)


def old_rgb(z, ct, valid, lo=0.9, hi=2.7):                                          # I1 i1lib.old_rgb, verbatim
    """the submitted look: yellow opacity z lo..hi over CLAHE CT at 45 % (make_final_image)."""
    a = np.where(valid, np.clip((z - lo) / (hi - lo), 0, 1), 0).astype(np.float32)
    return claim_rgb(a, valid, ct)


def save(fn, rgb):
    cv2.imwrite(fn, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92]); print(fn)


def main():
    maps, ctp, mesh, S = sys.argv[1:5]; os.makedirs(S, exist_ok=True)
    hdr0 = f'PHerc0306B | volume 20250521133212 (8.64 um, 116 keV) | mesh {os.path.basename(os.path.normpath(mesh))}'
    m, ct, valid, _ = C.load(maps, ctp, ('rv2', 'tr', 'v8'), L.CTX)
    U, V = L.grid_uv(L.CTX, L.UMX); inbox = L.box_mask(U, V, L.CBOX) & valid
    z = {k: L.zs(m[k], valid) * valid for k in m}
    bg = L.light_ct_bg(ct, valid, valid); R = lambda a: np.ascontiguousarray(a[::-1])   # noqa: E731
    PP = L.PPMX
    bx0 = int((L.CBOX['u0'] - L.CTX['u0']) * PP); bx1 = int((L.CBOX['u1'] - L.CTX['u0']) * PP)
    by0 = int((L.CTX['v1'] - L.CBOX['v1']) * PP); by1 = int((L.CTX['v1'] - L.CBOX['v0']) * PP)
    for k in ('rv2', 'tr', 'v8'):
        lv = L.q4_levels(z[k], inbox, 0.35); d = L.q4_tone(z[k], lv, valid)
        img = R(L.q4_rgb(d, bg, valid)); L.dashed_rect(img, bx0, by0, bx1, by1, (30, 90, 220))
        save(f'{S}/model_{k}_primary.jpg', L.frame_image(img, [hdr0, f'{C.NAMES[k]} ALONE, same rule fixed in advance (primary rung, its own p99 in the '
                                                                   f'4 cm2 box: z {lv["L"]:.2f}..{lv["H"]:.2f}). Reading orientation; dark = ink.'], []))
    sub = (slice(max(0, by0 - int(3 * PP)), by1 + int(3 * PP)), slice(max(0, bx0 - int(3 * PP)), bx1 + int(3 * PP)))
    for pair in ('rt', 'rv8'):
        fused = np.mean([z[k] for k in C.PAIRS[pair]], 0); names = ' + '.join(C.NAMES[k] for k in C.PAIRS[pair])
        img = R(old_rgb(fused, ct, valid)); L.dashed_rect(img, bx0, by0, bx1, by1, (220, 220, 220))
        save(f'{S}/old_display_yellow_{TAG[pair]}.jpg', L.frame_image(img, [hdr0, f'{names}: the same map drawn the way our 14 Oct image was '
                                                                              '(yellow opacity z 0.9..2.7 over CLAHE CT at 45 %).'], []))
        rows = []
        for f in (0.20, 0.35, 0.50):
            lv = L.q4_levels(fused, inbox, f); d = L.q4_tone(fused, lv, valid)
            rows.append(L.frame_image(R(L.q4_rgb(d, bg, valid))[sub], [f'{names} | L = {f:.2f} x p99 (z {lv["L"]:.2f}), full tone at p99 '
                                                                       f'(z {lv["H"]:.2f}){" (primary)" if f == 0.35 else " (secondary)" if f == 0.5 else ""}'], [], margin=40))
        save(f'{S}/display_ladder_L020_L035_L050_{TAG[pair]}.jpg', np.vstack(rows))
    x = ct[valid]; lo, hi = np.percentile(x, [1, 99.5]); g = (np.clip((ct - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8); g[~valid] = 0
    save(f'{S}/bare_ct.jpg', L.frame_image(R(np.repeat(g[..., None], 3, -1)), [hdr0, 'Bare CT, surface layer, linear 1-99.5 % stretch (no ink).'], []))


if __name__ == '__main__':
    main()
