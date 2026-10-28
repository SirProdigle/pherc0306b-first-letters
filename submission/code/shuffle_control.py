"""shuffle_control.py prep|score ... : depth-shuffle control of the claim display for both pairs, AT THE CLAIM LEVELS (lane I2).

Ink sits at a depth. A model that still paints marks after the order of the layers it reads is scrambled is reading in-plane
texture.  Q4 P-2's pre-registered control: the hot fraction of the shuffled Reader v2 + v8-in map may be at most +25 % relative to
the shuffled Reader v2 + TRACE map.

  prep  RENDER.zarr OUT.zarr [SEED]   copy the render over the 4 cm2 box + 256 px with layers 8..31 permuted (seed 1; every layer any
                                      model reads: Reader v2 / TRACE read 20 +- 10 with d -1..+1, v8-in reads 8..31); elsewhere 0.
  (GPU, same recipes as the claim maps: ink_v2.py OUT.zarr SHUF --models rv2,tr --crop BOX;  ink_v8.py OUT.zarr SHUF --region box)
  variants MAPS CT SHUF OUT.json     every display variant at its own levels (score_variants)
  score MAPS CT SHUF OUT.json        z of each shuffled map with the ORIGINAL map's context mean / std (as the image), fused per pair,
                                      darkness with the claim image's own levels (primary and secondary); hot = share of the box's
                                      on-sheet px painted (d > 0) and at half tone (d >= 0.5); r of the fused z with the original's.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claimlib_v2 as L, make_claim_image_v2 as C                                   # noqa: E402
BOXPX = (1639, 2745, 2431, 6829)                                                     # render px y0, y1, x0, x1 (centres inside the box)


def prep(render, out, seed=1):
    import zarr, numcodecs
    a = zarr.open(render, mode='r')['0']; D, H, Wd = a.shape; m = 256
    y0, y1, x0, x1 = BOXPX[0] - m, BOXPX[1] + m, BOXPX[2] - m, BOXPX[3] + m
    perm = np.random.default_rng(seed).permutation(24)
    assert (perm != np.arange(24)).sum() >= 20, perm
    order = np.arange(D); order[8:32] = 8 + perm
    g = zarr.open_group(out, mode='w', zarr_format=2)
    b = g.create_array('0', shape=(D, H, Wd), chunks=(1, 1024, 1024), dtype=a.dtype, fill_value=0,
                       compressors=numcodecs.Blosc(cname='zstd', clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE))
    blk = np.asarray(a[:, y0:y1, x0:x1])
    b[:, y0:y1, x0:x1] = blk[order]
    g.attrs['note'] = f'layers 8..31 of {render} permuted (seed {seed}): new layer k = old layer order[k]; region y {y0}:{y1}, x {x0}:{x1}'
    g.attrs['order'] = order.tolist()
    print(out, 'order', order[8:32].tolist())


def score(maps, ct, shuf, out):
    res = dict(box_px=BOXPX, rule='z with the original maps\' context mean/std; levels of the claim image (primary, secondary); '
                                  'hot = share of the box\'s on-sheet pixels painted (d > 0); half = d >= 0.5')
    U, V = L.grid_uv(L.CTX, L.UMX); box = L.box_mask(U, V, L.CBOX)
    for pair in ('rt', 'rv8'):
        mods = C.PAIRS[pair]
        m, ctg, valid, _ = C.load(maps, ct, mods, L.CTX)
        st = {k: (float(m[k][valid].astype(np.float64).mean()), float(m[k][valid].astype(np.float64).std())) for k in mods}
        ms = {k: L.sample_raster(L.load_raster_map(f'{shuf}/fwd_{k}.npy', (4900, 7780)), L.CTX, L.UMX) for k in mods}
        vb = valid & box & np.logical_and.reduce([ms[k] != 0 for k in mods])
        f0 = np.mean([(m[k] - st[k][0]) / (st[k][1] + 1e-9) for k in mods], 0)
        f1 = np.mean([(ms[k] - st[k][0]) / (st[k][1] + 1e-9) for k in mods], 0)
        inbox = box & valid
        r = dict(px=int(vb.sum()), fused_z_r_shuffled_vs_original=round(float(np.corrcoef(f0[vb], f1[vb])[0, 1]), 4))
        for rung in ('primary', 'secondary'):
            lv = L.q4_levels(f0 * valid, inbox, L.RUNGS[rung])                       # the claim image's levels (z over valid, as build())
            d0 = L.q4_tone(f0, lv, valid); d1 = L.q4_tone(f1, lv, vb)
            r[rung] = dict(levels={k: round(v, 3) for k, v in lv.items()},
                           hot_original=round(float((d0[vb] > 0).mean()), 4), hot_shuffled=round(float((d1[vb] > 0).mean()), 4),
                           half_original=round(float((d0[vb] >= 0.5).mean()), 5), half_shuffled=round(float((d1[vb] >= 0.5).mean()), 5))
        res[pair] = r
    for rung in ('primary', 'secondary'):
        a, b = res['rv8'][rung]['hot_shuffled'], res['rt'][rung]['hot_shuffled']
        res[f'q4_test_{rung}'] = dict(rv8_over_rt=round(a / max(b, 1e-9), 3), bar=1.25, passes=bool(a <= 1.25 * b))
    json.dump(res, open(out, 'w'), indent=1); print(json.dumps(res, indent=1))




def score_variants(maps, ct, shuf, out):
    """every display variant (build_variants registry) at its OWN levels: the shuffled maps are z-scored with the ORIGINAL maps'
    context mean / std and toned with the original variant's levels (threshold: L/H/P of the box; grey: context median / p99.9).
    Box read-outs: painted (d > 0), half tone (d >= 0.5), p99 tone; original vs shuffled.  Q4-style relative test per display kind:
    the v8Q variant's shuffled half-tone share <= 1.25 x the Reader v2 + TRACE variant's of the same kind."""
    import build_variants as B
    U, V = L.grid_uv(L.CTX, L.UMX); box = L.box_mask(U, V, L.CBOX); res = dict(box_px=BOXPX, shuffle='layers 8..31 permuted, seed 1 (prep)')
    for key, v in B.VARIANTS.items():
        mods = C.PAIRS[v['pair']]
        if not all(os.path.exists(f'{shuf}/fwd_{k}.npy') for k in mods): continue
        _, info, x = C.build(maps, ct, v['pair'], v['rung'], lambda lv: [''], lambda lv: [''], display=v['display'], bg_mode=v['bg'])
        m, ctg, valid, _ = C.load(maps, ct, mods, L.CTX)
        st = {k: (float(m[k][valid].astype(np.float64).mean()), float(m[k][valid].astype(np.float64).std())) for k in mods}
        ms = {k: L.sample_raster(L.load_raster_map(f'{shuf}/fwd_{k}.npy', (4900, 7780)), L.CTX, L.UMX) for k in mods}
        vb = valid & box & np.logical_and.reduce([ms[k] != 0 for k in mods])
        f1 = np.mean([(ms[k] - st[k][0]) / (st[k][1] + 1e-9) for k in mods], 0); lv = info['levels']
        if v['display'] == 'grey':
            d1 = np.where(vb, np.clip((f1 - lv['lo_median']) / (lv['hi_p999'] - lv['lo_median']), 0, 1), 0)
        else:
            d1 = L.q4_tone(f1, dict(L=lv['L'], H=lv['H'], P=lv['P']), vb)
        d0 = x['d']
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
    json.dump(res, open(out, 'w'), indent=1); print(json.dumps({k: v for k, v in res.items() if k.startswith('q4_')}, indent=1))


if __name__ == '__main__':
    if sys.argv[1] == 'prep': prep(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 1)
    elif sys.argv[1] == 'variants': score_variants(*sys.argv[2:6])
    else: score(*sys.argv[2:6])
