"""labels.py [REGION,..] : SELF pseudo-labels from the cleared teacher fusion.

Per region (bbox frame of teachers.py), valid = pool mask (allowed at pad 0.7 & CT on all 24 read layers & on-sheet tile score
>= 0.10) & every teacher finite.  Each teacher (v8-in, Reader V3, TRACE v3B, Reader v2) is z-scored over the region's valid pixels
(no statistic from the zone: valid pixels are all >= 2.9 mm from it), the four z maps are averaged and the average is z-scored again
over valid pixels -> fz ("fused z").  Labels (uint8: 1 ink, 0 background, 255 ignored):
  main  : dn343ft rule: ink = fz >= 2.5 in 8-connected blobs of 0.3-6 mm^2; bg = fz <= 0.3 and >= 20 px from ink; rest ignored
  sc    : sharp core: ink = main ink AND v8-in's own z >= SC_T (blobs >= 0.05 mm^2); the rest of main ink is ignored; bg = main bg
  decoy : main labels translated by (dy, dx) = DECOY_MM (3.5, 3.0) mm inside the bbox, kept only where valid (hallucination control)
Outputs m306/self/labels/<r>_labels.npz (fz float16, zv8 float16, main / sc / decoy uint8, valid, bbox) + <r>_labels.jpg +
labels/stats.json."""
import sys, os, json
import numpy as np, cv2, zarr
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import pool, masks   # noqa: E402
TEACH = f'{pool.SELF}/teach'; LAB = os.environ.get('SELF_LAB', f'{pool.SELF}/labels'); os.makedirs(LAB, exist_ok=True)
PX_MM2 = (1000 / 8.64) ** 2
POS_T, NEG_T, MIN_MM2, MAX_MM2, NEAR = 2.5, 0.3, 0.3, 6.0, 20
SC_T = float(os.environ.get('SC_T', 1.5)); DECOY_MM = (3.5, 3.0)
MODELS = tuple(os.environ.get('SELF_MODELS', 'v8,v3,trace,rv2').split(','))


def blobs(m, lo, hi):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    a = st[:, cv2.CC_STAT_AREA] / PX_MM2
    keep = (a >= lo) & (a <= hi); keep[0] = False
    return keep[lab]


def make(r):
    tj = json.load(open(f'{TEACH}/{r}_teach.json')); y0, y1, x0, x1 = tj['bbox']
    m = masks.load(r); a0, a7, qf = masks.derived(m)
    pm = (a7 & m['cov24'] & qf)[y0:y1, x0:x1]
    T = {k: np.load(f'{TEACH}/{r}_{k}.npy').astype(np.float32) for k in MODELS}
    valid = pm.copy()
    for k in MODELS: valid &= np.isfinite(T[k])
    Z = {}
    for k in MODELS:
        v = T[k][valid]; Z[k] = (T[k] - v.mean()) / max(v.std(), 1e-6)
    f = np.mean([Z[k] for k in MODELS], 0)
    v = f[valid]; fz = (f - v.mean()) / max(v.std(), 1e-6); fz[~valid] = np.nan
    zv8 = Z['v8'] if 'v8' in Z else Z[MODELS[0]].copy(); zv8[~valid] = np.nan
    fzn = np.nan_to_num(fz, nan=0.0)
    pos = blobs((fzn >= POS_T) & valid, MIN_MM2, MAX_MM2)
    near = cv2.dilate(pos.astype(np.uint8), np.ones((2 * NEAR + 1, 2 * NEAR + 1), np.uint8)) > 0
    neg = (fzn <= NEG_T) & valid & ~near & np.isfinite(fz)
    main = np.full(fz.shape, 255, np.uint8); main[neg] = 0; main[pos] = 1
    core = blobs(pos & (np.nan_to_num(zv8, nan=-9) >= SC_T), 0.05, 1e9)
    sc = np.full(fz.shape, 255, np.uint8); sc[neg] = 0; sc[core] = 1
    dy, dx = int(round(DECOY_MM[0] * 1000 / 8.64)), int(round(DECOY_MM[1] * 1000 / 8.64))
    dec = np.full(fz.shape, 255, np.uint8); H, Wd = fz.shape
    dec[dy:, dx:] = main[:H - dy, :Wd - dx]; dec[~valid] = 255
    np.savez_compressed(f'{LAB}/{r}_labels.npz', fz=fz.astype(np.float16), zv8=zv8.astype(np.float16), main=main, sc=sc, decoy=dec,
                        valid=valid, bbox=np.array(tj['bbox']))
    st = dict(region=r, bbox=tj['bbox'], valid_cm2=round(valid.sum() / PX_MM2 / 100, 3))
    for nm, L in (('main', main), ('sc', sc), ('decoy', dec)):
        n_bl = cv2.connectedComponents((L == 1).astype(np.uint8), connectivity=8)[0] - 1
        st[nm] = dict(ink_mm2=round((L == 1).sum() / PX_MM2, 2), bg_mm2=round((L == 0).sum() / PX_MM2, 1), ink_blobs=int(n_bl),
                      ink_frac_of_labelled=round(float((L == 1).sum() / max(1, (L != 255).sum())), 4))
    # teacher agreement on valid px (Pearson r with the fused map)
    st['r_with_fused'] = {k: round(float(np.corrcoef(Z[k][valid][::17], f[valid][::17])[0, 1]), 3) for k in MODELS}
    # overview: CT surface | fused z | labels (main: ink red, bg blue; sc core yellow)
    g = masks.regions()[r]
    ct = np.asarray(zarr.open(g['zarr'], mode='r')['0'][g['c'], y0:y1, x0:x1]).astype(np.float32)
    ctv = np.clip(ct * 1.2, 0, 255).astype(np.uint8); ctv[~valid] //= 3
    fzv = (np.clip((fzn + 1) / 4.5, 0, 1) * 255).astype(np.uint8); fzv[~valid] = 0
    lab = np.stack([ctv // 2] * 3, -1)
    lab[main == 0] = (lab[main == 0] * [1.6, 1.0, 0.7]).clip(0, 255).astype(np.uint8)
    lab[main == 1] = (0, 0, 255); lab[sc == 1] = (0, 255, 255)
    tiles = [np.stack([ctv] * 3, -1), np.stack([fzv] * 3, -1), lab]
    s = min(1.0, 1400 / max(fz.shape))
    tiles = [cv2.resize(t, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) for t in tiles]
    sep = np.full((tiles[0].shape[0], 8, 3), 255, np.uint8)
    im = np.hstack([tiles[0], sep, tiles[1], sep, tiles[2]]) if fz.shape[1] < 1.3 * fz.shape[0] else np.vstack(
        [tiles[0], np.full((8, tiles[0].shape[1], 3), 255, np.uint8), tiles[1], np.full((8, tiles[0].shape[1], 3), 255, np.uint8), tiles[2]])
    cv2.putText(im, f'{r}: CT | fused z (-1..3.5) | labels main ink red, sharp core yellow, bg tint', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imwrite(f'{LAB}/{r}_labels.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return st


if __name__ == '__main__':
    R = masks.regions(); regs = sys.argv[1].split(',') if len(sys.argv) > 1 else list(R)
    sp = f'{LAB}/stats.json'; S = json.load(open(sp)) if os.path.exists(sp) else {}
    for r in regs:
        if not all(os.path.exists(f'{TEACH}/{r}_{k}.npy') for k in MODELS): print(r, 'teachers missing'); continue
        S[r] = make(r); print(json.dumps(S[r]), flush=True)
    S['_rule'] = dict(pos_t=POS_T, neg_t=NEG_T, blob_mm2=[MIN_MM2, MAX_MM2], near_px=NEAR, sc_t=SC_T, decoy_mm=DECOY_MM)
    json.dump(S, open(sp, 'w'), indent=1)
