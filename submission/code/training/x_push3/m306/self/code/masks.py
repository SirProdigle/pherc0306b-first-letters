"""masks.py [REGION,..] : SELF pool masks per region (whole render) -> m306/self/pool/<r>_masks.npz 
  dist (H/4+1, W/4+1) float32 : mm from the lead-A exclusion zone (check.distance_mm, step 4, capped at 8 mm; nan = no mesh)
  qual (H/128, W/128) float32 : fl/scripts/quality.py depth-profile score per 128 px tile (on-sheet when >= 0.10, dn343ft rule)
  cov24 (H, W) bool            : CT > 0 on all 24 layers v8-in reads (layer0 .. layer0 + 23)
Derived (pool.dmask): allowed0 = pad 0 (teacher input: CT outside it is zeroed before any teacher runs), allowed07 = pad 0.7
(every labelled / input-tile pixel)."""
import sys, os, json
import numpy as np, zarr
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import pool   # noqa: E402
QT, QTHR = 128, 0.10


def qual_tiles(a):                                  # = fl/scripts/quality.py tile_scores, band-wise
    D, H, W = a.shape
    mid = np.asarray(a[D // 2]); valid = mid > 0
    med = float(np.median(mid[valid])) if valid.any() else 1.0
    ny, nx = H // QT, W // QT
    sc = np.full((ny, nx), np.nan, np.float32)
    for i in range(ny):
        band = np.asarray(a[:, i * QT:(i + 1) * QT, :nx * QT]).astype(np.float32)
        for j in range(nx):
            v = valid[i * QT:(i + 1) * QT, j * QT:(j + 1) * QT]
            if v.mean() < 0.8: continue
            p = band[:, :, j * QT:(j + 1) * QT][:, v].mean(1)
            sc[i, j] = (p.max() - p.min()) / max(med, 1.0)
    return sc


def regions():
    R = json.load(open(f'{pool.POOL}/regions_pool.json')); R.pop('_note', None); return R


def load(r):
    z = np.load(f'{pool.POOL}/{r}_masks.npz')
    return dict(dist=z['dist'], qual=z['qual'], cov24=np.unpackbits(z['cov24p'], axis=1)[:, :int(z['W'])].astype(bool),
                H=int(z['H']), W=int(z['W']))


def derived(m, pad=0.7):
    H, W = m['H'], m['W']
    a0 = pool.dmask(m['dist'], (H, W), 0.0); ap = pool.dmask(m['dist'], (H, W), pad)
    q = np.nan_to_num(m['qual'], nan=-1) >= QTHR
    qf = np.zeros((H, W), bool); qq = np.repeat(np.repeat(q, QT, 0), QT, 1); qf[:qq.shape[0], :qq.shape[1]] = qq
    return a0, ap, qf


if __name__ == '__main__':
    R = regions()
    for r in (sys.argv[1].split(',') if len(sys.argv) > 1 else list(R)):
        out = f'{pool.POOL}/{r}_masks.npz'
        if os.path.exists(out): continue
        g = R[r]; a = zarr.open(g['zarr'], mode='r')['0']; D, H, W = a.shape
        dist = pool.dist_grid(g['zarr'], (0, H, 0, W), step=4)
        qual = qual_tiles(a)
        l0 = g['layer0']; cov = np.ones((H, W), bool)
        for y in range(0, H, 512):
            blk = np.asarray(a[l0:l0 + 24, y:y + 512]); cov[y:y + 512] = (blk > 0).all(0)
        np.savez_compressed(out, dist=dist, qual=qual, cov24p=np.packbits(cov, axis=1), H=H, W=W)
        m = load(r); a0, a7, qf = derived(m)
        px = 0.00864 ** 2 / 100
        st = dict(region=r, shape=[H, W], ct_cm2=round(cov.sum() * px, 2), allowed0_cm2=round((a0 & cov).sum() * px, 2),
                  allowed07_cm2=round((a7 & cov).sum() * px, 2), onsheet_frac=round(float(np.nanmean(np.nan_to_num(qual, nan=0) >= QTHR)), 3),
                  pool_cm2=round((a7 & cov & qf).sum() * px, 2))
        json.dump(st, open(f'{pool.POOL}/{r}_masks.json', 'w'))
        print(json.dumps(st), flush=True)
