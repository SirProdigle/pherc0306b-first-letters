"""Patch quality from the render's depth profile.

On-sheet, the mesh follows one papyrus surface, so every tile shows the same layering along the
normal (sheet material on one side, the gap to the next sheet on the other) and the tile's mean
intensity per layer varies strongly with depth. Where the mesh cuts across sheets, a tile mixes
many depths and the profile goes flat. Score per 128x128 tile: (max - min) of the per-layer mean,
over the median valid intensity. Patch score: fraction of valid tiles above THR.

usage: quality.py render.zarr [...]   -> prints json per render, writes <render>/quality.json
"""
import sys, json, numpy as np, zarr

TILE = 128
THR = 0.10


def tile_scores(a):
    D, H, W = a.shape
    valid = a[D // 2] > 0
    med = float(np.median(a[D // 2][valid])) if valid.any() else 1.0
    ny, nx = H // TILE, W // TILE
    sc = np.full((ny, nx), np.nan, np.float32)
    for i in range(ny):
        for j in range(nx):
            sl = (slice(i * TILE, (i + 1) * TILE), slice(j * TILE, (j + 1) * TILE))
            v = valid[sl]
            if v.mean() < 0.8:
                continue
            blk = a[:, sl[0], sl[1]][:, v].astype(np.float32)
            p = blk.mean(1)
            sc[i, j] = (p.max() - p.min()) / max(med, 1.0)
    return sc


def quality(path):
    a = np.asarray(zarr.open(path, mode='r')['0'][:])
    sc = tile_scores(a)
    ok = sc[~np.isnan(sc)]
    res = dict(n_tiles=int(ok.size), good_frac=float((ok > THR).mean()) if ok.size else 0.0,
               median=float(np.median(ok)) if ok.size else 0.0,
               good_area_cm2=float((ok > THR).sum() * (TILE * 8.64e-4) ** 2))
    np.save(path.rstrip('/') + '/quality_tiles.npy', sc)
    json.dump(res, open(path.rstrip('/') + '/quality.json', 'w'))
    return res


if __name__ == '__main__':
    for p in sys.argv[1:]:
        print(p.split('/')[-1][:40], json.dumps(quality(p)))
