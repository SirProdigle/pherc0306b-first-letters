"""make_normals.py M7_ZARR OUT_DIR --z Z0 Z1 [--sigma 1.0] [--sigma-t 3.0] [--min-energy 2e-4] [--tile 512] [--device cuda]

Lasagna-compatible dense sheet normals for scrolls without a published Lasagna prediction, from the m7 surface
prediction alone: structure tensor of the m7 level-1 field (gradient after a Gaussian of --sigma level-1 voxels,
tensor smoothed with --sigma-t), principal eigenvector, mean-pooled 2x onto the Lasagna group-2 grid (scale 4,
32^3 chunks), sign fixed to nz >= 0, encoded as the fitter decodes them: u8 = 128 + 127 n (nx, ny); 0/0 = no data
(where the tensor energy is below --min-energy, i.e. no predicted sheet nearby).
Writes OUT_DIR/las_008_nx.ome.zarr and las_008_ny.ome.zarr (zarr v2, uncompressed chunks, only non-empty chunks),
group "2", covering working z [Z0, Z1) rounded out to 128.
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
import zarr


def gauss1d(sigma, device):
    r = max(1, int(round(3 * sigma)))
    x = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum(), r


def blur3(v, sigma):
    """separable Gaussian on (C, Z, Y, X), replicate padding"""
    k, r = gauss1d(sigma, v.device)
    C = v.shape[0]
    v = v[None]
    for dim in range(3):
        shape = [1, 1, 1, 1, 1]; shape[2 + dim] = len(k)
        w = k.view(shape).repeat(C, 1, 1, 1, 1)
        pad = [0, 0, 0, 0, 0, 0]; pad[2 * (2 - dim)] = r; pad[2 * (2 - dim) + 1] = r
        v = F.conv3d(F.pad(v, pad, mode='replicate'), w, groups=C)
    return v[0]


def normals_block(blk, sigma, sigma_t, min_energy):
    """blk (Z, Y, X) float in [0,1] on device -> (n (3, Z/2, Y/2, X/2) zyx unit, valid (Z/2, Y/2, X/2))"""
    s = blur3(blk[None], sigma)[0]
    gz = torch.zeros_like(s); gy = torch.zeros_like(s); gx = torch.zeros_like(s)
    gz[1:-1] = 0.5 * (s[2:] - s[:-2]); gy[:, 1:-1] = 0.5 * (s[:, 2:] - s[:, :-2]); gx[:, :, 1:-1] = 0.5 * (s[:, :, 2:] - s[:, :, :-2])
    T = torch.stack([gz * gz, gz * gy, gz * gx, gy * gy, gy * gx, gx * gx])
    T = blur3(T, sigma_t)
    T = F.avg_pool3d(T[None], 2)[0]                       # level-1 -> level-2 grid
    zz, zy, zx, yy, yx, xx = T
    M = torch.stack([torch.stack([zz, zy, zx], -1), torch.stack([zy, yy, yx], -1), torch.stack([zx, yx, xx], -1)], -2)
    shp = M.shape[:3]
    M = M.reshape(-1, 3, 3)
    # principal eigenvector by power iteration (batched cuSOLVER eigh asks for absurd workspaces on large batches)
    v = M[:, :, 0] + M[:, :, 1] + M[:, :, 2] + 1e-9                  # start: sum of the columns
    v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    for _ in range(25):
        v = torch.bmm(M, v[:, :, None])[:, :, 0]
        v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    lam1 = torch.einsum('ni,nij,nj->n', v, M, v)
    tr = M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2]
    n = torch.where(v[:, :1] < 0, -v, v)                    # nz >= 0
    valid = (tr > min_energy) & (lam1 / tr.clamp(min=1e-12) > 0.5)
    return n.T.reshape(3, *shp), valid.reshape(shp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('m7'); ap.add_argument('out')
    ap.add_argument('--z', type=int, nargs=2, required=True)
    ap.add_argument('--sigma', type=float, default=1.0); ap.add_argument('--sigma-t', type=float, default=3.0)
    ap.add_argument('--min-energy', type=float, default=2e-4)
    ap.add_argument('--tile', type=int, default=512)
    ap.add_argument('--y', type=int, nargs=2, default=None, help='level-2 y range (testing)')
    ap.add_argument('--x', type=int, nargs=2, default=None, help='level-2 x range (testing)')
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    src = zarr.open(a.m7, mode='r')['1']
    Z1s, Y1s, X1s = src.shape
    shp2 = tuple((v + 1) // 2 for v in src.shape)
    C = 32
    z2a = (a.z[0] // 128) * 32; z2b = min(shp2[0], -(-a.z[1] // 128) * 32)
    y2r = a.y or (0, shp2[1]); x2r = a.x or (0, shp2[2])
    T2 = a.tile // 2                                         # level-2 tile edge (multiple of 32)
    H = 12                                                   # level-1 halo
    outs = {}
    for ch in ('nx', 'ny'):
        p = os.path.join(a.out, f'las_008_{ch}.ome.zarr'); os.makedirs(os.path.join(p, '2'), exist_ok=True)
        json.dump({'zarr_format': 2}, open(os.path.join(p, '.zgroup'), 'w'))
        json.dump({'multiscales': [{'version': '0.4', 'axes': [{'name': n, 'type': 'space'} for n in 'zyx'],
                                    'datasets': [{'path': '2', 'coordinateTransformations': [{'type': 'scale', 'scale': [4.0, 4.0, 4.0]}]}]}],
                   'source': 'make_normals.py (m7 structure tensor)', 'm7': os.path.abspath(a.m7),
                   'sigma_l1': a.sigma, 'sigma_t_l1': a.sigma_t, 'min_energy': a.min_energy},
                  open(os.path.join(p, '.zattrs'), 'w'))
        json.dump({'shape': list(shp2), 'chunks': [C, C, C], 'dtype': '|u1', 'fill_value': 0, 'order': 'C', 'filters': None,
                   'dimension_separator': '/', 'compressor': None, 'zarr_format': 2}, open(os.path.join(p, '2', '.zarray'), 'w'))
        outs[ch] = p
    nchunks = 0
    for z2 in range(z2a, z2b, C):
        for y2 in range(y2r[0] // C * C, y2r[1], T2):
            for x2 in range(x2r[0] // C * C, x2r[1], T2):
                z1, y1, x1 = 2 * z2, 2 * y2, 2 * x2
                lo = [max(0, v - H) for v in (z1, y1, x1)]
                hi = [min(s, v + e + H) for s, v, e in zip((Z1s, Y1s, X1s), (z1, y1, x1), (2 * C, 2 * T2, 2 * T2))]
                blk = np.asarray(src[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
                if blk.max() == 0:
                    continue
                # pad so the core starts at an even offset H and the block has even size
                pre = [v - l for v, l in zip((z1, y1, x1), lo)]
                padb = [(H - p_, 0) for p_ in pre]
                blk = np.pad(blk, padb)
                tgt = [2 * C + 2 * H, 2 * T2 + 2 * H, 2 * T2 + 2 * H]
                blk = np.pad(blk, [(0, max(0, t - s)) for t, s in zip(tgt, blk.shape)])[:tgt[0], :tgt[1], :tgt[2]]
                t = torch.from_numpy(blk).to(a.device).float() / 255.0
                n, valid = normals_block(t, a.sigma, a.sigma_t, a.min_energy)
                h2 = H // 2
                n = n[:, h2:h2 + C, h2:h2 + T2, h2:h2 + T2]; valid = valid[h2:h2 + C, h2:h2 + T2, h2:h2 + T2]
                enc = torch.clamp(torch.round(128 + 127 * n), 1, 255).to(torch.uint8)
                enc = torch.where(valid[None], enc, torch.zeros_like(enc)).cpu().numpy()
                for cy in range(0, T2, C):
                    for cx in range(0, T2, C):
                        gy, gx = (y2 + cy) // C, (x2 + cx) // C
                        if (y2 + cy) >= shp2[1] or (x2 + cx) >= shp2[2]:
                            continue
                        sub = enc[:, :, cy:cy + C, cx:cx + C]
                        if not sub.any():
                            continue
                        for ch, k in (('nx', 2), ('ny', 1)):     # n is zyx
                            d = os.path.join(outs[ch], '2', str(z2 // C), str(gy))
                            os.makedirs(d, exist_ok=True)
                            with open(os.path.join(d, str(gx)), 'wb') as f:
                                f.write(np.ascontiguousarray(sub[k]).tobytes())
                        nchunks += 1
        print(f'z2 {z2} done, chunks {nchunks}', flush=True)
    print('DONE', nchunks)


if __name__ == '__main__':
    main()
