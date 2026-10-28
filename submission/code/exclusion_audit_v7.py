"""exclusion_audit_v7.py : the exclusion-zone audit of a version-7 ink model's training tiles.

  exclusion_audit_v7.py recompute TRAIN_CONFIG.json OUTDIR [--lane self|train]
        SELF lane: rebuild every PHerc0306B training tile from the machine labels and pool masks EXACTLY as
        m306/self/code/train_self.py (src_306) builds them, check each against the zone, and write OUTDIR/exclusion_audit.json,
        OUTDIR/training_tiles.npz (region index, y, x of each 64 px tile on its region's render) and OUTDIR/tile_regions.json
        (region -> render, mesh and px offset).  TRAIN lane: assert that no training source is PHerc0306B.
  exclusion_audit_v7.py verify TILES.npz TILE_REGIONS.json [--mesh-root DIR] [--zone-dir DIR]
        re-check a shipped tile list with only the zone, the meshes and check.py (no labels, no CT): the Docker image's
        'exclusion' mode.  Exit 1 if any tile fails.
Checks per tile: (a) centre + 4 corners at pad 0.7 mm (the training rule); (b) a 5 x 5 grid (0, 16, 32, 48, 63 px) at pad 0, so every
input pixel is >= 2.0 + 0.2 mm from the zone in 3-D; recompute also checks (c) every 16th labelled pixel at pad 0.
check.py rule: a render pixel maps through its tifxyz mesh to a 3-D point; allowed iff that point is >= 2.0 mm + 0.2 mm (slab) + pad
from every point of every surface shown in any lead-A claim image (zone.npz)."""
import os, sys, json, time, argparse
import numpy as np

T, STRIDE, MIN_LAB = 64, 32, 0.2
GRID = (0, 16, 32, 48, 63)
CORNERS = [(T // 2, T // 2), (0, 0), (0, T - 1), (T - 1, 0), (T - 1, T - 1)]


def check_tiles(check, Y, X, where):
    """where = dict(zarr=...) or dict(mesh=..., offset_yx=...) -> (okA, okB, dmin)"""
    kw = dict(mesh=where['mesh'], offset_yx=tuple(where['offset_yx'])) if where.get('mesh_only') else {}
    z = where.get('zarr') if not where.get('mesh_only') else None
    okA = np.ones(len(Y), bool)
    for dy, dx in CORNERS: okA &= check.allowed(z, Y + dy, X + dx, pad_mm=0.7, **kw)
    okB = np.ones(len(Y), bool); dmin = np.full(len(Y), np.inf)
    for dy in GRID:
        for dx in GRID:
            d = np.nan_to_num(check.distance_mm(z, Y + dy, X + dx, cap_mm=50.0, **kw), nan=-1.0)
            dmin = np.minimum(dmin, d); okB &= d >= check.RADIUS_MM + check.SLAB_MM
    return okA, okB, dmin


def recompute(cfg_path, out, lane, p3):
    sys.path.insert(0, f'{p3}/aclick/p306/exclusion'); import check                 # noqa: E402
    tc = json.load(open(cfg_path)); os.makedirs(out, exist_ok=True)
    rec = dict(train_config=os.path.abspath(cfg_path), lane=lane, radius_mm=check.RADIUS_MM, slab_mm=check.SLAB_MM)
    if lane == 'train':
        objs = [s.get('object') or {'1447': 'PHerc1447', '0343': 'PHerc0343'}.get(s['kind'], s['kind']) for s in tc['sources']]
        rec.update(sources=tc['sources'], objects=sorted(set(objs)), all_allowed=not any('0306' in str(o) for o in objs),
                   note='no PHerc0306B data in training: every tile is from another scroll, so none can overlap the zone')
        json.dump(rec, open(f'{out}/exclusion_audit.json', 'w'), indent=1); return rec
    sys.path.insert(0, f'{p3}/m306/self/code'); import masks                       # noqa: E402 (the lane's own code, read only)
    I = lambda m: np.pad(m.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))                       # noqa: E731 v8306lib.integral
    S = lambda I_, Y, X: I_[Y + T, X + T] - I_[Y, X + T] - I_[Y + T, X] + I_[Y, X]                          # noqa: E731 v8306lib.box_sum
    LAB = os.environ.get('SELF_LAB', f'{p3}/m306/self/labels'); R = masks.regions(); variant = tc['variant']
    regs, tiles, where = {}, [], {}
    for ri, r in enumerate(tc['regions']):
        t0 = time.time(); g = R[r]
        z = np.load(f'{LAB}/{r}_labels.npz'); y0, y1, x0, x1 = [int(v) for v in z['bbox']]; lab8 = z[variant]
        m = masks.load(r); a0, a7, qf = masks.derived(m); pm = (a7 & m['cov24'] & qf)[y0:y1, x0:x1]
        H, Wd = pm.shape
        Y, X = np.meshgrid(np.arange(0, H - T + 1, STRIDE), np.arange(0, Wd - T + 1, STRIDE), indexing='ij'); Y, X = Y.ravel(), X.ravel()
        ok = (S(I(pm), Y, X) == T * T) & (S(I(lab8 != 255), Y, X) >= MIN_LAB * T * T); Y, X = Y[ok] + y0, X[ok] + x0
        okA, okB, dmin = check_tiles(check, Y, X, dict(zarr=g['zarr']))
        ly, lx = np.nonzero(lab8[::16, ::16] != 255); ly, lx = ly * 16 + y0, lx * 16 + x0
        okC = check.allowed(g['zarr'], ly, lx, pad_mm=0.0)
        mesh, off = check.resolve(g['zarr'])
        where[r] = dict(zarr=g['zarr'], mesh=mesh, offset_yx=list(off))
        regs[r] = dict(zarr=g['zarr'], mesh=mesh, offset_yx=list(off), variant=variant, tiles=int(len(Y)),
                       rule_a_centre_corners_pad07_ok=int(okA.sum()), rule_b_grid25_pad0_ok=int(okB.sum()),
                       labelled_px_sampled=int(len(ly)), rule_c_labelled_px_ok=int(okC.sum()),
                       min_distance_mm_tile_grid=round(float(dmin.min()), 3) if len(dmin) else None,
                       median_distance_mm_tile_grid=round(float(np.median(dmin)), 3) if len(dmin) else None, seconds=round(time.time() - t0, 1))
        tiles.append(np.stack([np.full(len(Y), ri), Y, X], 1)); print(f'  exclusion {r}: {regs[r]}', flush=True)
    allok = all(v['rule_a_centre_corners_pad07_ok'] == v['tiles'] and v['rule_b_grid25_pad0_ok'] == v['tiles']
                and v['rule_c_labelled_px_ok'] == v['labelled_px_sampled'] for v in regs.values())
    rec.update(regions=regs, tiles_total=int(sum(v['tiles'] for v in regs.values())), all_allowed=bool(allok),
               teacher_input_rule='CT set to 0 wherever the pad-0 rule fails (< 2.2 mm from the zone, or no mesh) before any teacher ran '
                                  '(m306/self/code/teachers.py region_vol); label z-scores use pool pixels only (>= 2.9 mm from the zone)')
    np.savez_compressed(f'{out}/training_tiles.npz', tiles=np.concatenate(tiles).astype(np.int32), regions=np.array(tc['regions']))
    json.dump(dict(regions=tc['regions'], where=where, note='region index of training_tiles.npz -> render zarr, tifxyz mesh and the px offset '
                   'of the render inside the mesh grid (check.py resolve)'), open(f'{out}/tile_regions.json', 'w'), indent=1)
    json.dump(rec, open(f'{out}/exclusion_audit.json', 'w'), indent=1)
    return rec


def verify(tiles_npz, regions_json, mesh_root='', zone_dir=''):
    zd = zone_dir or os.path.join(os.path.dirname(os.path.abspath(regions_json)), 'exclusion')
    sys.path.insert(0, zd); import check                                            # noqa: E402
    z = np.load(tiles_npz); t = z['tiles']; RJ = json.load(open(regions_json)); bad = 0; res = {}
    for ri, r in enumerate(RJ['regions']):
        w = dict(RJ['where'][r], mesh_only=True)
        if mesh_root: w['mesh'] = os.path.join(mesh_root, r)
        sel = t[t[:, 0] == ri]; Y, X = sel[:, 1].astype(np.int64), sel[:, 2].astype(np.int64)
        okA, okB, dmin = check_tiles(check, Y, X, w)
        res[r] = dict(tiles=int(len(Y)), ok_a=int(okA.sum()), ok_b=int(okB.sum()), min_mm=round(float(dmin.min()), 3) if len(Y) else None)
        bad += int((~(okA & okB)).sum()); print(f'{r}: {res[r]}', flush=True)
    print(json.dumps(dict(tiles=int(len(t)), failed=bad, verdict='ALL ALLOWED' if bad == 0 else 'FAILED')), flush=True)
    return bad == 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=('recompute', 'verify')); ap.add_argument('a'); ap.add_argument('b')
    ap.add_argument('--lane', default='self'); ap.add_argument('--mesh-root', default=''); ap.add_argument('--zone-dir', default='')
    ap.add_argument('--p3', default='/home/liam/projects/vesuvius-work/.claude/worktrees/leadA-push3/x_push3')
    a = ap.parse_args()
    if a.mode == 'recompute':
        r = recompute(a.a, a.b, a.lane, a.p3); sys.exit(0 if r['all_allowed'] else 1)
    sys.exit(0 if verify(a.a, a.b, a.mesh_root, a.zone_dir) else 1)
