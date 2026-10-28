"""track.py : CT-driven sheet tracking over lead A's column (lane P1).  Builds the refined surfaces V1 (trackA) and V2 (trackP)
on the claim-frame carrier grid (carrier.py), blind to ink.

Profiles prof[i, j, t] (CT along the carrier normal, t = -T..+T vx, 3x3 footprint mean) are smoothed in depth (Gaussian sigma
--dsig, default 2 vx: merges the two fibre layers of a sheet into one hump, N4) and lightly in-plane (sigma --psig nodes).
Sheet candidates = strict local maxima along t with prominence >= --minprom grey levels.

Seeds: nodes of the line band (|v| <= 2.5 mm, u -15..+12) where A, big (and R1 where present) cross the normal within 3 vx of each
other; the seed's sheet = the candidate peak nearest to A's crossing within [t_A - 6, t_A + 2].  delta = median(t_A - t_peak) over
the seeds (A's reading-face offset from the sheet's density peak, preserved everywhere).

Best-first propagation (a priority queue on peak prominence): a frontier node predicts t from a weighted local plane through the
assigned nodes within 3 cells (confidence- and distance-weighted), then takes the candidate peak within +-W vx (--win, default 4;
the sheet pitch is 16-22 vx, so one step can never hop a sheet) maximising prominence - 0.5 |dt|; with no candidate it coasts on
the prediction (low confidence) for at most --maxcoast steps from the last real peak.  Unreached nodes are filled by normalised
convolution of the tracked field.

Outputs (OUT/): track.npz (t_sheet, conf, coast, delta, seeds, t_final_V1, t_final_V2, weights) and two tifxyz meshes
V1_trackA/, V2_trackP/ (x/y/z.tif on the carrier grid, scale 0.05, meta.json), position = carrier P + t_final * n.
V1 keeps the ORIGINAL surface (A, else w074big) wherever it lies within --keep vx of tracked + delta (feathered 0.5 mm), and uses the
tracked surface elsewhere.  Both: in-plane 3x3 median + Gaussian sigma --fsig nodes on the final offset.
"""
import os, sys, json, heapq, argparse, time
import numpy as np, tifffile
from scipy import ndimage as ndi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p1geom as g                                                                   # noqa: E402


def candidates(ps, minprom, span=8):
    """ps (NR, NC, D) smoothed profiles -> list per node of (t_index, prominence) via strict local maxima"""
    D = ps.shape[-1]
    left = np.roll(ps, 1, -1); right = np.roll(ps, -1, -1)
    ismax = (ps > left) & (ps > right); ismax[..., 0] = False; ismax[..., -1] = False
    # prominence: peak - max(min over the span to the left, min over the span to the right)
    from scipy.ndimage import minimum_filter1d
    mins = minimum_filter1d(ps, size=span + 1, axis=-1, origin=span // 2)      # min over [k-span, k]
    mins_r = minimum_filter1d(ps, size=span + 1, axis=-1, origin=-(span // 2))  # min over [k, k+span]
    prom = ps - np.maximum(mins, mins_r)
    ok = ismax & (prom >= minprom) & np.isfinite(ps)
    return ok, prom


def subpix(ps, i, j, k):
    D = ps.shape[-1]
    if k <= 0 or k >= D - 1: return float(k)
    a, b, c = ps[i, j, k - 1], ps[i, j, k], ps[i, j, k + 1]
    den = a - 2 * b + c
    return float(k + np.clip(0.5 * (a - c) / den, -0.5, 0.5)) if abs(den) > 1e-6 else float(k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--carrier', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--prof', default='prof.npy', help='profile file inside the carrier dir (prof.npy = CT, prof_m7.npy = m7)')
    ap.add_argument('--dsig', type=float, default=2.0); ap.add_argument('--psig', type=float, default=0.7)
    ap.add_argument('--minprom', type=float, default=2.0); ap.add_argument('--win', type=float, default=4.0)
    ap.add_argument('--maxcoast', type=int, default=12); ap.add_argument('--keep', type=float, default=3.0)
    ap.add_argument('--fsig', type=float, default=1.0); ap.add_argument('--tag', default='')
    ap.add_argument('--delta', type=float, default=None, help='override delta (else measured on the seeds)')
    ap.add_argument('--seed-target', type=float, default=-1.7, help='expected sheet-peak offset from A at the seeds (CT: -1.7, N4)')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    C = np.load(f'{a.carrier}/carrier.npz')
    P, n, ok, U, Vv = C['P'], C['n'], C['ok'], C['u'], C['v']
    prof = np.load(f'{a.carrier}/{a.prof}').astype(np.float32); ts = np.load(f'{a.carrier}/prof_ts.npy')
    T0 = -ts[0]; NR, NC, D = prof.shape
    valid_t = (prof > 0) if 'm7' not in a.prof else np.broadcast_to(ok[..., None], prof.shape)   # m7: 0 = no sheet, a valid value
    pf = np.where(valid_t, prof, np.nan)
    # fill NaN with the node mean before smoothing (masked areas = outside the scroll mask)
    fillv = np.nanmean(pf, -1, keepdims=True); pf = np.where(np.isfinite(pf), pf, np.nan_to_num(fillv))
    ps = ndi.gaussian_filter1d(pf, a.dsig, axis=-1)
    if a.psig > 0: ps = ndi.gaussian_filter(ps, (a.psig, a.psig, 0))
    cand, prom = candidates(ps, a.minprom)
    print('candidate peaks per node: median', np.median(cand.sum(-1)), flush=True)

    xA, xB = C['x_A'], C['x_big']; xR = C['x_R1']
    # ---- seeds
    band = (np.abs(Vv) <= 2.5) & (U >= -15) & (U <= 12) & ok
    agreeAB = np.isfinite(xA) & np.isfinite(xB) & (np.abs(xA - xB) <= 3)
    agreeR = ~np.isfinite(xR) | (np.abs(xR - xA) <= 3)
    sd = band & agreeAB & agreeR
    seeds = []; dl = []
    for i, j in zip(*np.nonzero(sd)):
        ks = np.nonzero(cand[i, j])[0]
        if not len(ks): continue
        tk = ks - T0
        sel = (tk >= xA[i, j] - 6) & (tk <= xA[i, j] + (2 if a.seed_target < 0 else 6))
        if not sel.any(): continue
        k = ks[sel][np.argmin(np.abs(tk[sel] - (xA[i, j] + a.seed_target)))]
        tp = subpix(ps, i, j, k) - T0
        seeds.append((i, j, tp, float(prom[i, j, k]))); dl.append(xA[i, j] - tp)
    delta = float(np.median(dl)) if a.delta is None else a.delta
    print(f'seeds {len(seeds)} of {sd.sum()} consensus nodes ({band.sum()} band nodes); delta = {delta:.2f} vx '
          f'(IQR {np.percentile(dl, 25):.2f}..{np.percentile(dl, 75):.2f})', flush=True)

    # ---- best-first propagation
    t = np.full((NR, NC), np.nan); conf = np.zeros((NR, NC)); coast = np.zeros((NR, NC), int)
    heap = []
    for i, j, tp, pr in seeds:
        t[i, j] = tp; conf[i, j] = pr
    nb8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    R = 3; yy, xx = np.mgrid[-R:R + 1, -R:R + 1]; wd = np.exp(-(yy ** 2 + xx ** 2) / (2 * 1.5 ** 2))

    def predict(i, j):
        i0, i1, j0, j1 = max(0, i - R), min(NR, i + R + 1), max(0, j - R), min(NC, j + R + 1)
        tt = t[i0:i1, j0:j1]; m = np.isfinite(tt)
        if not m.any(): return None, 0
        w = (wd[i0 - i + R:i1 - i + R, j0 - j + R:j1 - j + R] * np.maximum(conf[i0:i1, j0:j1], 0.3))[m]
        di = (np.arange(i0, i1)[:, None] - i + 0 * tt)[m]; dj = (np.arange(j0, j1)[None, :] - j + 0 * tt)[m]
        y = tt[m]
        if m.sum() >= 4 and np.ptp(di) + np.ptp(dj) >= 2:
            A_ = np.stack([np.ones_like(di), di, dj], 1) * np.sqrt(w)[:, None]
            coef, *_ = np.linalg.lstsq(A_, y * np.sqrt(w), rcond=None)
            # clamp the extrapolated slope (|grad| <= 1 vx per vx = 20 vx per cell)
            pred = coef[0]
            if not np.isfinite(pred) or abs(pred - np.average(y, weights=w)) > 25: pred = np.average(y, weights=w)
        else:
            pred = np.average(y, weights=w)
        return float(pred), float(np.average(conf[i0:i1, j0:j1][m], weights=w))

    def evaluate(i, j):
        pred, c0 = predict(i, j)
        if pred is None: return None
        ks = np.nonzero(cand[i, j])[0]; tk = ks - T0
        sel = np.abs(tk - pred) <= a.win
        if sel.any():
            sc = prom[i, j, ks[sel]] - 0.5 * np.abs(tk[sel] - pred)
            k = ks[sel][np.argmax(sc)]
            return subpix(ps, i, j, k) - T0, float(prom[i, j, k]), 0
        # coast
        cn = [coast[i + di, j + dj] for di, dj in nb8 if 0 <= i + di < NR and 0 <= j + dj < NC and np.isfinite(t[i + di, j + dj])]
        cc = min(cn) + 1 if cn else 99
        if cc > a.maxcoast: return None
        return pred, 0.3 * min(c0, 5.0) / (1 + cc), cc

    def push_nbrs(i, j):
        for di, dj in nb8:
            ii, jj = i + di, j + dj
            if 0 <= ii < NR and 0 <= jj < NC and ok[ii, jj] and not np.isfinite(t[ii, jj]):
                r = evaluate(ii, jj)
                if r is not None: heapq.heappush(heap, (-r[1], ii, jj))
    for i, j, _, _ in seeds: push_nbrs(i, j)
    t_start = time.time(); nassign = 0
    while heap:
        negs, i, j = heapq.heappop(heap)
        if np.isfinite(t[i, j]): continue
        r = evaluate(i, j)
        if r is None: continue
        if r[1] < -negs - 1e-6 and heap and -heap[0][0] > r[1]:      # stale priority: re-queue with the fresh score
            heapq.heappush(heap, (-r[1], i, j)); continue
        tt, cf, cc = r
        if abs(tt) > T0 - 4: continue
        t[i, j] = tt; conf[i, j] = cf; coast[i, j] = cc; nassign += 1
        push_nbrs(i, j)
        if nassign % 20000 == 0: print(f'  assigned {nassign}  {time.time() - t_start:.0f}s', flush=True)
    reached = np.isfinite(t)
    print(f'reached {reached.mean():.3f} of the grid; real peaks {(reached & (coast == 0)).mean():.3f}; '
          f'coasting {(reached & (coast > 0)).mean():.3f}', flush=True)
    # fill unreached by normalised convolution (2 mm), then wider
    w = reached.astype(float) * np.maximum(conf, 0.1)
    tf = t.copy()
    for sig in (2.0, 6.0, 20.0):
        sg = sig / C['CELL'] if False else sig / float(C['CELL'])
        num = ndi.gaussian_filter(np.nan_to_num(t) * w, sg); den = ndi.gaussian_filter(w, sg)
        fill = num / np.maximum(den, 1e-9); tf = np.where(np.isfinite(tf), tf, np.where(den > 1e-3, fill, np.nan))
    tf = np.nan_to_num(tf)
    sheet = tf + delta                                                     # V2 raw offset (reading face)
    # V1: original surface where it lies on the tracked sheet
    orig = np.where(np.isfinite(xA), xA, xB)
    src = np.where(np.isfinite(xA), 1, np.where(np.isfinite(xB), 2, 0))
    keep = np.isfinite(orig) & (np.abs(orig - sheet) <= a.keep)
    keep_o = ndi.binary_opening(keep, iterations=1)
    wk = ndi.gaussian_filter(keep_o.astype(float), 0.5 / float(C['CELL']) / 2)
    wk = np.where(keep_o, np.maximum(wk, 0.5), wk) * np.isfinite(orig)
    v1 = np.where(np.isfinite(orig), wk * np.nan_to_num(orig) + (1 - wk) * sheet, sheet)

    def finish(f):
        f = ndi.median_filter(f, size=3)
        return ndi.gaussian_filter(f, a.fsig) if a.fsig > 0 else f
    # V0: the ORIGINAL surfaces (A where it crosses the normal, else w074big) resampled on the same grid = the control arm
    of = np.isfinite(orig); wo = of.astype(float); v0 = np.where(of, orig, np.nan)
    for sig in (0.5, 2.0, 8.0):
        sg = sig / float(C['CELL'])
        num = ndi.gaussian_filter(np.nan_to_num(orig) * wo, sg); den = ndi.gaussian_filter(wo, sg)
        v0 = np.where(np.isfinite(v0), v0, np.where(den > 1e-3, num / np.maximum(den, 1e-9), np.nan))
    v0 = np.where(np.isfinite(v0), v0, sheet)
    out = {'V0_orig': v0, 'V1_trackA': finish(v1), 'V2_trackP': finish(sheet)}
    np.savez_compressed(f'{a.out}/track{a.tag}.npz', t_sheet=t.astype(np.float32), t_fill=tf.astype(np.float32), conf=conf.astype(np.float32),
                        coast=coast.astype(np.int16), delta=delta, seeds=np.array([(s[0], s[1], s[2]) for s in seeds]),
                        keep=keep, wk=wk.astype(np.float32), orig=orig.astype(np.float32), src=src.astype(np.int8),
                        **{k: v.astype(np.float32) for k, v in out.items()}, params=json.dumps(vars(a)))
    meta0 = json.load(open(f'{g.SURF["A"]}/meta.json'))
    for name, f in out.items():
        d = f'{a.out}/{name}{a.tag}'; os.makedirs(d, exist_ok=True)
        Q = P + f[..., None] * n
        Q[~ok] = -1
        for k, c in enumerate('xyz'):
            tifffile.imwrite(f'{d}/{c}.tif', Q[..., k].astype(np.float32))
        m = dict(meta0); m['uuid'] = m['name'] = os.path.basename(d); m['source'] = 'P1_surface_refine/track.py'
        m['bbox'] = [np.nanmin(Q[ok], 0).tolist(), np.nanmax(Q[ok], 0).tolist()]
        m['p1'] = dict(variant=name, carrier=os.path.abspath(a.carrier), grid_frame=dict(U0=float(C['U0']), V0=float(C['V0']), cell_mm=float(C['CELL'])),
                       delta=delta, params=vars(a), rule='vertex = carrier P + offset * carrier normal; offset = tracked CT sheet peak + delta'
                       + (' (original A/big kept where within keep vx)' if name.startswith('V1') else ''))
        for k_ in ('vc_gsfs_params', 'vc_gsfs_mode', 'elapsed_time_s', 'max_gen', 'seed', 'area_vx2', 'area_cm2'): m.pop(k_, None)
        json.dump(m, open(f'{d}/meta.json', 'w'), indent=1)
        print('wrote', d, flush=True)
    # stats
    dv = lambda f: np.abs(np.diff(f, axis=0)); dh = lambda f: np.abs(np.diff(f, axis=1))
    for name, f in out.items():
        e = np.concatenate([dv(f).ravel(), dh(f).ravel()])
        print(f'{name}: |offset| median {np.median(np.abs(f)):.1f} vx; edges with |d| > 4 vx {np.mean(e > 4):.4f}; '
              f'kept original {wk.mean() if name.startswith("V1") else 0:.3f}')


if __name__ == '__main__':
    main()
