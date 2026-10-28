"""seam_test.py : does the PHerc0306B user row (A) sit on a papyrus STRIP SEAM?

A recto is built from horizontal strips laid edge to edge, with joins running along the fibres every ~1-3 cm.  A model that
reacts to a join would draw one straight row of marks along the fibres on intact papyrus.  For a row (centre, direction u,
s-span) we resample CT layers d-4..+4 in row coordinates (s along the row, t across it, +-6 mm) and compute, per depth:
  edge   E(t) = mean over s of |dI/dt| after smoothing I along s (sigma 0.5 mm); E_rel = E / median(E) over the window
  steps  for density (CT mean), structure-tensor coherence and fibre orientation (cos 2theta, sin 2theta): at each t0 the
         difference between the 1.5 mm windows above and below, per 1 mm s-bin, as a t-statistic across the bins
         (a join changes the texture consistently all along the row)
Seam evidence near the row = max E_rel and max |step t| within |t| <= 2.5 mm (a seam at the row or one bracketing it).
Compared with PHerc0343's text lines (placed from the x_px343 stroke labels: lines tilted -7.5 deg, 5.7 mm apart) and with a
null of random positions with the same direction and length on each patch (rows excluded).
"""
import json, os, sys
import numpy as np, cv2, zarr
cv2.setNumThreads(int(os.environ.get('OMP_NUM_THREADS', '4')))
W = '/home/liam/projects/vesuvius-work'; UM = 8.64; PX = 1000 / UM
DS = range(-4, 5); T_MM = 6.0; NEAR = 2.5; WSTEP = 1.5
rng = np.random.default_rng(0)


def resample(L, m, u, s0, s1):
    """L: (n_d, H, W) layers; returns (n_d, nt, ns) in row coords (rows = t from +T to -T), s/t axes in mm"""
    vv = np.array([u[1], -u[0]])                       # +t = image 'up' relative to the row (sign only matters for display)
    s = np.arange(s0 * PX, s1 * PX, 1, dtype=np.float32); t = np.arange(T_MM * PX, -T_MM * PX, -1, dtype=np.float32)
    Sg, Tg = np.meshgrid(s, t)
    X = (m[0] + Sg * u[0] + Tg * vv[0]).astype(np.float32); Y = (m[1] + Sg * u[1] + Tg * vv[1]).astype(np.float32)
    return np.stack([cv2.remap(l, X, Y, cv2.INTER_LINEAR, borderValue=0) for l in L]), s / PX, t / PX


def features(I, V):
    k = int(0.5 * PX) * 6 + 1
    Ism = cv2.GaussianBlur(I, (k, 3), sigmaX=0.5 * PX, sigmaY=1.0)
    G = np.abs(np.gradient(Ism, axis=0))
    Vs = cv2.erode(V.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    E = np.where(Vs.sum(1) > 0.5 * V.shape[1], (G * Vs).sum(1) / np.maximum(Vs.sum(1), 1), np.nan)
    Ig = cv2.GaussianBlur(I, (0, 0), 1.5); gx = cv2.Sobel(Ig, cv2.CV_32F, 1, 0); gy = cv2.Sobel(Ig, cv2.CV_32F, 0, 1)
    jxx, jyy, jxy = (cv2.GaussianBlur(a, (0, 0), 6) for a in (gx * gx, gy * gy, gx * gy))
    tr = jxx + jyy + 1e-6; coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / tr
    c2, s2 = (jxx - jyy) / tr, 2 * jxy / tr
    dens = cv2.GaussianBlur(I, (0, 0), 6)
    return E, dict(density=dens, coherence=coh, cos2=c2, sin2=s2), Vs


def step_t(F, Vs, t_mm, s_mm):
    """t-statistic across 1 mm s-bins of (mean above t0) - (mean below t0), 1.5 mm windows; returns array over t"""
    w = int(WSTEP * PX); nb = int((s_mm[-1] - s_mm[0]) // 1.0)
    edges = np.linspace(0, F.shape[1], nb + 1).astype(int)
    Fm = np.where(Vs, F, np.nan)
    col = np.stack([np.nanmean(Fm[:, a:b], 1) for a, b in zip(edges[:-1], edges[1:])], 1)      # (nt, nb)
    out = np.full(F.shape[0], np.nan)
    cs = np.nancumsum(np.nan_to_num(col), 0); cn = np.cumsum(np.isfinite(col), 0)
    for i in range(w, F.shape[0] - w, 4):
        up = (cs[i] - cs[i - w]) / np.maximum(cn[i] - cn[i - w], 1); dn = (cs[i + w] - cs[i]) / np.maximum(cn[i + w] - cn[i], 1)
        ok = (cn[i] - cn[i - w] > 0.5 * w) & (cn[i + w] - cn[i] > 0.5 * w)
        if ok.sum() < 5: continue
        d = (up - dn)[ok]; out[i] = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-9)
    return out


def measure(L, m, u, s0, s1):
    R, s_mm, t_mm = resample(L, m, u, s0, s1)
    near = np.abs(t_mm) <= NEAR; at = np.abs(t_mm) <= 0.5
    per_d = []; prof = {}
    for j, I in enumerate(R):
        V = I > 0
        if V.mean() < 0.4: per_d.append(None); continue
        E, F, Vs = features(I, V)
        Er = E / np.nanmedian(E)
        row = dict(E_rel_near=float(np.nanmax(Er[near])), E_rel_at=float(np.nanmax(Er[at])), t_E=float(t_mm[near][np.nanargmax(Er[near])]))
        for k, f in F.items():
            st = np.abs(step_t(f, Vs, t_mm, s_mm))
            row[f'step_{k}'] = float(np.nanmax(st[near])) if np.isfinite(st[near]).any() else np.nan
            row[f'step_{k}_far'] = float(np.nanmedian(st[~near])) if np.isfinite(st[~near]).any() else np.nan
        row['step_max'] = max(row[f'step_{k}'] for k in F)
        per_d.append(row)
        if j == len(R) // 2:
            prof = dict(t_mm=t_mm[::8].round(3).tolist(), E_rel=np.round(Er[::8], 3).tolist(),
                        step_density=np.round(step_t(F['density'], Vs, t_mm, s_mm)[::8], 2).tolist(),
                        step_coherence=np.round(step_t(F['coherence'], Vs, t_mm, s_mm)[::8], 2).tolist())
    ok = [r for r in per_d if r]
    if len(ok) < 5: return None
    agg = {k: float(np.nanmedian([r[k] for r in ok])) for k in ok[0]}
    return dict(agg=agg, per_depth=per_d, profile_d0=prof)


def load_layers(render, c):
    R = zarr.open(render, mode='r')['0']
    return [np.asarray(R[c + d]).astype(np.float32) for d in DS]


def null(L, centre_fn, u, length, n=40, tries=400, exclude=None):
    out = []
    for _ in range(tries):
        if len(out) >= n: break
        m = centre_fn()
        if exclude and exclude(m): continue
        r = measure(L, m, u, -length / 2, length / 2)
        if r: out.append(dict(centre=m.tolist(), **r['agg']))
    return out


def pct(v, arr):
    a = np.array([x for x in arr if np.isfinite(x)]); return round(float((a < v).mean() * 100), 1) if len(a) else None


def main():
    res = {}
    # A: own-sheet patch, row fit from own_patch.py
    ro = json.load(open(f'{W}/x_focus/out/A_own/own_patch.json'))['row_fit']
    m = np.array(ro['centre_px']); u = np.array(ro['u']); s0, s1 = ro['s_range_mm']
    LA = load_layers(f'{W}/x_focus/renders/A_own_r41.zarr', 20)
    mA = m + u * PX * (s0 + s1) / 2; LEN = s1 - s0
    res['A_row'] = measure(LA, mA, u, -LEN / 2, LEN / 2)
    vvA = np.array([-u[1], u[0]]); H, Wd = LA[0].shape; ok0 = LA[4] > 0
    def cA():
        while True:
            p = np.array([rng.uniform(0, Wd), rng.uniform(0, H)])
            if ok0[int(p[1]), int(p[0])]: return p
    def exA(p):
        d = p - m; t = abs(d @ vvA) / PX; s = d @ u / PX
        return t < 4.0 and -12 < s < 16                                     # skip the row and its continuation
    res['A_null'] = null(LA, cA, u, LEN, exclude=exA)
    del LA
    # PHerc0343: Erwin's segment, lines from the x_px343 stroke labels (tilt -7.5 deg)
    L3 = load_layers(f'{W}/fl/0343/concat_66.zarr', 33)
    a = np.radians(-7.5); u3 = np.array([np.cos(a), np.sin(a)])
    lines = {'343_line1': (np.array([3270.0, 1556 + np.tan(a) * 3270]), (5336 - 1206) / PX),
             '343_line2': (np.array([2015.0, 2220 + np.tan(a) * 2015]), (2357 - 1672) / PX)}
    for k, (mc, ln) in lines.items():
        res[k] = measure(L3, mc, u3, -ln / 2, ln / 2); res[k]['length_mm'] = round(ln, 1)
    ok3 = L3[4] > 0; H3, W3 = ok3.shape; vv3 = np.array([-u3[1], u3[0]])
    def c3():
        while True:
            p = np.array([rng.uniform(0, W3), rng.uniform(0, H3)])
            if ok3[int(p[1]), int(p[0])]: return p
    def ex3(p):
        return any(abs((p - mc) @ vv3) / PX < 3.0 for mc, _ in lines.values())
    res['343_null'] = null(L3, c3, u3, min(LEN, 12.0), n=30, exclude=ex3)
    # summary
    keys = ['E_rel_near', 'E_rel_at', 'step_density', 'step_coherence', 'step_cos2', 'step_sin2', 'step_max']
    summ = {}
    for name, nul in (('A_row', 'A_null'), ('343_line1', '343_null'), ('343_line2', '343_null')):
        if not res.get(name): continue
        g = res[name]['agg']
        summ[name] = {k: dict(value=round(g[k], 3), pct_vs_own_null=pct(g[k], [x[k] for x in res[nul]]),
                              pct_vs_A_null=pct(g[k], [x[k] for x in res['A_null']])) for k in keys}
        summ[name]['t_E_mm'] = round(g['t_E'], 2)
    for nul in ('A_null', '343_null'):
        summ[nul] = {k: [round(float(np.nanpercentile([x[k] for x in res[nul]], q)), 3) for q in (50, 95)] for k in keys}
        summ[nul]['n'] = len(res[nul])
    res['summary'] = summ
    os.makedirs(f'{W}/x_focus/out/seam', exist_ok=True)
    json.dump(res, open(f'{W}/x_focus/out/seam/seam_test.json', 'w'), indent=1, default=lambda o: None)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
