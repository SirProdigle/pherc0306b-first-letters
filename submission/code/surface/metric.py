"""metric.py MESHDIR [MESHDIR ...] : flattening distortion of carrier-grid meshes (method of research lane Q5, q5_metric.py, extended
with the area ratio).  The render's (u, v) is the grid index (20 vx = 0.1728 mm per cell), so a letter's drawn size is distorted by
the mesh metric: stretch_u = |dP/dcol| / 20, stretch_v = |dP/drow| / 20, shear = angle between the tangents - 90 deg, area ratio =
|dP/dcol x dP/drow| / 400 (3-D area per drawn area; > 1 = papyrus squeezed in the image).  Per region: p5 / p50 / p95 and the share
of cells outside +-5 % (the coordinator's correction threshold) and outside +-10 % / 10 deg."""
import sys, json, os
import numpy as np, tifffile
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import p1geom as g                                                                   # noqa: E402
REG = {'claim rect': (-19.9, 12.3, -7.8, 4.6), 'L line band': (-15, 12, -2.5, 2.5), 'RA rows above': (-20, 15, -8, -2.5),
       'CA column above': (-30, 25, -20, -8), 'RB rows below': (-20, 15, 2.5, 8), 'CB column below': (-30, 25, 8, 20),
       'LL line row left': (-35, -20, -4, 4), 'LR line row right': (12, 30, -4, 4)}         # P1's pre-registered regions (evalP1.REG)
U0, V0 = -36.0, -21.0
REG2 = dict(REG); REG2['right group'] = (15.5, 23.5, -2.7, 1.6); REG2['rect line+right+row2'] = (-12.2, 23.54, -6.47, 2.53)
CELL = g.G / g.K


def metric(P):
    du = (P[:, 1:] - P[:, :-1])[:-1]; dv = (P[1:] - P[:-1])[:, :-1]
    su = np.linalg.norm(du, axis=-1) / 20; sv = np.linalg.norm(dv, axis=-1) / 20
    cr = np.linalg.norm(np.cross(du, dv), axis=-1) / 400
    cos = (du * dv).sum(-1) / np.maximum(np.linalg.norm(du, axis=-1) * np.linalg.norm(dv, axis=-1), 1e-9)
    return su, sv, np.degrees(np.arcsin(np.clip(cos, -1, 1))), cr


def summary(su, sv, sh, ar, m):
    f = lambda x: [round(float(np.percentile(x[m], q)), 3) for q in (5, 50, 95)]
    o5 = ((np.abs(su[m] - 1) > .05) | (np.abs(sv[m] - 1) > .05)).mean()
    o10 = ((np.abs(su[m] - 1) > .1) | (np.abs(sv[m] - 1) > .1) | (np.abs(sh[m]) > 10)).mean()
    return {'stretch_u': f(su), 'stretch_v': f(sv), 'shear_deg': f(sh), 'area': f(ar), 'share_out_5pct': round(float(o5), 3),
            'share_out_10pct_10deg': round(float(o10), 3), 'n': int(m.sum())}


if __name__ == '__main__':
    out = {}
    for d in sys.argv[1:]:
        P = g.load(d); su, sv, sh, ar = metric(P)
        ok = np.isfinite(su) & np.isfinite(sv)
        H, W = su.shape; jj, ii = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        Uc, Vc = -36.0 + jj * CELL, -21.0 + ii * CELL
        r = {}
        for k, (u0, u1, v0, v1) in REG2.items():
            m = ok & (Uc >= u0) & (Uc < u1) & (Vc >= v0) & (Vc < v1)
            if m.sum() >= 20: r[k] = summary(su, sv, sh, ar, m)
        out[os.path.basename(os.path.normpath(d))] = r
        print(d); [print('  ', k, json.dumps(x)) for k, x in r.items() if k in ('right group', 'LR line row right', 'claim rect', 'rect line+right+row2', 'L line band')]
    json.dump(out, open(f'{g.T}/eval/metric.json', 'w'), indent=1)
