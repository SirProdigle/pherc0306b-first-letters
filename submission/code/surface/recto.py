"""recto.py ARM [ARM ...] : label-free recto-fibre score H (method of research lane Q5, x_refine/research/Q5_surface_for_ink/q5_recto.py,
re-run here so P1's adoption decision is reproducible from P1's code).  No ink, no labels.

On the recto the writing runs along the horizontal fibres = along u.  Per ~1 mm tile (116 px) of CT layer 20 + k of the arm's render,
gradient orientations theta weighted by gradient energy e (Gaussian 1 px, Sobel): H = -Re(sum e e^{2i theta}) / sum e  (+1 = all
structure runs along u = the recto family; -1 = along v = verso).  Validated by Q5 on PHerc1447: ink tiles .11-.16 at the surface vs
-.05/-.135 on the verso.  Reported per region at k = 0 (the surface): median H and the share of tiles with H > 0.15."""
import sys, json
import numpy as np, zarr, cv2
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import p1geom as g                                                                   # noqa: E402
REG = {'claim rect': (-19.9, 12.3, -7.8, 4.6), 'L line band': (-15, 12, -2.5, 2.5), 'RA rows above': (-20, 15, -8, -2.5),
       'CA column above': (-30, 25, -20, -8), 'RB rows below': (-20, 15, 2.5, 8), 'CB column below': (-30, 25, 8, 20),
       'LL line row left': (-35, -20, -4, 4), 'LR line row right': (12, 30, -4, 4)}         # P1's pre-registered regions (evalP1.REG)
U0, V0 = -36.0, -21.0
cv2.setNumThreads(4)
TP = 116
REG2 = dict(REG); REG2['right group'] = (15.5, 23.5, -2.7, 1.6)


def tile_scores(img):
    q = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.0)
    gx = cv2.Sobel(q, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(q, cv2.CV_32F, 0, 1, ksize=3)
    e = gx * gx + gy * gy; th = np.arctan2(gy, gx); e = np.where(img > 0, e, 0)
    c2 = e * np.cos(2 * th); H, W = img.shape; gh, gw = H // TP, W // TP
    ts = lambda a: a[:gh * TP, :gw * TP].reshape(gh, TP, gw, TP).sum((1, 3))
    E, C2 = ts(e), ts(c2); Vf = ts((img > 0).astype(np.float32)) / (TP * TP)
    return np.where(Vf > 0.8, -C2 / np.maximum(E, 1e-6), np.nan)


out = {}
for arm in sys.argv[1:]:
    arm, _, rpath = arm.partition('=')                                              # I2: ARM=RENDER.zarr (else g.T/renders/ARM.zarr)
    Hm = tile_scores(np.asarray(zarr.open(rpath or f'{g.T}/renders/{arm}.zarr', mode='r')['0'][20]))
    gh, gw = Hm.shape; UU, VV = np.meshgrid(U0 + (np.arange(gw) + 0.5) * TP / g.K, V0 + (np.arange(gh) + 0.5) * TP / g.K)
    r = {}
    for k, (u0, u1, v0, v1) in REG2.items():
        m = (UU >= u0) & (UU < u1) & (VV >= v0) & (VV < v1); h = Hm[m]; h = h[np.isfinite(h)]
        r[k] = dict(n=int(h.size), H_med=round(float(np.median(h)), 3) if h.size else None, share_gt015=round(float((h > 0.15).mean()), 3) if h.size else None)
    out[arm] = r; np.save(f'{g.T}/eval/recto_{arm}.npy', Hm.astype(np.float32))
    print(arm); [print(f'   {k:18s} n {x["n"]:4d}  H {x["H_med"]}  share>.15 {x["share_gt015"]}') for k, x in r.items()]
json.dump(out, open(f'{g.T}/eval/recto_{"_".join(out)}.json', 'w'), indent=1)
