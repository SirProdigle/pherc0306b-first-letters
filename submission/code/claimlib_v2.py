"""claimlib_v2.py : the pre-registered claim display of the v2 draft (lane I2).

Every function below is copied VERBATIM from lane I1 (x_refine/I1_display_read/i1lib.py and build_claim_v2.py, 15 Oct 2026),
which implements PREREG Amendment 1 A1.1 + A3 (x_refine/trials/I1_display_read/PREREG.md):
  fused z  = mean of per-model z, each z over the on-sheet pixels of the CONTEXT view (u -30..+30 x v -20..+15 mm);
  H = p99, P = p99.9 of fused z over the on-sheet pixels of the CLAIM BOX; L = 0.35 H (primary) or 0.5 H (secondary);
  darkness d = 0.85 clip((z - L) / (H - L), 0, 1) + 0.15 clip((z - H) / (P - H), 0, 1);
  pixel = bg (1 - d) + INK d, bg = PAGE (0.75 + 0.25 ct_norm) (surface CT layer, linear 1-99.5 % stretch, no CLAHE);
  off-sheet = flat blue-grey, small holes (< 300 px) = plain page.
Only the module layout differs (no dependency on x_submit/make_submission_A), so the package runs on its own.
"""
import os, json, hashlib
os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np, cv2
from scipy import ndimage as ndi
cv2.setNumThreads(4)

UM = 8.64; K = 1000.0 / UM
PX_UM = 10.0
BOX = dict(u0=-15.0, u1=25.0, v0=-9.0, v1=6.0)
PAGE = np.array([243, 236, 222], np.float32); INK = np.array([22, 20, 18], np.float32)
P1U0, P1V0 = -36.0, -21.0                                                         # carrier raster origin (mm): u = -36 + x / K, v = -21 + y / K
CTX = dict(u0=-30.0, u1=30.0, v0=-20.0, v1=15.0)                                  # context view (claim frame mm)
CBOX = dict(u0=-15.0, u1=23.0, v0=-6.84, v1=2.72)                                 # claim box, version 4c: drawn by the team leader on the model image
#   (v4b: u -15..+25 x v -7.35..+2.525; v4: u -15..+25 x v -5..+4.875, the pre-registered model-only box search;
#    v2 / v3: u -12.1..+23.5 x v -8.55..+2.5, PREREG A1.3 + Amendment 4)
RUNGS = {'primary': 0.35, 'secondary': 0.50, 'low': 0.20}
OFFQ = np.array([150, 160, 175], np.float32)                                      # off-sheet: blue-grey, never ink-like (A3)
UMX = 8.64; PPMX = 1000 / UMX
HDR_SCALE = 2.2

def grid_uv(box=BOX, px_um=PX_UM):
    uu = np.arange(box['u0'], box['u1'], px_um / 1000.0) + px_um / 2000.0
    vv = np.arange(box['v0'], box['v1'], px_um / 1000.0) + px_um / 2000.0
    return np.meshgrid(uu.astype(np.float32), vv.astype(np.float32))


def sample_raster(img, box=BOX, px_um=PX_UM, nn=False):
    """P1 carrier raster (u = -36 + x/K, v = -21 + y/K) -> claim-frame box."""
    U, V = grid_uv(box, px_um)
    X = (U - P1U0) * K - 0.5; Y = (V - P1V0) * K - 0.5
    return cv2.remap(np.ascontiguousarray(img), X.astype(np.float32), Y.astype(np.float32),
                     cv2.INTER_NEAREST if nn else cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def zs(a, m):
    x = a[m].astype(np.float64); return ((a - x.mean()) / (x.std() + 1e-9)).astype(np.float32)


def q4_levels(z, box_valid, frac):
    H = float(np.percentile(z[box_valid], 99)); P = float(np.percentile(z[box_valid], 99.9))
    return dict(L=frac * H, H=H, P=max(P, H + 1e-3), frac=frac)


def q4_tone(z, lv, valid):
    d = 0.85 * np.clip((z - lv['L']) / (lv['H'] - lv['L']), 0, 1) + 0.15 * np.clip((z - lv['H']) / (lv['P'] - lv['H']), 0, 1)
    return np.where(valid, d, 0).astype(np.float32)


def light_ct_bg(ct, valid_ref, valid):
    x = ct[valid_ref & (ct > 0)] if (valid_ref & (ct > 0)).any() else ct[valid]
    lo, hi = np.percentile(x, [1, 99.5]); cn = np.clip((ct - lo) / (hi - lo + 1e-6), 0, 1)
    bg = PAGE[None, None] * (0.75 + 0.25 * cn[..., None])
    bg[~valid] = 60
    return bg.astype(np.float32)


def q4_rgb(d, bg, valid, hole_px=300):
    """A3: small data holes inside the papyrus (< hole_px px) are shown as plain page (no ink, no CT); larger off-sheet areas
    are flat blue-grey, so that no data gap can look like an ink mark."""
    out = bg * (1 - d[..., None]) + INK[None, None] * d[..., None]
    inv = ~valid
    lab, n = ndi.label(inv)
    if n:
        sz = ndi.sum(np.ones_like(lab), lab, range(1, n + 1)); small = np.isin(lab, np.where(sz < hole_px)[0] + 1)
    else:
        small = np.zeros_like(inv)
    out[inv] = OFFQ; out[small] = PAGE * 0.87
    return np.clip(out, 0, 255).astype(np.uint8)


def box_mask(U, V, b):
    return (U >= b['u0']) & (U <= b['u1']) & (V >= b['v0']) & (V <= b['v1'])


def load_raster_map(path, shape):
    """a map on P1's carrier raster; if PATH.json has a 'box' (y0, y1, x0, x1), paste the crop into a zero raster of `shape`."""
    a = np.nan_to_num(np.load(path).astype(np.float32)); j = path[:-4] + '.json'
    if a.shape != tuple(shape) and os.path.exists(j) and 'box' in json.load(open(j)):
        y0, y1, x0, x1 = json.load(open(j))['box']; out = np.zeros(shape, np.float32); out[y0:y1, x0:x1] = a[:y1 - y0, :x1 - x0]; return out
    if a.shape != tuple(shape): raise ValueError(f'{path}: shape {a.shape} is not the carrier raster {shape} and has no box json')
    return a


def load_raster_cov(path, shape):
    """v5: coverage (uint8 0/1 on the carrier raster) of a map whose value 0 is data (e.g. uint8/255 output with exact zeros on blank
    papyrus): finite values, inside the map's crop box when PATH.json has one.  Used instead of 'map != 0' for such maps."""
    a = np.isfinite(np.load(path, mmap_mode='r')).astype(np.uint8); j = path[:-4] + '.json'
    if a.shape != tuple(shape) and os.path.exists(j) and 'box' in json.load(open(j)):
        y0, y1, x0, x1 = json.load(open(j))['box']; out = np.zeros(shape, np.uint8); out[y0:y1, x0:x1] = a[:y1 - y0, :x1 - x0]; return out
    if a.shape != tuple(shape): raise ValueError(f'{path}: shape {a.shape} is not the carrier raster {shape} and has no box json')
    return a


def configure(cfg):
    """v5: take the claim box (and, if given, the context) from a build config (build_v5.py).  Every function above reads the
    module globals at call time, so this is the only change; with no config the values are version 4c's."""
    global CBOX, CTX
    if cfg.get('box'): CBOX = {k: float(cfg['box'][k]) for k in ('u0', 'u1', 'v0', 'v1')}
    if cfg.get('context'): CTX = {k: float(cfg['context'][k]) for k in ('u0', 'u1', 'v0', 'v1')}
    return CBOX


def sha(p):
    h = hashlib.sha256(); f = open(p, 'rb')
    for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()


def rows_margin(d, inbox, ppm):
    """row centres (image rows, reading orientation) from the displayed ink (d >= 0.5) inside the box: profile, peaks >= 3 mm."""
    prof = ((d >= 0.5) & inbox).sum(1).astype(np.float32) / np.maximum(inbox.sum(1), 1)
    prof = cv2.GaussianBlur(prof.reshape(-1, 1), (1, 0), sigmaX=0.1, sigmaY=0.6 * ppm).ravel()
    peaks = []
    for i in np.argsort(-prof):
        if prof[i] < 0.25 * prof.max(): break
        if all(abs(i - p) > 3.0 * ppm for p in peaks): peaks.append(int(i))
    return sorted(peaks), prof


def dashed_rect(img, x0, y0, x1, y1, col, th=4, dash=40, gap=24):
    for (a, b) in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        a = np.array(a, float); b = np.array(b, float); n = np.linalg.norm(b - a); u = (b - a) / max(n, 1)
        t = 0.0
        while t < n:
            p = a + u * t; q = a + u * min(t + dash, n)
            cv2.line(img, tuple(int(v) for v in p), tuple(int(v) for v in q), col, th, cv2.LINE_AA); t += dash + gap


def frame_image(rgb, header, footer, rows=None, margin=220):
    H, Wd = rgb.shape[:2]
    body = np.full((H, Wd + 2 * margin, 3), 255, np.uint8); body[:, margin:margin + Wd] = rgb
    for k, r in enumerate(rows or []):
        yb = int(r + 1.6 * PPMX)                                                     # baseline 1.6 mm under the row centre
        for x0 in (12, margin + Wd + 12):
            cv2.line(body, (x0, yb), (x0 + margin - 40, yb), (200, 120, 0), 5, cv2.LINE_AA)
        cv2.putText(body, f'row {k + 1}', (14, yb - 14), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (200, 120, 0), 3, cv2.LINE_AA)
    lh = int(26 * HDR_SCALE)
    head = np.full((lh * len(header) + 24, body.shape[1], 3), 255, np.uint8)
    for i, t in enumerate(header):
        cv2.putText(head, t, (16, lh * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * HDR_SCALE, (15, 15, 15), 2, cv2.LINE_AA)
    foot = np.full((lh * (len(footer) + 2), body.shape[1], 3), 255, np.uint8); Lb = int(round(10 * PPMX))
    cv2.rectangle(foot, (margin, 18), (margin + Lb, 40), (15, 15, 15), -1)
    cv2.putText(foot, '1 cm', (margin + Lb + 24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * HDR_SCALE, (15, 15, 15), 2, cv2.LINE_AA)
    for i, t in enumerate(footer):
        cv2.putText(foot, t, (16, lh * (i + 2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * HDR_SCALE, (15, 15, 15), 2, cv2.LINE_AA)
    return np.vstack([head, body, foot])
