"""Synthetic Greek-majuscule text and synthetic non-letter artefacts, rendered on the 50 um grid and degraded to
look like ink-model output, then composited into real (negative) a-map windows.

All shapes are drawn on a canvas of CAN x CAN px (50 um/px), rotated by a random angle, and the central WIN x WIN
window is returned, so letters/lines are cut at the window edges like in real maps.
"""
import glob, math, random
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import skeletonize
from common import WIN, UM

CAN = 200          # canvas (10 mm); window = central 128 px after rotation
PXMM = 1000.0 / UM  # px per mm (20)

_FONT_NAMES = """DejaVuSans DejaVuSans-Bold NotoSans-Regular NotoSans-Medium NotoSans-Bold NotoSans-Light NotoSans-Black
NotoSerif-Regular NotoSerif-Medium NotoSerif-Bold NotoSerif-Light LiberationSans-Regular LiberationSans-Bold
LiberationSerif-Regular LiberationSerif-Bold OpenSans-Regular OpenSans-Bold OpenSans-Light Carlito-Regular Carlito-Bold
DroidSans DroidSans-Bold Inter-Regular Inter-Bold Inter-Light C059-Roman C059-Bold P052-Roman NimbusRoman-Regular
NimbusSans-Regular NimbusSans-Bold URWBookman-Light URWBookman-Demi URWGothic-Book URWGothic-Demi Cantarell-Regular
Cantarell-Bold Comfortaa-Regular STIXTwoText-Regular STIXTwoText-Bold AdwaitaSans-Regular""".split()
FONTS = []
for n in _FONT_NAMES:
    fs = [f for f in glob.glob(f'/usr/share/fonts/**/{n}.*', recursive=True) if f.endswith(('.ttf', '.otf'))]
    if fs:
        FONTS.append(fs[0])

# Greek majuscule bookhand repertoire with rough Greek letter frequencies (lunate sigma/epsilon as variants)
LETTERS = list('ΑΕΟΝΤΙΣΡΚΛΜΥΠΗΔΓΘΩΧΦΒΖΞΨ') + ['Ϲ', 'Є', 'ω']
FREQ = np.array([11, 8, 9, 7.5, 7.5, 7.5, 4, 4.5, 4, 3, 3.5, 4, 3.5, 2.5, 2.5, 1.5, 1.5, 1.5, 1.2, 0.8, 0.6, 0.4, 0.3, 0.2, 3.5, 2, 0.8])
FREQ = FREQ / FREQ.sum()


def _rng(seed=None):
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------- letters
_glyph_cache = {}


def glyph(ch, font, size_px):
    """Binary glyph image (uint8 0/1) rendered at 2x then downsampled; returns crop of the ink bbox."""
    key = (ch, font, size_px)
    if key in _glyph_cache:
        return _glyph_cache[key]
    s2 = size_px * 2
    f = ImageFont.truetype(font, s2)
    im = Image.new('L', (s2 * 2, s2 * 2), 0)
    ImageDraw.Draw(im).text((s2 // 2, s2 // 2), ch, fill=255, font=f)
    a = np.asarray(im)
    ys, xs = np.nonzero(a > 127)
    if len(ys) == 0:
        g = np.zeros((2, 2), np.float32)
    else:
        a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32) / 255
        g = cv2.resize(a, (max(1, a.shape[1] // 2), max(1, a.shape[0] // 2)), interpolation=cv2.INTER_AREA)
    if len(_glyph_cache) < 20000:
        _glyph_cache[key] = g
    return g


def monoline(g, width_px, rng):
    """Replace font strokes by a uniform pen stroke of width_px (skeleton + disk dilation)."""
    sk = skeletonize(g > 0.5)
    r = max(1, int(round(width_px / 2)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    pad = r + 2
    skp = np.pad(sk.astype(np.uint8), pad)
    return cv2.dilate(skp, k).astype(np.float32)


def text_canvas(rng, h_mm=None):
    """Lines of Greek letters on a CAN x CAN canvas (float 0..1, per-letter intensity), plus letter height in px."""
    can = np.zeros((CAN, CAN), np.float32)
    h_mm = h_mm or rng.uniform(1.8, 4.5)
    hpx = h_mm * PXMM
    pitch = hpx + rng.uniform(1.0, 2.5) * PXMM          # line pitch 4-6 mm typical
    font = FONTS[rng.integers(len(FONTS))]
    mono = rng.random() < 0.6
    sw = rng.uniform(0.3, 0.7) * PXMM                      # pen width
    gap = rng.uniform(0.25, 1.2) * PXMM
    nlines = rng.integers(1, 4)
    y0 = CAN / 2 - (nlines - 1) * pitch / 2 + rng.uniform(-pitch / 2, pitch / 2)
    shear = math.tan(math.radians(rng.uniform(-10, 10)))
    p_lacuna = rng.uniform(0, 0.35)
    for li in range(nlines):
        base = y0 + li * pitch + hpx / 2
        x = rng.uniform(-hpx, CAN * 0.15) if rng.random() < 0.8 else rng.uniform(CAN * 0.15, CAN * 0.5)
        xend = CAN if rng.random() < 0.8 else rng.uniform(CAN * 0.5, CAN)   # partial lines
        while x < xend:
            ch = LETTERS[rng.choice(len(LETTERS), p=FREQ)]
            size = int(hpx * rng.uniform(1.3, 1.5))              # font size ~ cap height x 1.4
            g = glyph(ch, font, max(8, size))
            if g.shape[0] < 3:
                continue
            sc = hpx * rng.uniform(0.9, 1.1) / g.shape[0]
            if ch in 'ΦΨ':
                sc *= 1.15
            g = cv2.resize(g, (max(2, int(g.shape[1] * sc)), max(2, int(g.shape[0] * sc))), interpolation=cv2.INTER_AREA)
            if mono:
                g = monoline(g, sw * rng.uniform(0.85, 1.15), rng)
            else:
                g = (g > 0.5).astype(np.float32)
                d = int(round((sw - 0.35 * PXMM) / 2))
                if d > 0:
                    g = cv2.dilate(g, np.ones((2 * d + 1, 2 * d + 1), np.uint8))
                elif d < 0:
                    g = cv2.erode(g, np.ones((-2 * d + 1, -2 * d + 1), np.uint8))
            # per-letter rotation jitter and shear (slant), then crop to the ink bbox
            ang = rng.uniform(-8, 8)
            pad = int(0.3 * max(g.shape)) + 2
            g = np.pad(g, pad)
            hh, ww = g.shape
            M = cv2.getRotationMatrix2D((ww / 2, hh / 2), ang, 1.0)
            M[0, 1] += shear; M[0, 2] -= shear * hh / 2
            g = cv2.warpAffine(g, M, (ww, hh))
            ys, xs = np.nonzero(g > 0.05)
            if len(ys) == 0:
                continue
            g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            hh, ww = g.shape
            if rng.random() > p_lacuna:
                yy = int(base - hh / 2 + rng.normal(0, 0.06 * hpx)); xx = int(x)
                inten = rng.uniform(0.55, 1.0)
                ys0, xs0 = max(0, yy), max(0, xx)
                ys1, xs1 = min(CAN, yy + hh), min(CAN, xx + ww)
                if ys1 > ys0 and xs1 > xs0:
                    sub = g[ys0 - yy:ys1 - yy, xs0 - xx:xs1 - xx] * inten
                    can[ys0:ys1, xs0:xs1] = np.maximum(can[ys0:ys1, xs0:xs1], sub)
            x += ww + gap * rng.uniform(0.7, 1.3)
    return can, hpx


# --------------------------------------------------------------------------- non-letter artefacts
def _ellipse(can, cy, cx, ry, rx, ang, val):
    cv2.ellipse(can, (int(cx), int(cy)), (max(1, int(rx)), max(1, int(ry))), ang, 0, 360, float(val), -1)


def artefact_canvas(rng, kind=None):
    can = np.zeros((CAN, CAN), np.float32)
    kinds = ['scatter', 'streak', 'dashed', 'lattice', 'cluster', 'band', 'blocky', 'single', 'ridge']
    kind = kind or kinds[rng.integers(len(kinds))]
    if kind == 'scatter':          # isolated blobs, no baseline
        for _ in range(rng.integers(3, 18)):
            r = rng.uniform(0.2, 1.6) * PXMM
            _ellipse(can, rng.uniform(0, CAN), rng.uniform(0, CAN), r * rng.uniform(0.4, 1), r, rng.uniform(0, 180), rng.uniform(0.5, 1))
    elif kind == 'streak':         # continuous crack/seam-following streaks
        for _ in range(rng.integers(1, 4)):
            y = rng.uniform(0.2, 0.8) * CAN; w = rng.uniform(0.2, 1.4) * PXMM
            xs = np.arange(-10, CAN + 10, 4); ys = y + np.cumsum(rng.normal(0, 1.2, len(xs)))
            pts = np.stack([xs, ys], 1).astype(np.int32)
            cv2.polylines(can, [pts], False, float(rng.uniform(0.5, 1)), max(1, int(w)))
    elif kind == 'dashed':         # blobs strung along a line (crack artefact): round/oval blobs, no letter shapes
        for _ in range(rng.integers(1, 3)):
            y = rng.uniform(0.25, 0.75) * CAN; x = rng.uniform(-20, 20)
            s = rng.uniform(0.5, 3.0) * PXMM
            while x < CAN + 20:
                L = rng.uniform(0.4, 3.0) * PXMM
                _ellipse(can, y + rng.normal(0, 0.15 * PXMM), x + L / 2, rng.uniform(0.2, 1.3) * PXMM, L / 2, rng.normal(0, 8), rng.uniform(0.5, 1))
                x += L + s * rng.uniform(0.3, 1.2)
    elif kind == 'lattice':        # fibre cross-hatch: periodic ridges in one or two directions
        per = rng.uniform(0.5, 1.6) * PXMM; w = per * rng.uniform(0.2, 0.5)
        yy, xx = np.mgrid[0:CAN, 0:CAN].astype(np.float32)
        a1 = rng.uniform(0, np.pi)
        f = np.cos((xx * np.cos(a1) + yy * np.sin(a1)) * 2 * np.pi / per)
        if rng.random() < 0.6:
            a2 = a1 + rng.uniform(np.pi / 3, 2 * np.pi / 3)
            f = f + np.cos((xx * np.cos(a2) + yy * np.sin(a2)) * 2 * np.pi / (per * rng.uniform(0.8, 1.3)))
        f = (f > np.percentile(f, 100 - rng.uniform(8, 30))).astype(np.float32)
        noise = cv2.GaussianBlur(rng.random((CAN, CAN)).astype(np.float32), (0, 0), 6)
        can = f * (noise > np.percentile(noise, rng.uniform(20, 70)))
    elif kind == 'cluster':        # dense irregular cluster (cavity edge / damage)
        cy, cx = rng.uniform(0.3, 0.7, 2) * CAN; R = rng.uniform(1.5, 4) * PXMM
        for _ in range(rng.integers(8, 40)):
            rr = R * np.sqrt(rng.random()); th = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(0.15, 0.8) * PXMM
            _ellipse(can, cy + rr * np.sin(th), cx + rr * np.cos(th), r * rng.uniform(0.4, 1), r, rng.uniform(0, 180), rng.uniform(0.5, 1))
    elif kind == 'band':           # broad density band
        y = rng.uniform(0.3, 0.7) * CAN; w = rng.uniform(1.5, 5) * PXMM
        yy = np.arange(CAN, dtype=np.float32)[:, None]
        noise = cv2.GaussianBlur(rng.random((CAN, CAN)).astype(np.float32), (0, 0), rng.uniform(2, 8))
        can = np.exp(-0.5 * ((yy - y) / (w / 2)) ** 2) * np.ones((1, CAN), np.float32)
        can = can * (noise > np.percentile(noise, rng.uniform(30, 70)))
    elif kind == 'blocky':         # tile-aligned rectangular blobs (v8-in tile artefacts)
        t = int(rng.uniform(0.4, 1.4) * PXMM)
        for _ in range(rng.integers(3, 14)):
            y, x = rng.integers(0, CAN // t) * t, rng.integers(0, CAN // t) * t
            h, w = rng.integers(1, 4) * t, rng.integers(1, 4) * t
            can[y:y + h, x:x + w] = rng.uniform(0.5, 1)
    elif kind == 'single':         # one big isolated blob (ring / pocket / pit)
        r = rng.uniform(1, 3) * PXMM
        c = CAN / 2 + rng.normal(0, 15, 2)
        _ellipse(can, c[0], c[1], r * rng.uniform(0.5, 1), r, rng.uniform(0, 180), 1)
        if rng.random() < 0.4:
            _ellipse(can, c[0], c[1], r * 0.45, r * 0.5, 0, 0)    # ring
    elif kind == 'ridge':          # wavy fibre-parallel short dashes (fibre texture)
        for _ in range(rng.integers(10, 40)):
            y, x = rng.uniform(0, CAN, 2); L = rng.uniform(0.6, 3) * PXMM
            cv2.line(can, (int(x), int(y)), (int(x + L), int(y + rng.normal(0, 2))), float(rng.uniform(0.5, 1)), max(1, int(rng.uniform(0.15, 0.5) * PXMM)))
    return can, kind


# --------------------------------------------------------------------------- degradation + compositing
def smooth_field(rng, shape, sigma):
    f = cv2.GaussianBlur(rng.standard_normal(shape).astype(np.float32), (0, 0), sigma)
    return f / (f.std() + 1e-6)


def degrade(can, rng, family='fused_hd'):
    """Make a clean 0..1 shape map look like ink-model output. Mimics the model response: wobble, small
    morphological change, partial dropout, blur with the model's spatial support (0.15-0.5 mm), then a soft
    threshold (models saturate -> smooth rounded blobs), and the tile/checker texture of v8-in maps."""
    H, W = can.shape
    if rng.random() < 0.7:
        amp = rng.uniform(0.5, 3.0)
        dx = smooth_field(rng, (H, W), 8) * amp; dy = smooth_field(rng, (H, W), 8) * amp
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        can = cv2.remap(can, xx + dx, yy + dy, cv2.INTER_LINEAR)
    r = rng.integers(-1, 3)
    if r > 0:
        can = cv2.dilate(can, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    elif r < 0:
        can = cv2.erode(can, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    if rng.random() < 0.8:                       # ink models return stroke FRAGMENTS (1-2 mm blobs)
        keep = rng.uniform(0.35, 0.95)
        f = smooth_field(rng, (H, W), rng.uniform(2, 7))
        can = can * (f > np.quantile(f, 1 - keep))
    can = can * np.clip(1 + 0.25 * smooth_field(rng, (H, W), rng.uniform(4, 12)), 0.3, 1.5)
    can = cv2.GaussianBlur(can, (0, 0), rng.uniform(0.1, 0.35) * PXMM)
    mx = float(can.max())
    if mx > 1e-6 and rng.random() < 0.7:          # saturating response -> smooth blobs with firm edges
        t = rng.uniform(0.25, 0.55) * mx; w = rng.uniform(0.03, 0.12) * mx
        can = mx / (1 + np.exp(-(can - t) / w))
        can = cv2.GaussianBlur(can, (0, 0), rng.uniform(0.5, 1.5))
    if family == 'v8in' and rng.random() < 0.8:  # v8-in: fine checker / tile texture
        b = int(rng.integers(2, 4))
        yy, xx = np.mgrid[0:H, 0:W]
        can = can * (rng.uniform(0.55, 0.85) + rng.uniform(0.15, 0.45) * (((yy // b) + (xx // b)) % 2))
    return np.clip(can, 0, None).astype(np.float32)


def rotate_crop(can, ang, win=WIN):
    H, W = can.shape
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
    r = cv2.warpAffine(can, M, (W, H), flags=cv2.INTER_LINEAR)
    y0, x0 = (H - win) // 2, (W - win) // 2
    return r[y0:y0 + win, x0:x0 + win]


def composite(a_bg, m_bg, shape, rng, contrast=None):
    """contrast: z units added at shape==1 (default: 80 % log-uniform 1.2-8, 20 % faint 0.7-1.5)."""
    """Composite a shape (0..~1) into a real a-window: invert a -> approximate z, add contrast*shape, re-clip."""
    zb = np.where(a_bg > 0, 0.9 + 1.8 * a_bg, 0.9 - np.abs(rng.normal(0.9, 0.5)) * (0.5 + 0.5 * np.abs(smooth_field(rng, a_bg.shape, 2))))
    c = contrast if contrast is not None else (float(np.exp(rng.uniform(np.log(1.2), np.log(8.0)))) if rng.random() < 0.8 else rng.uniform(0.7, 1.5))
    tex = 1 + 0.15 * smooth_field(rng, a_bg.shape, 1.5)          # within-stroke texture
    z = zb + c * shape * tex
    a = np.clip((z - 0.9) / 1.8, 0, 1)
    a[~m_bg] = 0
    return a.astype(np.float32), c


def make_text(rng, family='fused_hd'):
    can, hpx = text_canvas(rng)
    can = degrade(can, rng, family)
    return rotate_crop(can, rng.uniform(0, 360)), hpx


def make_artefact(rng, kind=None, family='fused_hd'):
    can, kind = artefact_canvas(rng, kind)
    can = degrade(can, rng, family)
    return rotate_crop(can, rng.uniform(0, 360)), kind
