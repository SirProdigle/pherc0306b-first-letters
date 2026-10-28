"""Row-vs-fibre alignment check.

Text rows run parallel to one fibre family. From a CT layer of the render, measure the angular power of
fibre-scale texture (period 0.15–0.8 mm) on the on-sheet area; the fibre families show up as peaks in
frequency direction. Rows found by spectral_rows give a frequency direction `row_angle` (perpendicular to
the rows). Rows parallel to a fibre family <=> row_angle close to that family's frequency angle.

fibre_peaks(ct, valid) -> list of (angle_deg, strength) for the top 2 separated angular peaks
alignment(ct, valid, row_angle) -> dict(fibre_angles, min_diff_deg)
"""
import numpy as np, cv2
import os as _os; cv2.setNumThreads(int(_os.environ.get('OMP_NUM_THREADS', '4')))  # OpenCV ignores OMP caps; uncapped it used all 32 cores

PX_MM = 1000 / 8.64


def angular_power(img, valid, pmin_mm=0.15, pmax_mm=0.8, nang=72, px_mm=PX_MM):
    x = img.astype(np.float32)
    v = valid.astype(np.float32)
    m = x[valid].mean() if valid.any() else 0
    x = (x - m) * v
    x = x - cv2.GaussianBlur(x, (0, 0), 0.8 * px_mm) * v     # remove large-scale shading
    h, w = x.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    F = np.abs(np.fft.fftshift(np.fft.fft2(x * win))) ** 2
    yy, xx = np.mgrid[:h, :w]
    fy = (yy - h // 2) / h; fx = (xx - w // 2) / w
    r = np.hypot(fy, fx) * px_mm
    ang = (np.degrees(np.arctan2(fy, fx)) + 180) % 180
    band = (r >= 1 / pmax_mm) & (r <= 1 / pmin_mm)
    ab = (ang // (180 / nang)).astype(int)
    pw = np.array([F[band & (ab == k)].mean() if (band & (ab == k)).any() else 0 for k in range(nang)])
    return pw / (np.median(pw) + 1e-9), 180 / nang


def fibre_peaks(img, valid, **kw):
    pw, step = angular_power(img, valid, **kw)
    sm = np.convolve(np.r_[pw[-2:], pw, pw[:2]], np.ones(5) / 5, 'same')[2:-2]
    order = np.argsort(-sm)
    peaks = []
    for k in order:
        a = (k + 0.5) * step
        if all(min(abs(a - b), 180 - abs(a - b)) > 30 for b, _ in peaks):
            peaks.append((float(a), float(sm[k])))
        if len(peaks) == 2:
            break
    return peaks


def alignment(img, valid, row_angle, **kw):
    peaks = fibre_peaks(img, valid, **kw)
    diffs = [min(abs(row_angle - a), 180 - abs(row_angle - a)) for a, _ in peaks]
    return dict(fibre_angles=[round(a, 1) for a, _ in peaks], fibre_strength=[round(s, 2) for _, s in peaks],
                min_diff_deg=round(float(min(diffs)), 1) if diffs else None)
