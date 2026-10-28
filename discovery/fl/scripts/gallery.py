"""gallery.py OUT.jpg MAPDIR [MAPDIR ...] [--tile 520] [--cols 2] [--qthr 0.15] [--sort]
For each patch (a screen2 maps/<uuid> dir): fwd and reverse ensemble overlays side by side,
tiles whose depth-profile quality is below qthr tinted red (off-sheet), label with good area."""
import sys, glob, os, json, cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
args = sys.argv[1:]
tile, cols, qthr, do_sort = 520, 2, 0.10, False
for flag in ('--tile', '--cols', '--qthr'):
    if flag in args:
        i = args.index(flag); v = args[i + 1]; del args[i:i + 2]
        if flag == '--tile': tile = int(v)
        elif flag == '--cols': cols = int(v)
        else: qthr = float(v)
if '--sort' in args:
    args.remove('--sort'); do_sort = True
out, pdirs = args[0], args[1:]


def qmap(p):
    p = p.rstrip('/')
    r = os.path.join(os.path.dirname(os.path.dirname(p)), 'renders', os.path.basename(p) + '.zarr')
    f = os.path.join(r, 'quality_tiles.npy')
    if os.path.exists(os.path.join(p, 'quality_tiles.npy')):
        f = os.path.join(p, 'quality_tiles.npy')
    elif not os.path.exists(f):
        from quality import quality
        try:
            quality(r)
        except Exception:
            return None, 0.0
    sc = np.load(f)
    good = np.nan_to_num(sc, nan=-1) >= qthr
    return sc, float(good.sum() * (128 * 8.64e-4) ** 2)


def spec(p):
    f = os.path.join(p.rstrip('/'), 'spectral.json')
    try:
        return float(json.load(open(f)).get('max', -9))
    except Exception:
        return -9.0


items = []
for p in pdirs:
    sc, area = qmap(p)
    items.append((area, p, sc))
if do_sort:
    items.sort(key=lambda x: (-spec(x[1]), -x[0]))
cells = []
for area, p, sc in items:
    pair = []
    for side in ['fwd', 'reverse']:
        fs = sorted(glob.glob(f'{p}/*_{side}_overlay.png') + glob.glob(f'{p}/maps/*_{side}_overlay.png'))
        if not fs:
            continue
        a = cv2.imread(fs[0])
        if sc is not None:
            m = (np.nan_to_num(sc, nan=1.0) < qthr).astype(np.uint8)       # bad tiles (nan = outside: leave)
            m = cv2.resize(m, (sc.shape[1] * 128 // 3, sc.shape[0] * 128 // 3), interpolation=cv2.INTER_NEAREST)
            m = m[:a.shape[0], :a.shape[1]]
            red = a.copy(); red[:, :, 2] = np.maximum(red[:, :, 2], 90)
            a[:m.shape[0], :m.shape[1]][m > 0] = (0.5 * a[:m.shape[0], :m.shape[1]][m > 0] + 0.5 * red[:m.shape[0], :m.shape[1]][m > 0]).astype(np.uint8)
        s = tile / max(a.shape[:2])
        a = cv2.resize(a, (int(a.shape[1] * s), int(a.shape[0] * s)), interpolation=cv2.INTER_AREA)
        c = np.zeros((tile + 24, tile, 3), np.uint8); c[24:24 + a.shape[0], :a.shape[1]] = a
        cv2.putText(c, f'{os.path.basename(p.rstrip("/"))[-15:]} {side[:3]} good={area:.1f} rows={spec(p):.1f}', (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        pair.append(c)
    if pair:
        cells.append(np.hstack(pair + [np.zeros((tile + 24, 8, 3), np.uint8)]))
if not cells:
    sys.exit('nothing')
w = max(c.shape[1] for c in cells); cells = [np.pad(c, ((0, 0), (0, w - c.shape[1]), (0, 0))) for c in cells]
rows = [np.hstack(cells[i:i + cols] + [np.zeros_like(cells[0])] * (cols - len(cells[i:i + cols]))) for i in range(0, len(cells), cols)]
cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
print(out, len(cells), [(round(a, 1), os.path.basename(p.rstrip('/'))[-15:]) for a, p, _ in items])
