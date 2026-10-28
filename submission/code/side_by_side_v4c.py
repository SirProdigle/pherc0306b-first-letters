"""side_by_side_v3.py PKG : views_side_by_side.jpg for the v3 package: the 4 cm2 box of five views of the package, stacked, in reading
orientation, each with the same 1.5 mm margin around the box, labelled with its models and display rule only.  Crops come from the
package's own images and their json (claim_box_image_px); nothing is drawn on the ink apart from the dashed box the images carry."""
import sys, json
import numpy as np, cv2

PKG = sys.argv[1]; MESH = 'PHerc0306B_20250521133212_w074own_touchup_V5Rc'
VIEWS = [(f'{MESH}', 'Main image: Reader V3 + TRACE v3B, grey rule (no threshold)'),
         ('supplementary/V3_TRACE_secondary_rung_L050', 'Reader V3 + TRACE v3B, threshold rule, secondary rung (blank-strip rule)'),
         ('supplementary/V3_grey_page', 'Reader V3 alone, grey rule'),
         ('supplementary/Rv2_v8Q_grey_page', 'Reader v2 + v8Q, grey rule (version 2 main image pair)'),
         ('supplementary/Rv2_TRACE_grey_page', 'Reader v2 + TRACE v3B, grey rule')]
PPM = 1000 / 8.64; PAD = int(1.5 * PPM)
tiles = []
for f, label in VIEWS:
    img = cv2.imread(f'{PKG}/{f}.png'); j = json.load(open(f'{PKG}/{f}.json')); b = j['claim_box_image_px']
    crop = img[max(0, b['y0'] - PAD):b['y1'] + PAD, max(0, b['x0'] - PAD):b['x1'] + PAD]
    bar = np.full((70, crop.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, label, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (30, 30, 30), 3, cv2.LINE_AA)
    tiles.append(np.concatenate([bar, crop, np.full((24, crop.shape[1], 3), 255, np.uint8)], 0))
w = max(t.shape[1] for t in tiles)
tiles = [np.pad(t, ((0, 0), (0, w - t.shape[1]), (0, 0)), constant_values=255) for t in tiles]
out = np.concatenate(tiles, 0)
L = int(round(5 * PPM)); y = out.shape[0] + 30
out = np.concatenate([out, np.full((90, w, 3), 255, np.uint8)], 0)
cv2.rectangle(out, (10, y), (10 + L, y + 14), (0, 0, 0), -1)
cv2.putText(out, '5 mm; the 4 cm2 box of each view plus 1.5 mm; reading orientation; 8.64 um/px', (30 + L, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
cv2.imwrite(f'{PKG}/views_side_by_side.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 92])
print(out.shape)
