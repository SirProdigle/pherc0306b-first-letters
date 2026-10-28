"""side_by_side_v5.py PKG VIEWS_JSON : views_side_by_side.jpg, the 4 cm2 box of several views of the package, stacked, in reading
orientation, each with the same 1.5 mm margin around the box, labelled with its models and display rule only (version 5 copy of
version 4c's side_by_side_v4c.py; the view list and labels come from the build config: VIEWS_JSON = [[file stem, label], ...],
the stems relative to PKG, the first being the main image).  Crops come from the package's own images and their json
(claim_box_image_px); nothing is drawn on the ink apart from the dashed box the images carry."""
import sys, json
import numpy as np, cv2


def side_by_side(PKG, VIEWS, out=None):
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
    out_img = np.concatenate(tiles, 0)
    L = int(round(5 * PPM)); y = out_img.shape[0] + 30
    out_img = np.concatenate([out_img, np.full((90, w, 3), 255, np.uint8)], 0)
    cv2.rectangle(out_img, (10, y), (10 + L, y + 14), (0, 0, 0), -1)
    cv2.putText(out_img, '5 mm; the 4 cm2 box of each view plus 1.5 mm; reading orientation; 8.64 um/px', (30 + L, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(out or f'{PKG}/views_side_by_side.jpg', out_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out_img.shape


if __name__ == '__main__':
    print(side_by_side(sys.argv[1], json.load(open(sys.argv[2]))))
