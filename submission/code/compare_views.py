"""compare_views.py [PKG] : views_side_by_side.jpg, the claim box (+ 3 mm) in the main image and three other views, stacked, plus the
whole area of the main image. Cut from the package's own PNGs (sidecar geometry), so it shows exactly the submitted pixels."""
import os, sys, json, glob
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

W = '/home/liam/projects/vesuvius-work'
PKG = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('I2_PKG', f'{W}/x_refine/trials/I2_package/SUBMISSION_v2_DRAFT_owner_only')
MESH = 'PHerc0306B_20250521133212_w074own_touchup_V5Rc'
FONT = '/usr/share/fonts/nerd-fonts/UbuntuSans/UbuntuSansMonoNerdFontPropo-Medium.ttf'
PPMX = 1000 / 8.64; CTX = dict(u0=-30.0, v1=15.0); BOX = dict(u0=-15.0, u1=23.0, v0=-6.84, v1=2.72); M = 3.0
TEXT = {'rv8q_grey': 'Reader v2 + v8Q (fine-tuned), no cut-off',
        'rv8q_secondary': 'Reader v2 + v8Q (fine-tuned), fixed cut-off: keeps letters apart',
        'rt_grey': 'Reader v2 + TRACE (two other models), no cut-off',
        'rt_primary': 'Reader v2 + TRACE, fixed cut-off'}
ORDER = ['rv8q_grey', 'rv8q_secondary', 'rt_grey', 'rt_primary']


def views():
    out = {}
    for j in [f'{PKG}/{MESH}.json'] + glob.glob(f'{PKG}/supplementary/*.json'):
        d = json.load(open(j)); v = d.get('variant')
        if v and 'image_px' in d: out[v['key']] = (j[:-5] + '.png', d['image_px'], v)
    return out


def body(png, ip):
    a = cv2.imread(png)
    return a[ip['map_top']:ip['map_top'] + ip['map_h'], ip['map_left']:ip['map_left'] + ip['map_w']]


def main():
    V = views(); f = ImageFont.truetype(FONT, 46); fs = ImageFont.truetype(FONT, 34); W_ = 2889
    x0 = int((BOX['u0'] - M - CTX['u0']) * PPMX); x1 = int((BOX['u1'] + M - CTX['u0']) * PPMX)
    y0 = int((CTX['v1'] - BOX['v1'] - M) * PPMX); y1 = int((CTX['v1'] - BOX['v0'] + M) * PPMX)
    tiles = []
    def label(t):
        im = Image.new('RGB', (W_, 70), 'white'); ImageDraw.Draw(im).text((12, 10), t, fill=(0, 0, 0), font=f); return np.array(im)[..., ::-1]
    fit = lambda a: cv2.resize(a, (W_, int(a.shape[0] * W_ / a.shape[1])), interpolation=cv2.INTER_AREA)
    lead = next(k for k, (_, _, v) in V.items() if v.get('lead'))
    png, ip, v = V[lead]
    tiles += [label(f"Whole area, main image (view {v['view']})"), fit(body(png, ip))]
    for k in [lead] + [k for k in ORDER if k != lead and k in V]:
        png, ip, v = V[k]
        tag = 'MAIN IMAGE, ' if k == lead else ''
        tiles += [np.full((24, W_, 3), 255, np.uint8), label(f"View {v['view']}: {tag}{TEXT.get(k, k)}"), fit(body(png, ip)[y0:y1, x0:x1])]
    foot = Image.new('RGB', (W_, 60), 'white')
    ImageDraw.Draw(foot).text((12, 12), 'Same surface, frame and 4 cm2 box (dashed) in every view. Nothing is drawn on the ink. '
                              'Box zooms: the box plus 3 mm.', fill=(0, 0, 0), font=fs)
    out = np.vstack(tiles + [np.array(foot)[..., ::-1]])
    cv2.imwrite(f'{PKG}/views_side_by_side.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 92]); print('views_side_by_side.jpg', out.shape, lead)


if __name__ == '__main__':
    main()
