"""banner_models.py PKG : writes a 'reading aid, not the claim image' banner into the empty right end of the header of every page in
PKG/supporting/models/ (x >= 5150, header rows only; the ink, box, frame and margins are untouched), and updates each page's
sidecar image_sha256.  Page 01 (the claim rule) gets a 'same rule as the claim image' banner instead."""
import sys, os, glob, json, hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
PKG = sys.argv[1]; D = f'{PKG}/supporting/models'                                   # models_individual/ when it ran
FB = '/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf'; FR = '/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf'
CLAIM = 'PHerc0306B_20250521133212_w074own_touchup_V5Rc.png'
for f in sorted(glob.glob(f'{D}/[0-9][0-9]_*.png')):
    im = Image.open(f).convert('RGB'); a = np.array(im)
    assert (a[:190, 5150:] > 200).all(), f'{f}: header space not empty'
    main = os.path.basename(f).startswith('01_')
    t1 = 'SAME RULE AS THE CLAIM IMAGE, SHOWN FOR COMPARISON' if main else 'READING AID, NOT THE CLAIM IMAGE'
    t2 = f'The claim image is {CLAIM} in the package root.'
    d = ImageDraw.Draw(im)
    d.rectangle([5150, 14, 7365, 182], outline=(150, 0, 0), width=5)
    d.text((5180, 30), t1, fill=(150, 0, 0), font=ImageFont.truetype(FB, 60 if not main else 46))
    d.text((5180, 118), t2, fill=(60, 0, 0), font=ImageFont.truetype(FR, 33))
    im.save(f, compress_level=6)
    j = f[:-4] + '.json'
    if os.path.exists(j):
        s = json.load(open(j)); s['image_sha256'] = hashlib.sha256(open(f, 'rb').read()).hexdigest()
        s['banner'] = f'{t1}. {t2} (written in the header after rendering; ink, box and frame unchanged)'
        json.dump(s, open(j, 'w'), indent=1, ensure_ascii=False)
    print(os.path.basename(f), flush=True)
