"""mirror_control.py PACKAGE_DIR [IMAGE.png] : reading control (lane I2).  The claim box (+3 mm) of an image of the package (default the lead), flipped left-right.
A reader who 'reads' letters in the mirror panel as readily as in the image is reading from expectation, not from the strokes
(the papyrologist's blind read includes this panel; PREREG: SC(mirror) <= 1).  Pixels are the lead image's own; nothing added."""
import os, sys, json
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claimlib_v2 as L                                                            # noqa: E402

pkg = sys.argv[1]; src = sys.argv[2] if len(sys.argv) > 2 else ''                      # src: an image of the package (default: the lead)
mesh = [d for d in os.listdir(f'{pkg}/mesh') if os.path.isdir(f'{pkg}/mesh/{d}') and not d.endswith('_region')][0]
png = src or f'{pkg}/{mesh}.png'; side = json.load(open(png[:-4] + '.json')); b = side['claim_box_image_px']; ip = side['image_px']
m = int(round(3 * L.PPMX))
img = cv2.cvtColor(cv2.imread(png), cv2.COLOR_BGR2RGB)
y0, y1 = max(ip['map_top'], b['y0'] - m), min(ip['map_top'] + ip['map_h'], b['y1'] + m)
x0, x1 = max(ip['map_left'], b['x0'] - m), min(ip['map_left'] + ip['map_w'], b['x1'] + m)
crop = np.ascontiguousarray(img[y0:y1, x0:x1][:, ::-1])
lv = side['display']['levels']; kind = side['display'].get('kind', 'threshold')
lvt = (f'L {lv["L"]:.2f}, p99 {lv["H"]:.2f}' if kind == 'threshold' else f'grey: median {lv["lo_median"]:.2f} .. p99.9 {lv["hi_p999"]:.2f}')
fn = f'{pkg}/supplementary/ctrl_mirror_{side.get("variant", {}).get("file", "lead")}.jpg'
out = L.frame_image(crop, [f'CONTROL: MIRROR (left-right) of the claim box + 3 mm of {os.path.basename(png)} ({" + ".join(side["ink"]["models"])}, '
                           f'same pixels, same rule: {lvt}). A reading control, not a claim.'], [], margin=40)
cv2.imwrite(fn, cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92]); print(fn, out.shape)
