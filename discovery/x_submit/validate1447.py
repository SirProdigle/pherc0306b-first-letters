"""validate1447.py : held-out validation of the submission's max-quality ink recipe (x_deepread/ink_full.py) on PHerc1447 w058,
the labelled winding on the SAME scan protocol as PHerc0306B (8.64 um, 116 keV; HF YoussefMoNader/ink-8um-pherc1447-surfaces).

Input: the x_gates positive-control render of the t6 label crop (x_gates/PHerc1447/renders/pc1447_w058.zarr, 31 layers,
index 15 = the dataset's read layer 12, text on the reverse face) and the dataset's inklabels.png / mask.png over the same
crop.  Runs the exact recipe (Hecate s16 + Reader v2 + TRACE, 8x TTA, d -1/0/+1, z-fused) and scores each model and the
fusion inside the mask (160 px from the crop border): AUC, d', AP.  Writes x_submit/val1447/{metrics.json, maps.npz,
w058_labels_vs_fused.jpg} and copies the metrics and image into the draft's controls/.
"""
import os, sys, json, shutil, time
import numpy as np, cv2, zarr
cv2.setNumThreads(4)
W = '/home/liam/projects/vesuvius-work'
sys.path.insert(0, f'{W}/x_zoom'); sys.path.insert(0, f'{W}/x_deepread')
import zoomlib as Z                                                     # noqa: E402
from ink_full import recipe                                             # noqa: E402
OUT = f'{W}/x_submit/val1447'; os.makedirs(OUT, exist_ok=True)
DRAFT = f'{W}/CANDIDATES/SUBMISSION_DRAFT_A_owner_only'
R = f'{W}/x_gates/PHerc1447/renders/pc1447_w058.zarr'; C, MARGIN = 15, 160
y0, y1, x0, x1 = json.load(open(f'{W}/x_gates/PHerc1447/rv2/pc1447_w058/meta.json'))['crop']
LAB = cv2.imread(f'{W}/data/pherc1447/w058/labels/inklabels.png', 0)[y0:y1, x0:x1] > 127
MSK = cv2.imread(f'{W}/data/pherc1447/w058/labels/mask.png', 0)[y0:y1, x0:x1] > 0


def score(a, valid):
    reg = MSK & valid; reg[:MARGIN] = reg[-MARGIN:] = False; reg[:, :MARGIN] = reg[:, -MARGIN:] = False
    pos, neg = a[reg & LAB], a[reg & ~LAB]
    lo, hi = min(pos.min(), neg.min()), max(pos.max(), neg.max())
    hp, _ = np.histogram(pos, 1024, (lo, hi)); hn, _ = np.histogram(neg, 1024, (lo, hi))
    auc = float((hp * (np.cumsum(hn) - hn / 2.0)).sum() / (hp.sum() * hn.sum()))
    tp = np.cumsum(hp[::-1]); fp = np.cumsum(hn[::-1]); prec = tp / np.maximum(tp + fp, 1); rec = tp / hp.sum()
    ap = float(np.sum(np.diff(np.concatenate([[0], rec])) * prec))
    return dict(auc=round(auc, 4), dprime=round(float((pos.mean() - neg.mean()) / (neg.std() + 1e-6)), 3), ap=round(ap, 4),
                ink_frac=round(float(len(pos) / (len(pos) + len(neg))), 4))


if __name__ == '__main__':
    vol = np.asarray(zarr.open(R, mode='r')['0'][:]); valid = vol[C] > 0
    t = time.time(); m = recipe(vol, C, 8.64, 'reverse'); el = round(time.time() - t)
    np.savez_compressed(f'{OUT}/maps.npz', **{k: v.astype(np.float16) for k, v in m.items()})
    res = {k: score(v.astype(np.float32), valid) for k, v in m.items()}
    res['_note'] = dict(crop=[y0, y1, x0, x1], face='reverse', read_layer_index=C, seconds=el, chance_ap='= ink_frac',
                        previous_t6_auc_at_wrong_7p91um=dict(hec=0.7897, rv2=0.8376, trace=0.7974, hec_rv2=0.8485))
    json.dump(res, open(f'{OUT}/metrics.json', 'w'), indent=1); print(json.dumps(res), flush=True)
    z = m['fused'].astype(np.float32); a = np.clip((z - 1.0) / 2.0, 0, 1); a[~valid] = 0
    g = np.where(valid, 1 - 0.92 * a, 0.12); pg = (np.stack([g * 0.80, g * 0.88, g * 0.93], -1) * 255).astype(np.uint8)
    lb = np.full_like(pg, 235); lb[LAB] = (30, 30, 30); lb[~MSK] = (90, 90, 90)
    img = np.concatenate([lb, np.full((12, pg.shape[1], 3), 40, np.uint8), pg], 0)
    s = 0.5; img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    cv2.putText(img, 'PHerc1447 w058: ink labels (top, grey = outside mask) vs our recipe, fused (bottom); 17.3 um/px', (10, 30), 0, 0.8, (30, 30, 200), 2)
    cv2.rectangle(img, (10, img.shape[0] - 30), (10 + int(10000 / 8.64 * s), img.shape[0] - 20), (30, 30, 200), -1)
    cv2.imwrite(f'{OUT}/w058_labels_vs_fused.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    for f in ('metrics.json', 'w058_labels_vs_fused.jpg'): shutil.copy2(f'{OUT}/{f}', f'{DRAFT}/controls/heldout_PHerc1447_{f}')
    print('wrote', OUT)
