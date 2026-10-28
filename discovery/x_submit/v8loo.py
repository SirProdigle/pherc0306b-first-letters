"""v8loo.py : a fourth, independently trained ink model on lead A's claim region: v8-in fine-tuned on PHerc1447 w058 + w060
(YoussefMoNader/ink-8um-v8in-pherc1447-loo-w062; weights only: config.json + model.safetensors, sha256 4b28e153...26b5, loaded
with the local ink8um class whose code is identical to the release).  The base v8-in trained on Scroll 1, Scroll 5, 1667, 0139,
0814, 0500P2 and a fragment; the fine-tune added 1447 only, so neither saw PHerc0306B.  It is the only public model trained on
text from the 8.64 um / 116 keV protocol of 0306B.

Layer order is checked empirically: both orders are run on the PHerc1447 w058 crop (inside its training set, so the right order
gives crisp text; scored against the labels) and on A's rectangle bbox.  Writes x_submit/v8loo/{metrics.json, A_v8loo_<order>.npy,
v8loo_A_vs_recipe.jpg}.  GPU: run under x_discord/gpu.lock.
"""
import os, sys, json
import numpy as np, cv2, zarr
cv2.setNumThreads(4)
W = '/home/liam/projects/vesuvius-work'
sys.path.insert(0, f'{W}/third_party/v8in')
from ink8um import InkDetector, predict_stack                           # noqa: E402
OUT = f'{W}/x_submit/v8loo'; os.makedirs(OUT, exist_ok=True)
DRAFT = f'{W}/CANDIDATES/SUBMISSION_DRAFT_A_owner_only'
M = InkDetector.from_pretrained(f'{W}/models/community/v8in_1447_loo_w062').cuda().eval()


def stack(render, c, box):
    y0, y1, x0, x1 = box
    a = np.asarray(zarr.open(render, mode='r')['0'][c - 12:c + 12, y0:y1, x0:x1])
    return np.ascontiguousarray(np.minimum(a, 200).transpose(1, 2, 0)).astype(np.uint8)


if __name__ == '__main__':
    sys.path.insert(0, f'{W}/x_submit'); import validate1447 as V                 # labels + scorer for the 1447 crop
    res = {}
    st = stack(f'{W}/x_gates/PHerc1447/renders/pc1447_w058.zarr', 15, [0, V.LAB.shape[0], 0, V.LAB.shape[1]])
    valid = st[..., 12] > 0
    for rev in (False, True):
        p = predict_stack(M, st, reverse=rev, batch_size=8, num_workers=0)
        res[f'1447_w058_reverse{rev}'] = V.score(p.astype(np.float32), valid); print('1447', rev, res[f'1447_w058_reverse{rev}'], flush=True)
    roi = json.load(open(f'{DRAFT}/controls/panel_spec.json'))['target']['roi']
    st = stack(f'{W}/x_focus306/renders/w074own.zarr', 20, roi)
    for rev in (False, True):
        p = predict_stack(M, st, reverse=rev, batch_size=8, num_workers=0)
        np.save(f'{OUT}/A_v8loo_reverse{rev}.npy', p.astype(np.float16))
        res[f'A_reverse{rev}'] = dict(mean=round(float(p[st[..., 12] > 0].mean()), 4), frac_gt05=round(float((p[st[..., 12] > 0] > 0.5).mean()), 4))
        print('A', rev, res[f'A_reverse{rev}'], flush=True)
    res['_note'] = dict(roi_render_px=roi, model='YoussefMoNader/ink-8um-v8in-pherc1447-loo-w062', sha256='4b28e153e0a631679c79b6c9b8e3e159108d75aeddc6fdea648708ffec5026b5')
    json.dump(res, open(f'{OUT}/metrics.json', 'w'), indent=1); print('wrote', OUT)
