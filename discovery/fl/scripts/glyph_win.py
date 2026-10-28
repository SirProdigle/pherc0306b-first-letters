"""glyph_win.py MAPS_ROOT [--tag l1218cw] [--um 8.64] [--model x_glyph/models/gA.pt] [--redo] : learned "is this text?"
score for every tri_rank window -> <winding>/glyph_win.json {"fwd": [[x0, score], ...], "rev": [...], "model": ...}

Score = x_glyph glyph-likeness CNN (6.4 mm windows on the 50 um overlay map, trained on synthetic Greek letters plus
real ink-map text of OTHER scrolls), max over 128 px sub-windows (stride 16, >= 70 % on-sheet) of the 5-fold-TTA logit,
inside the tri_rank window (2600 px at a 1300 step, full height, skipped quality tiles masked).
Validation (x_textdet/, leave-one-dataset-out models, 1276 windows of l1218cw + l1545acw as negatives): 0343 lane text
windows rank 1st/2nd/17th, PHerc1447 w058 two-row text 2nd/2nd (the row score ranks it 630th-990th), PHerc0841 crop 7th.
Use with tri_rank.py --glyph: windows with glyph >= GLYPH_P99 enter the gates even when their row score is < --min.
Needs CUDA (x_glyph.evaluate.score_windows); GPU memory fraction capped at 6 % (~1.4 GB).
"""
import sys, os, glob, json, argparse
import numpy as np, cv2, torch
cv2.setNumThreads(int(os.environ.get('OMP_NUM_THREADS', '4')))
W = '/home/liam/projects/vesuvius-work'
sys.path.insert(0, f'{W}/x_glyph')
from common import a_from_z, onsheet_windows, WIN as GWIN, UM          # noqa: E402  x_glyph helpers
from evaluate import score_windows                                      # noqa: E402
from train import load_model                                            # noqa: E402

WIN, STEP = 2600, 1300


def skip_mask(z, tile=128):
    """tiles the lane skipped (fastink --mask): constant minimum z over the tile"""
    valid = z != 0
    if not valid.any():
        return np.zeros_like(valid)
    at_min = valid & (z == z[valid].min()); sk = np.zeros_like(valid)
    for i in range(0, z.shape[0], tile):
        for j in range(0, z.shape[1], tile):
            v = valid[i:i + tile, j:j + tile]; n = v.sum()
            if n and at_min[i:i + tile, j:j + tile].sum() >= 0.9 * n:
                sk[i:i + tile, j:j + tile] = v
    return sk


def window_scores(z, um, net):
    """[[x0, score], ...] for the tri_rank windows of one fused z map (native px)"""
    valid = (z != 0) & ~skip_mask(z)
    A, M = a_from_z(z, um, valid)
    f = um / UM; out = []
    for x0 in range(0, max(1, z.shape[1] - WIN + 1), STEP):
        if (z[:, x0:x0 + WIN] != 0).mean() < 0.3:
            continue
        x50, w50 = int(round(x0 * f)), int(round(min(WIN, z.shape[1] - x0) * f))
        it = dict(a=A[:, x50:x50 + w50], mask=M[:, x50:x50 + w50])
        ws = onsheet_windows(it, GWIN, 16, 0.7)
        out.append([x0, round(float(score_windows(net, it, ws).max()), 3) if ws else None])
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('maps'); ap.add_argument('--tag', default='')
    ap.add_argument('--um', type=float, default=8.64); ap.add_argument('--model', default=f'{W}/x_glyph/models/gA.pt')
    ap.add_argument('--redo', action='store_true')
    a = ap.parse_args()
    dev = 'cuda'
    if dev == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.06)
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    net = load_model(a.model, dev)
    for d in sorted(glob.glob(os.path.join(a.maps, (a.tag + '_' if a.tag else '') + '*/'))):
        out = os.path.join(d, 'glyph_win.json')
        if os.path.exists(out) and not a.redo:
            continue
        res = {'model': a.model}
        for f in glob.glob(d + '*_d0_*_z.npy'):
            side = 'rev' if '_reverse_' in f else 'fwd'
            res[side] = window_scores(np.load(f).astype(np.float32), a.um, net)
        if len(res) > 1:
            json.dump(res, open(out, 'w'))
            print(os.path.basename(d.rstrip('/')), {k: max((s for _, s in v if s is not None), default=None) for k, v in res.items() if k != 'model'}, flush=True)
