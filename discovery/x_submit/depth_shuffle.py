"""depth_shuffle.py : depth-shuffle control for lead A's claim rectangle, with PHerc0343's known letters as the yardstick.

Ink sits at a depth; a model that draws the same shapes after the layer order is scrambled is reading in-plane texture, not
ink at a depth. For each target, the layers the models read (c-9..c+9) are randomly permuted (two seeds) and Hecate s16,
Reader v2 and TRACE (d0, no TTA: the control compares like with like) are run on the original and on each shuffle.
Per target: Pearson r and hot-mask Jaccard (fused z > 2) of each shuffle against the original, hot fractions; for 0343 also
AUC / d' / AP against the stroke labels (x_px343).  Writes x_submit/depth_shuffle/{metrics.json, depth_shuffle_control.jpg}
and copies both into CANDIDATES/SUBMISSION_DRAFT_A_owner_only/controls/.  GPU: run under x_discord/gpu.lock.
"""
import os, sys, json, shutil
import numpy as np, cv2
cv2.setNumThreads(4)
W = '/home/liam/projects/vesuvius-work'
sys.path.insert(0, f'{W}/x_zoom'); sys.path.insert(0, f'{W}/x_zoom')
import zoomlib as Z                                                     # noqa: E402
OUT = f'{W}/x_submit/depth_shuffle'; os.makedirs(OUT, exist_ok=True)
DRAFT = f'{W}/CANDIDATES/SUBMISSION_DRAFT_A_owner_only'
roi = json.load(open(f'{DRAFT}/controls/panel_spec.json'))['target']['roi']     # A rectangle bbox in render px
TARGETS = {
    'A': dict(render=f'{W}/x_focus306/renders/w074own.zarr', c=20, face='forward', box=roi, label='lead A rectangle (bbox), PHerc0306B'),
    '0343': dict(render=f'{W}/fl/0343/concat_66.zarr', c=33, face='reverse', box=[450, 2350, 2400, 5300], label='PHerc0343 known letters'),
}
SEEDS = (1, 2)


def run(vol, c, face):
    hec = Z.hecate_map(vol, 8.64, c, 0, face, stride=16)
    rv2 = Z.villa_map('rv2', vol, c, 0, face, overlap=0.5)
    tr = Z.villa_map('trace', vol, c, 0, face, overlap=0.5)
    valid = vol[c] > 0
    return Z.zfuse([hec, rv2, tr], valid), valid


def page(z, v, s=8.64 / 25):
    a = np.clip((z - 1.0) / 2.0, 0, 1); a[~v] = 0; g = np.where(v, 1 - 0.92 * a, 0.12)
    im = (np.stack([g * 0.80, g * 0.88, g * 0.93], -1) * 255).astype(np.uint8)
    return cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)), interpolation=cv2.INTER_AREA)


if __name__ == '__main__':
    sys.path.insert(0, f'{W}/x_zoom')
    res, rows = {}, []
    for t, T in TARGETS.items():
        y0, y1, x0, x1 = T['box']; vol, _ = Z.crop(T['render'], y0, y1, x0, x1); c = T['c']
        f0, v = run(vol, c, T['face']); np.save(f'{OUT}/{t}_orig.npy', f0.astype(np.float16))
        r = dict(label=T['label'], hot_orig=round(float((f0[v] > 2).mean()), 4), shuffles={})
        if t == '0343':
            import validate343 as V
            r['orig_vs_labels'] = V.score(f0, v)
        panels = [page(f0, v)]
        for sd in SEEDS:
            lo, hi = c - 9, c + 10; perm = np.random.default_rng(sd).permutation(np.arange(lo, hi))
            vs = vol.copy(); vs[lo:hi] = vol[perm]
            fs, _ = run(vs, c, T['face']); np.save(f'{OUT}/{t}_shuf{sd}.npy', fs.astype(np.float16))
            a, b = f0[v], fs[v]; h0, h1 = a > 2, b > 2
            s = dict(perm=perm.tolist(), pearson_r=round(float(np.corrcoef(a, b)[0, 1]), 3),
                     hot_jaccard=round(float((h0 & h1).sum() / max(1, (h0 | h1).sum())), 3), hot_shuf=round(float(h1.mean()), 4))
            if t == '0343': s['vs_labels'] = V.score(fs, v)
            r['shuffles'][sd] = s; panels.append(page(fs, v)); print(t, sd, json.dumps({k: s[k] for k in s if k != 'perm'}), flush=True)
        res[t] = r
        H = max(p.shape[0] for p in panels)
        row = np.concatenate([np.pad(p, ((0, H - p.shape[0]), (0, 12), (0, 0)), constant_values=40) for p in panels], 1)
        head = np.full((50, row.shape[1], 3), 255, np.uint8)
        cv2.putText(head, f"{T['label']}: original layer order | shuffled (seed 1) | shuffled (seed 2); 25 um/px", (10, 34), 0, 0.8, (20, 20, 20), 2)
        rows.append(np.concatenate([head, row], 0))
    Wd = max(r_.shape[1] for r_ in rows)
    img = np.concatenate([np.pad(r_, ((0, 12), (0, Wd - r_.shape[1]), (0, 0)), constant_values=40) for r_ in rows], 0)
    cv2.rectangle(img, (10, img.shape[0] - 30), (10 + 400, img.shape[0] - 20), (30, 30, 200), -1)       # 1 cm at 25 um/px
    cv2.imwrite(f'{OUT}/depth_shuffle_control.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    json.dump(res, open(f'{OUT}/metrics.json', 'w'), indent=1)
    for f in ('depth_shuffle_control.jpg', 'metrics.json'):
        shutil.copy2(f'{OUT}/{f}', f"{DRAFT}/controls/depth_shuffle_{f.replace('depth_shuffle_', '')}")
    print('wrote', OUT)
