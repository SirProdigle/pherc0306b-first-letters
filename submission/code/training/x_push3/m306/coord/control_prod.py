"""control_prod.py : controls for the product-rule candidate TRACE x A (A = mean of self_main, s1, s2 at TTA 1), against C3 and the
product-rule baseline TRACE x v8-in: depth shuffle (shuffled TRACE x shuffled A, at the real view's levels), blank R, and the decoy
(TRACE x self_decoy). Writes out/ctl_prod.json and out/ctl_prod_{shuffle,decoy}.jpg."""
import os, sys, json
import numpy as np
sys.path.insert(0, '/home/liam/projects/vesuvius-work/.claude/worktrees/leadA-push3/x_push3/m306/eval/code')
import evlib as E                                                                  # noqa: E402
P3 = '/home/liam/projects/vesuvius-work/.claude/worktrees/leadA-push3/x_push3'; S = f'{P3}/m306/self/leadA'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
ld = lambda p: np.load(p).astype(np.float32)
tr, trs = ld(E.TR), ld(E.TR_SHUF)
A = ld(f'{OUT}/ens_main3_t1.npy'); As = np.mean([ld(f'{S}/{n}_shuf_t1.npy') for n in ('self_main', 'self_main_s1', 'self_main_s2')], 0)
v8s = ld(E.V8G3_SHUF)
np.save(f'{OUT}/prod_tr_A_shuf.npy', (trs * As).astype(np.float16)); np.save(f'{OUT}/prod_tr_v8_shuf.npy', (trs * v8s).astype(np.float16))
np.save(f'{OUT}/prod_tr_decoy.npy', (tr * ld(f'{S}/self_decoy_fwd_t1.npy')).astype(np.float16))
models = {'c3tr': ('TRACE', E.TR), 'c3v8': ('v8-in', E.V8G3), 'pta': ('TRACE x A', f'{OUT}/prod_tr_A.npy'),
          'ptv': ('TRACE x v8-in', f'{OUT}/prod_tr_v8.npy'), 'ptd': ('TRACE x decoy', f'{OUT}/prod_tr_decoy.npy')}
shuf = {'c3tr': E.TR_SHUF, 'c3v8': E.V8G3_SHUF, 'pta': f'{OUT}/prod_tr_A_shuf.npy', 'ptv': f'{OUT}/prod_tr_v8_shuf.npy'}
pairs = {'c3': ['c3tr', 'c3v8'], 'pta': ['pta'], 'ptv': ['ptv'], 'ptd': ['ptd']}
cx = E.make_ctx(models, pairs, f'{OUT}/work_ctlprod', shuf)
res, shrows, drows = {}, [], []
for k, lab in [('c3', 'C3 = TRACE + v8-in (submitted)'), ('ptv', 'TRACE x v8-in (product)'), ('pta', 'TRACE x A (product)'), ('ptd', 'TRACE x DECOY (product)')]:
    v = E.pair_view(cx, k); srgb, sst = E.shuffle_view(cx, k, v)
    res[k] = dict(label=lab, box_half=v['stats']['box']['half_tone'], blank_R_half=v['stats']['blank_R']['half_tone'], shuffle=sst)
    print(lab, json.dumps(dict(box=round(res[k]['box_half'], 4), R=round(res[k]['blank_R_half'], 4), shuf=(sst or {}).get('shuffled'))), flush=True)
    drows.append((lab, E.crop(v['rgb'], E.ZOOM, 2)))
    if srgb is not None: shrows += [(f'{lab}: real', E.crop(v['rgb'], E.BOX)), (f'{lab}: DEPTH-SHUFFLED, same levels', E.crop(srgb, E.BOX))]
E.save(f'{OUT}/ctl_prod_shuffle.jpg', E.stack_sheet(shrows, 2400)); E.save(f'{OUT}/ctl_prod_decoy.jpg', E.stack_sheet(drows, 2400))
json.dump(res, open(f'{OUT}/ctl_prod.json', 'w'), indent=1, default=str); print('done')
