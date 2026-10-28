"""train_self.py CONFIG.json [--max-minutes 26] [--bench N] [--dry] : SELF lane fine-tune of v8-in on PHerc0306B pseudo-labels
(outside the lead-A exclusion zone) mixed with the PHerc1447 w058 / w060 labelled anchor.

Uses the TRAIN lane's library (m306/train/v8306lib.py, imported read-only): src_1447 (organisers' tile grid, 1447 stacks read
23 -> 0), augment_tile (organisers' LOO augmentation), loss_fn (0.5 Dice + 0.5 SoftBCE smooth 0.25 on keep pixels), load_base,
save_ckpt (safetensors + config.json + meta.json), lr_at.  BatchNorm in TRAIN mode at micro-batch 4 with per-block activation checkpointing (BN
running stats restored after backward, copied from TRAIN's train_v8306.py), bf16 autocast, AdamW, grad
clip 1.0, VRAM cap (set_per_process_memory_fraction).
0306B sources (m306/self/labels/<r>_labels.npz, variant main | sc | decoy): img = the 24 layers v8-in reads (layer0.., stored order
= base v8-in's lead-A order), clipped at 200, with CT zeroed wherever the pad-0 zone rule fails; lab = 1 on pseudo-ink; keep = 1 on
labelled pixels (ink or background), 0 on ignored ones.  Tiles: 64 px on a stride-32 grid, the WHOLE tile inside the pool mask
(pad 0.7 + on-sheet + CT), >= 20 % labelled; every tile's four corners and centre are re-checked with check.allowed(pad_mm=0.7)
at load (assert).  Arrays are padded by 64 px of zeros so the 128 px augmentation context exists.
Batches: each micro-batch = `n1447` anchor tiles (uniform over w058 + w060 tiles) + the rest 0306B tiles (half from tiles that
contain pseudo-ink, half uniform; regions in proportion to their tile counts).  loss = (1 - w306) * loss(anchor) + w306 * loss(0306B).
Outputs m306/self/runs/<name>/: state.pt (resume), train_log.jsonl, step_<s>/, final/ (InkDetector.from_pretrained dir), done.json,
tiles.json (tile lists + label stats)."""
import argparse, json, os, signal, sys, time, threading, queue
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch, zarr

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
P3 = '/home/liam/projects/vesuvius-work/.claude/worktrees/leadA-push3/x_push3'
sys.path.insert(0, f'{P3}/m306/train')
import v8306lib as L                                                               # noqa: E402
import masks                                                                       # noqa: E402
sys.path.insert(0, f'{P3}/aclick/p306/exclusion'); import check                    # noqa: E402
SELF = f'{P3}/m306/self'; LAB = os.environ.get('SELF_LAB', f'{SELF}/labels')
PADM = 64


def _oom_exit(tp, val, tb):
    if issubclass(tp, torch.OutOfMemoryError) or 'out of memory' in str(val).lower():
        print(f'CUDA OOM ({val}); exit 3 for a retry', flush=True); os._exit(3)
    sys.__excepthook__(tp, val, tb)


def src_306(r, variant, stride=32, min_lab=0.2):
    R = masks.regions()[r]
    z = np.load(f'{LAB}/{r}_labels.npz'); y0, y1, x0, x1 = [int(v) for v in z['bbox']]
    lab8 = z[variant]; valid = z['valid']
    m = masks.load(r); a0, a7, qf = masks.derived(m)
    pm = (a7 & m['cov24'] & qf)[y0:y1, x0:x1]
    l0 = R['layer0']
    vol = np.asarray(zarr.open(R['zarr'], mode='r')['0'][l0:l0 + 24, y0:y1, x0:x1])
    vol[:, ~a0[y0:y1, x0:x1]] = 0
    if R['face'] != 'fwd': vol = vol[::-1]
    img = np.minimum(vol, L.CLIP).astype(np.uint8).transpose(1, 2, 0); del vol
    img = np.ascontiguousarray(np.pad(img, ((PADM, PADM), (PADM, PADM), (0, 0))))
    lab = np.pad((lab8 == 1).astype(np.float32), PADM); keep = np.pad((lab8 != 255).astype(np.uint8), PADM)
    H, Wd = pm.shape; T = L.TILE
    Ip = L.integral(pm); Il = L.integral(lab8 != 255); Ii = L.integral(lab8 == 1)
    Y, X = np.meshgrid(np.arange(0, H - T + 1, stride), np.arange(0, Wd - T + 1, stride), indexing='ij'); Y, X = Y.ravel(), X.ravel()
    ok = (L.box_sum(Ip, Y, X) == T * T) & (L.box_sum(Il, Y, X) >= min_lab * T * T)
    Y, X = Y[ok], X[ok]
    nink = L.box_sum(Ii, Y, X)
    # exact zone re-check (render px): centre + 4 corners of every tile at pad 0.7
    for dy, dx in ((T // 2, T // 2), (0, 0), (0, T - 1), (T - 1, 0), (T - 1, T - 1)):
        check.assert_allowed(R['zarr'], Y + y0 + dy, X + x0 + dx, pad_mm=0.7)
    org = np.stack([Y + PADM, X + PADM], 1).astype(np.int64)
    info = dict(source=f'PHerc0306B {r} SELF pseudo-labels ({variant})', zarr=R['zarr'], bbox=[y0, y1, x0, x1], tiles=int(len(org)),
                ink_tiles=int((nink >= 0.01 * T * T).sum()), labelled_px=int((lab8 != 255).sum()), ink_px=int((lab8 == 1).sum()),
                order=f'stored, layers {l0}..{l0 + 23}', zone_check='every tile centre + corners check.allowed(pad_mm=0.7)')
    return dict(img=img, lab=lab, keep=keep, org=org, info=info, name=r, weight_map=None, ink=(nink >= 0.01 * T * T))


def main():
    sys.excepthook = _oom_exit
    ap = argparse.ArgumentParser()
    ap.add_argument('config'); ap.add_argument('--max-minutes', type=float, default=26.0)
    ap.add_argument('--bench', type=int, default=0); ap.add_argument('--dry', action='store_true')
    a = ap.parse_args(); T0 = time.time()
    cfg = json.load(open(a.config)); NAME = cfg['name']; RUN = f'{SELF}/runs/{NAME}'
    if not a.bench and not a.dry:
        os.makedirs(RUN, exist_ok=True)
        if os.path.exists(f'{RUN}/done.json'): print('already done', RUN, flush=True); sys.exit(0)
        json.dump(cfg, open(f'{RUN}/config_train.json', 'w'), indent=1)
    STOP = {'flag': False}
    signal.signal(signal.SIGTERM, lambda *_: STOP.__setitem__('flag', True))
    Tr = cfg['training']
    micro, accum, n1447 = int(Tr.get('micro', 4)), int(Tr.get('accum', 4)), int(Tr.get('n1447', 2))
    lr, lr_min, wd = float(Tr['lr']), float(Tr['min_lr']), float(Tr.get('weight_decay', 1e-6))
    seed = int(Tr.get('seed', 0)); vram = float(Tr.get('vram_gb', 9.0)); STEPS = int(Tr['steps'])
    w306 = float(Tr.get('w306', 0.5)); save_steps = set(int(s) for s in Tr.get('save_steps', []))
    A = dict(L.AUG_DEFAULT, **cfg.get('augmentation', {})); smooth = float(cfg.get('loss', {}).get('smooth', 0.25))

    anchor = [L.src_1447(w) for w in cfg.get('anchor', ['w058', 'w060'])]
    s306 = [src_306(r, cfg['variant']) for r in cfg['regions']]
    for d in anchor: d['weight_map'] = None
    info = dict(anchor=[d['info'] for d in anchor], pseudo=[d['info'] for d in s306])
    print(f'{NAME}: anchor tiles {sum(len(d["org"]) for d in anchor)}, 0306B tiles {sum(len(d["org"]) for d in s306)} '
          f'(ink {sum(int(d["ink"].sum()) for d in s306)}) ({time.time() - T0:.0f}s)', flush=True)
    if not a.bench and not a.dry: json.dump(info, open(f'{RUN}/tiles.json', 'w'), indent=1)
    srcs = anchor + s306; na = len(anchor)
    aidx = np.concatenate([np.stack([np.full(len(d['org']), i), np.arange(len(d['org']))], 1) for i, d in enumerate(anchor)])
    pidx = np.concatenate([np.stack([np.full(len(d['org']), na + i), np.arange(len(d['org']))], 1) for i, d in enumerate(s306)])
    pink = np.concatenate([d['ink'] for d in s306]); pidx_ink = pidx[pink]

    def pairs(j):                                                   # micro-batch j -> (source, tile) pairs, deterministic
        g = np.random.default_rng([seed, 31337, j])
        out = [aidx[g.integers(len(aidx))] for _ in range(n1447)]
        for _ in range(micro - n1447):
            pool_ = pidx_ink if (len(pidx_ink) and g.random() < 0.5) else pidx
            out.append(pool_[g.integers(len(pool_))])
        return np.array(out)

    if a.dry:
        import cv2
        x, y, k, _ = L.make_micro(srcs, pairs(0), seed, 0, A)
        print('dry batch', x.shape, y.shape, float(y.mean()), float(k.mean()))
        rows = []
        for i in range(micro):
            im = (x[i, 12].numpy() * 255).astype(np.uint8); lb = (y[i, 0].numpy() * 255).astype(np.uint8); kp = (k[i, 0].numpy() * 255).astype(np.uint8)
            rows.append(np.hstack([im, lb, kp]))
        cv2.imwrite(f'{SELF}/logs/dry_{NAME}.png', cv2.resize(np.vstack(rows), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
        return

    dev = 'cuda'; torch.backends.cudnn.benchmark = True
    tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    torch.cuda.set_per_process_memory_fraction(min(1.0, vram / tot))
    init = cfg.get('init', 'base')
    model = L.load_base('base' if init in ('base', 'loo') or init.endswith('.safetensors') else init)
    init_sha = L.BASE_SHA
    if init.endswith('.safetensors'):
        from safetensors.torch import load_file
        model.load_state_dict(load_file(init), strict=True); init_sha = L.sha256(init)
    model = model.to(dev).train()
    BN = [m for m in model.modules() if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm3d))]
    GCK = Tr.get('grad_ckpt', 'block')                 # copied from m306/train/train_v8306.py: per-block activation checkpointing
    if GCK:
        import torch.utils.checkpoint as cp
        def wrap(mod):
            orig = mod.forward
            def fwd(*x): return cp.checkpoint(orig, *x, use_reentrant=False) if torch.is_grad_enabled() else orig(*x)
            mod.forward = fwd
        bb = model.backbone
        if GCK == 'block':
            for nm in ('layer1', 'layer2', 'layer3', 'layer4'):
                for blk in getattr(bb, nm): wrap(blk)
            for c_ in model.decoder.convs: wrap(c_)
        else:
            for nm in ('layer1', 'layer2', 'layer3', 'layer4'): wrap(getattr(bb, nm))
        if Tr.get('ckpt_stem', GCK == 'block'):
            class Stem(torch.nn.Module):
                def __init__(s_, b): super().__init__(); s_.b = [b]
                def forward(s_, x): b = s_.b[0]; return b.maxpool(b.relu(b.bn1(b.conv1(x))))
            stem = Stem(bb); wrap(stem)
            def bb_fwd(x, bb=bb, stem=stem):
                x = stem(x); x1 = bb.layer1(x); x2 = bb.layer2(x1); x3 = bb.layer3(x2); x4 = bb.layer4(x3); return [x1, x2, x3, x4]
            bb.forward = bb_fwd
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    step = 0
    if not a.bench and os.path.exists(f'{RUN}/state.pt'):
        ck = torch.load(f'{RUN}/state.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); step = int(ck['step'])
        print(f'resumed at step {step}/{STEPS}', flush=True)

    def save_state():
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'step': step}, f'{RUN}/state.pt.tmp')
        os.replace(f'{RUN}/state.pt.tmp', f'{RUN}/state.pt')

    def meta(s):
        return dict(name=NAME, step=s, steps=STEPS, effective_batch=micro * accum, micro=micro, n1447=n1447, w306=w306, lr=lr,
                    min_lr=lr_min, seed=seed, bn='train', grad_ckpt=GCK, data=info, augmentation=A, loss=dict(smooth=smooth), init=init,
                    init_sha256=init_sha, variant=cfg['variant'], regions=cfg['regions'],
                    layer_order='0306B: stored render order layers c-12..c+11 (= base v8-in lead-A order, asc); 1447 HF stacks reversed',
                    config=cfg)

    pool = ThreadPoolExecutor(int(Tr.get('workers', 6))); Q = queue.Queue(maxsize=3 * accum)
    END = {'end': False}

    def producer(s0):
        for s in range(s0, STEPS):
            for j in range(s * accum, (s + 1) * accum):
                if END['end']: return
                Q.put((s, j, L.make_micro(srcs, pairs(j), seed, j, A, pool)))

    threading.Thread(target=producer, args=(step,), daemon=True).start()
    logf = None if a.bench else open(f'{RUN}/train_log.jsonl', 'a')
    torch.cuda.reset_peak_memory_stats()
    last_state, n_run, t_chunk = time.time(), 0, time.time()
    run = dict(loss=[], la=[], lp=[])
    while step < STEPS:
        for g in opt.param_groups: g['lr'] = L.lr_at(step, STEPS, lr, lr_min, float(Tr.get('warmup_frac', 0.05)))
        opt.zero_grad(set_to_none=True)
        for j in range(step * accum, (step + 1) * accum):
            s, jj, (x, y, k, _) = Q.get(); assert s == step and jj == j
            x, y, k = x.to(dev, non_blocking=True), y.to(dev), k.to(dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lg = model(x)
            lg = lg.float()
            if GCK: after = [(m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone()) for m in BN]
            la, _, _ = L.loss_fn(lg[:n1447], y[:n1447], k[:n1447], None, smooth=smooth)
            lp, _, _ = L.loss_fn(lg[n1447:], y[n1447:], k[n1447:], None, smooth=smooth)
            loss = (1 - w306) * la + w306 * lp
            (loss / accum).backward()
            if GCK:                                     # undo the recompute's second BN running-stat update (as TRAIN)
                for m, (rm, rv, nb) in zip(BN, after):
                    m.running_mean.copy_(rm); m.running_var.copy_(rv); m.num_batches_tracked.copy_(nb)
            run['loss'].append(float(loss)); run['la'].append(float(la)); run['lp'].append(float(lp))
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float(Tr.get('clip', 1.0)))
        opt.step(); step += 1; n_run += 1
        if step % 10 == 0 or a.bench or step == STEPS:
            el = time.time() - t_chunk
            rec = dict(step=step, loss=round(float(np.mean(run['loss'])), 5), l1447=round(float(np.mean(run['la'])), 5),
                       l306=round(float(np.mean(run['lp'])), 5), lr=opt.param_groups[0]['lr'], gnorm=round(float(gn), 4),
                       tiles_per_s=round(n_run * micro * accum / max(el, 1e-6), 2),
                       peak_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), t=round(time.time() - T0, 1))
            run = dict(loss=[], la=[], lp=[]); print(json.dumps(rec), flush=True)
            if logf: logf.write(json.dumps(rec) + '\n'); logf.flush()
        if a.bench and step >= a.bench:
            print(f'BENCH {(time.time() - t_chunk) / n_run:.2f} s/step, peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GB', flush=True)
            END['end'] = True; os._exit(0)
        if step in save_steps and step < STEPS:
            model.eval(); s_ = L.save_ckpt(model, f'{RUN}/step_{step:04d}', meta(step)); model.train()
            print(f'saved step_{step:04d} {s_[:12]}', flush=True)
        if time.time() - last_state > 5 * 60 and step < STEPS:
            save_state(); last_state = time.time()
        if (STOP['flag'] or time.time() - T0 > a.max_minutes * 60) and step < STEPS:
            save_state(); print(f'chunk end at step {step}/{STEPS}', flush=True); END['end'] = True; os._exit(3)
    model.eval()
    s_ = L.save_ckpt(model, f'{RUN}/final', meta(step))
    json.dump(dict(name=NAME, steps=STEPS, final=f'{RUN}/final', sha256=s_, minutes_this_chunk=round((time.time() - T0) / 60, 1)),
              open(f'{RUN}/done.json', 'w'), indent=1)
    if os.path.exists(f'{RUN}/state.pt'): os.remove(f'{RUN}/state.pt')
    print('done', RUN, s_, flush=True); END['end'] = True; os._exit(0)


if __name__ == '__main__':
    main()
