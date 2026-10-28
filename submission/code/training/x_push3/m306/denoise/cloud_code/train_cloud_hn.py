"""train_cloud_hn.py CONFIG.json (DENOISE copy of SELF's train_cloud.py + optional online hard-negative weighting, cfg loss.hn) [--max-minutes 600] : SELF lane trainer for rented GPUs.
Same recipe as code/train_self.py (which trained self_main), with the data read from pre-built, locally zone-checked packs
(code/make_packs.py) instead of renders / meshes:
  * 0306B sources = packs/306_<r>.npz (img + lab_<variant> / org_<variant> / ink_<variant>), identical arrays and tile lists to
    train_self.src_306, so with the same seed the micro-batch stream is identical;
  * anchors = packs/a_<name>.npz (TRAIN's v8306lib src_1447 / src_0343 / src_human outputs).  cfg 'anchor' = list of pack names;
    cfg training.group_weights (optional, e.g. {"1447": .5, "0343": .2, "h0139": .3}) = TRAIN's group quotas: each anchor slot
    draws a group with that probability, then a tile uniformly in the group (without it: uniform over all anchor tiles, = self_main);
  * model / augmentation / loss / checkpoints: TRAIN's v8306lib (copied unchanged next to this file); BN train mode at micro 4;
    training.grad_ckpt 'block' (as self_main) or false (bigger GPUs: identical maths; BN running stats are restored after the
    recompute anyway, so both give the same model up to float non-determinism).
Env: SELF_ROOT (runs/ and packs/ live here; default the local lane folder), V8_DIR (base v8-in dir with model.safetensors + ink8um).
Outputs SELF_ROOT/runs/<name>/ as train_self.py (state.pt resume, train_log.jsonl, step_<s>/, final/, done.json, tiles.json)."""
import argparse, json, os, signal, sys, time, threading, queue
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
V8_DIR = os.environ.get('V8_DIR', '/home/liam/projects/vesuvius-work/third_party/v8in'); sys.path.insert(0, V8_DIR)
if os.path.exists(f'{HERE}/v8306lib.py'): import v8306lib as L                   # cloud: copy shipped next to this file
else:
    sys.path.insert(0, '/home/liam/projects/vesuvius-work/.claude/worktrees/leadA-push3/x_push3/m306/train'); import v8306lib as L
L.BASE_DIR = V8_DIR
SELF = os.environ.get('SELF_ROOT', '/home/liam/projects/vesuvius-work/.claude/worktrees/leadA-push3/x_push3/m306/self')
PK = f'{SELF}/packs'


def _oom_exit(tp, val, tb):
    if issubclass(tp, torch.OutOfMemoryError) or 'out of memory' in str(val).lower():
        print(f'CUDA OOM ({val}); exit 3 for a retry', flush=True); os._exit(3)
    sys.__excepthook__(tp, val, tb)


def load_306(r, v):
    z = np.load(f'{PK}/306_{r}.npz')
    lab8 = z[f'lab_{v}']
    info = json.loads(str(z['info']))[v]
    return dict(img=z['img'], lab=(lab8 == 1).astype(np.float32), keep=(lab8 != 255).astype(np.uint8), org=z[f'org_{v}'],
                ink=z[f'ink_{v}'], info=info, name=r, weight_map=None)


def load_anchor(n):
    z = np.load(f'{PK}/a_{n}.npz')
    return dict(img=z['img'], lab=z['lab'].astype(np.float32), keep=z['keep'], org=z['org'], info=json.loads(str(z['info'])),
                name=n, group=str(z['group']), weight_map=None)


def main():
    sys.excepthook = _oom_exit
    ap = argparse.ArgumentParser()
    ap.add_argument('config'); ap.add_argument('--max-minutes', type=float, default=600.0)
    ap.add_argument('--bench', type=int, default=0)
    a = ap.parse_args(); T0 = time.time()
    cfg = json.load(open(a.config)); NAME = cfg['name']; RUN = f'{SELF}/runs/{NAME}'
    if not a.bench:
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
    HN = cfg.get('loss', {}).get('hn')                    # DENOISE: {'gain_306': g, 'gain_1447': g2} or None (= train_cloud.py)
    GW = Tr.get('group_weights')

    import glob
    names = []
    for n in cfg['anchor']:                                         # pack names; '*' patterns expand over packs/a_<pattern>.npz
        names += sorted(os.path.basename(p)[2:-4] for p in glob.glob(f'{PK}/a_{n}.npz')) if '*' in n else [n]
    anchor = [load_anchor(n) for n in names]
    s306 = [load_306(r, cfg['variant']) for r in cfg['regions']]
    info = dict(anchor=[d['info'] for d in anchor], pseudo=[d['info'] for d in s306], packs=json.load(open(f'{PK}/MANIFEST.json'))['files']
                if os.path.exists(f'{PK}/MANIFEST.json') else None)
    print(f'{NAME}: anchor tiles {sum(len(d["org"]) for d in anchor)}, 0306B tiles {sum(len(d["org"]) for d in s306)} '
          f'(ink {sum(int(d["ink"].sum()) for d in s306)}) ({time.time() - T0:.0f}s)', flush=True)
    if not a.bench: json.dump(info, open(f'{RUN}/tiles.json', 'w'), indent=1)
    srcs = anchor + s306; na = len(anchor)
    aidx = np.concatenate([np.stack([np.full(len(d['org']), i), np.arange(len(d['org']))], 1) for i, d in enumerate(anchor)])
    pidx = np.concatenate([np.stack([np.full(len(d['org']), na + i), np.arange(len(d['org']))], 1) for i, d in enumerate(s306)])
    pink = np.concatenate([d['ink'] for d in s306]); pidx_ink = pidx[pink]
    if GW:
        groups = sorted(GW); gp = np.array([GW[g] for g in groups], np.float64); gp /= gp.sum()
        g_of = np.array([anchor[i]['group'] for i in aidx[:, 0]]); gidx = {g: aidx[g_of == g] for g in groups}
        assert all(len(gidx[g]) for g in groups), {g: len(gidx[g]) for g in groups}

    def pairs(j):                                                   # = train_self.pairs when GW is None
        g = np.random.default_rng([seed, 31337, j])
        if GW:
            out = []
            for _ in range(n1447):
                gg = groups[int(g.choice(len(groups), p=gp))]; out.append(gidx[gg][g.integers(len(gidx[gg]))])
        else:
            out = [aidx[g.integers(len(aidx))] for _ in range(n1447)]
        for _ in range(micro - n1447):
            pool_ = pidx_ink if (len(pidx_ink) and g.random() < 0.5) else pidx
            out.append(pool_[g.integers(len(pool_))])
        return np.array(out)

    dev = 'cuda'; torch.backends.cudnn.benchmark = True
    tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    torch.cuda.set_per_process_memory_fraction(min(1.0, vram / tot))
    init = cfg.get('init', 'base')
    model = L.load_base('base'); init_sha = L.BASE_SHA
    if init.endswith('.safetensors'):
        from safetensors.torch import load_file
        model.load_state_dict(load_file(init), strict=True); init_sha = L.sha256(init)
    model = model.to(dev).train()
    BN = [m for m in model.modules() if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm3d))]
    GCK = Tr.get('grad_ckpt', 'block')                 # copied from m306/train/train_v8306.py: per-block activation checkpointing
    if os.environ.get('GCK_FORCE'): GCK = False if os.environ['GCK_FORCE'] == '0' else os.environ['GCK_FORCE']   # OOM fallback
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
                    min_lr=lr_min, seed=seed, bn='train', grad_ckpt=GCK, group_weights=GW, data=info, augmentation=A,
                    loss=dict(smooth=smooth, hn=HN), init=init, init_sha256=init_sha, variant=cfg['variant'], regions=cfg['regions'],
                    anchor=cfg['anchor'], trainer='code/train_cloud.py (packs)',
                    layer_order='0306B: stored render order layers c-12..c+11 (= base v8-in lead-A order, asc); 1447 HF stacks reversed; '
                                '0343 desc c33; 0139 human v2 forward', config=cfg)

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
            wa = wp = None
            if HN:                                           # DENOISE: online hard negatives = labelled-background px weighted by the
                with torch.no_grad():                        # model's own current ink probability: w = 1 + gain * [y == 0] * keep * p
                    pr = torch.sigmoid(lg); bg = (y == 0).float() * k.float()
                    if HN.get('gain_1447'): wa = 1 + float(HN['gain_1447']) * bg[:n1447] * pr[:n1447]
                    if HN.get('gain_306'): wp = 1 + float(HN['gain_306']) * bg[n1447:] * pr[n1447:]
            la, _, _ = L.loss_fn(lg[:n1447], y[:n1447], k[:n1447], wa, smooth=smooth)
            lp, _, _ = L.loss_fn(lg[n1447:], y[n1447:], k[n1447:], wp, smooth=smooth)
            loss = (1 - w306) * la + w306 * lp
            (loss / accum).backward()
            if GCK:
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
