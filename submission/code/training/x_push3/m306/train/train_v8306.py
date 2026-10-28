"""train_v8306.py : M306 TRAIN fine-tune of v8-in, the organisers' LOO-w062 recipe.

    envs/vcpy/bin/python m306/train/train_v8306.py CONFIG.json [--max-minutes 26] [--bench N] [--dry]

Run through P3/gpu_run.sh NAME ... (resumable: exits 3 at a chunk end / SIGTERM, 0 when done; CUDA OOM -> exit 3).
Recipe (v8306lib docstring): all weights trained, BatchNorm in train mode at micro-batch `micro`, AdamW (wd 1e-6) lr 1e-5 ->
1e-6 per-step cosine, grad-norm clip 1.0, bf16 autocast, effective batch micro x accum = 32, epochs over the union tile list.
VRAM capped with set_per_process_memory_fraction(vram_gb); optional activation checkpointing of the encoder stages (BN running
statistics are restored after backward so the recompute does not update them twice).
Outputs runs/<name>/: state.pt (resume), train_log.jsonl, step_<s>/ and final/ = {model.safetensors, config.json, meta.json}
(InkDetector.from_pretrained(dir) loads them; config reverse_layers = true = 1447 HF stacks read 23 -> 0), done.json.
"""
import argparse, json, os, signal, sys, time, threading, queue
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import v8306lib as L                                                               # noqa: E402


def _oom_exit(tp, val, tb):
    if issubclass(tp, torch.OutOfMemoryError) or 'out of memory' in str(val).lower():
        print(f'CUDA OOM ({val}); exit 3 for a retry', flush=True); os._exit(3)
    sys.__excepthook__(tp, val, tb)


sys.excepthook = _oom_exit
ap = argparse.ArgumentParser()
ap.add_argument('config'); ap.add_argument('--max-minutes', type=float, default=26.0)
ap.add_argument('--bench', type=int, default=0, help='run N optimiser steps, report speed + peak VRAM, write nothing')
ap.add_argument('--override', default=None, help='JSON merged into the config (bench only)')
a = ap.parse_args()
T0 = time.time()
cfg = json.load(open(a.config))
if a.override:
    if not a.bench: raise SystemExit('--override only with --bench')
    cfg.update(json.loads(a.override))
NAME = cfg['name']; RUN = f'{L.TR}/runs/{NAME}'
if not a.bench:
    os.makedirs(RUN, exist_ok=True)
    if os.path.exists(f'{RUN}/done.json'): print('already done', RUN, flush=True); sys.exit(0)
    json.dump(cfg, open(f'{RUN}/config_train.json', 'w'), indent=1)
STOP = {'flag': False}
signal.signal(signal.SIGTERM, lambda *_: STOP.__setitem__('flag', True))

T = cfg.get('training', {})
micro, accum = int(T.get('micro', 4)), int(T.get('accum', 8))
lr, lr_min, wd = float(T.get('lr', 1e-5)), float(T.get('min_lr', 1e-6)), float(T.get('weight_decay', 1e-6))
seed = int(T.get('seed', 0)); bn_mode = T.get('bn', 'train'); vram = float(T.get('vram_gb', 9.0))
state_min = float(T.get('state_minutes', 5)); save_steps = set(int(s) for s in T.get('save_steps', []))
A = dict(L.AUG_DEFAULT, **cfg.get('augmentation', {}))
LS = cfg.get('loss', {}); smooth = float(LS.get('smooth', 0.25))

# ---------------------------------------------------------------------------------------------------------------- data
sources = L.load_sources(cfg)
stream = L.Stream(sources, micro, accum, seed, epochs=T.get('epochs'), steps=T.get('steps'), group_weights=T.get('group_weights'))
STEPS = stream.steps
info = [d['info'] for d in sources]
print(f'{NAME}: {stream.n} tiles, {stream.per_epoch} micro-batches/epoch, {STEPS} steps x {micro}x{accum}; '
      f'data {json.dumps(info)} ({time.time() - T0:.0f}s)', flush=True)

# ---------------------------------------------------------------------------------------------------------------- model
dev = 'cuda'
torch.backends.cudnn.benchmark = True
tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
torch.cuda.set_per_process_memory_fraction(min(1.0, vram / tot))
model = L.load_base(cfg.get('init', 'base')).to(dev)
model.train()
if bn_mode == 'frozen':
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)): m.eval()
BN = [m for m in model.modules() if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm3d))]
GCK = T.get('grad_ckpt', False)                                                 # False | 'layer' | 'block' (+ stem)
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
        for c in model.decoder.convs: wrap(c)
    else:
        for nm in ('layer1', 'layer2', 'layer3', 'layer4'): wrap(getattr(bb, nm))
    if T.get('ckpt_stem', GCK == 'block'):                                         # conv1 + bn1 + relu + maxpool
        class Stem(torch.nn.Module):
            def __init__(s_, b): super().__init__(); s_.b = [b]
            def forward(s_, x): b = s_.b[0]; return b.maxpool(b.relu(b.bn1(b.conv1(x))))
        stem = Stem(bb); wrap(stem)
        def bb_fwd(x, bb=bb, stem=stem):
            x = stem(x); x1 = bb.layer1(x); x2 = bb.layer2(x1); x3 = bb.layer3(x2); x4 = bb.layer4(x3); return [x1, x2, x3, x4]
        bb.forward = bb_fwd
opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
step, start = 0, time.time()
if not a.bench and os.path.exists(f'{RUN}/state.pt'):
    ck = torch.load(f'{RUN}/state.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); step = int(ck['step'])
    print(f'resumed at step {step}/{STEPS}', flush=True)


def save_state():
    if a.bench: return
    torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'step': step}, f'{RUN}/state.pt.tmp')
    os.replace(f'{RUN}/state.pt.tmp', f'{RUN}/state.pt')


def meta(s):
    return dict(name=NAME, step=s, steps=STEPS, effective_batch=micro * accum, micro=micro, lr=lr, min_lr=lr_min, seed=seed,
                bn=bn_mode, grad_ckpt=GCK, data=info, augmentation=A, loss=dict(smooth=smooth), init=cfg.get('init', 'base'),
                init_sha256=L.BASE_SHA, layer_order='1447 HF stacks reversed (reverse_layers true); lead A: as base v8-in',
                config=cfg)


# ---------------------------------------------------------------------------------------------------------------- loader
pool = ThreadPoolExecutor(int(T.get('workers', 6)))
Q = queue.Queue(maxsize=3 * accum)


def producer(s0):
    for s in range(s0, STEPS):
        for j in stream.step_micros(s):
            Q.put((s, j, L.make_micro(sources, stream.micro_batch(j), seed, j, A, pool)))
            if STOP.get('end'): return


threading.Thread(target=producer, args=(step,), daemon=True).start()
logf = None if a.bench else open(f'{RUN}/train_log.jsonl', 'a')
torch.cuda.reset_peak_memory_stats()
last_state, n_steps_run, t_chunk = time.time(), 0, time.time()
run = dict(loss=[], dice=[], bce=[])
while step < STEPS:
    for g in opt.param_groups: g['lr'] = L.lr_at(step, STEPS, lr, lr_min, float(T.get('warmup_frac', 0.0)))
    opt.zero_grad(set_to_none=True)
    mics = stream.step_micros(step)
    for j in mics:
        s, jj, (x, y, k, w) = Q.get()
        assert s == step and jj == j, (s, step, jj, j)
        x, y, k = x.to(dev, non_blocking=True), y.to(dev), k.to(dev)
        w = None if w is None else w.to(dev)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lg = model(x)
        if GCK and bn_mode == 'train':
            after = [(m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone()) for m in BN]
        loss, dl, bl = L.loss_fn(lg.float(), y, k, w, smooth=smooth)
        (loss / len(mics)).backward()
        if GCK and bn_mode == 'train':                                             # undo the recompute's second BN update
            for m, (rm, rv, nb) in zip(BN, after):
                m.running_mean.copy_(rm); m.running_var.copy_(rv); m.num_batches_tracked.copy_(nb)
        run['loss'].append(float(loss)); run['dice'].append(dl); run['bce'].append(bl)
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float(T.get('clip', 1.0)))
    opt.step(); step += 1; n_steps_run += 1
    if step % 10 == 0 or a.bench or step == STEPS:
        el = time.time() - t_chunk
        rec = dict(step=step, loss=round(float(np.mean(run['loss'])), 5), dice=round(float(np.mean(run['dice'])), 5),
                   bce=round(float(np.mean(run['bce'])), 5), lr=opt.param_groups[0]['lr'], gnorm=round(float(gn), 4),
                   tiles_per_s=round(n_steps_run * micro * accum / max(el, 1e-6), 2),
                   peak_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), t=round(time.time() - T0, 1))
        run = dict(loss=[], dice=[], bce=[])
        print(json.dumps(rec), flush=True)
        if logf: logf.write(json.dumps(rec) + '\n'); logf.flush()
    if a.bench and step >= a.bench:
        print(f'BENCH micro {micro} accum {accum} bn {bn_mode} gck {GCK}: {(time.time() - t_chunk) / n_steps_run:.2f} s/step, '
              f'peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GB', flush=True)
        STOP['end'] = True; os._exit(0)
    if step in save_steps and step < STEPS:
        s_ = L.save_ckpt(model, f'{RUN}/step_{step:04d}', meta(step)); print(f'saved step_{step:04d} {s_[:12]}', flush=True)
    if time.time() - last_state > state_min * 60 and step < STEPS:
        save_state(); last_state = time.time()
    if (STOP['flag'] or time.time() - T0 > a.max_minutes * 60) and step < STEPS:
        save_state(); print(f'chunk end at step {step}/{STEPS}', flush=True); STOP['end'] = True; os._exit(3)

model.eval()
s_ = L.save_ckpt(model, f'{RUN}/final', meta(step))
json.dump(dict(name=NAME, steps=STEPS, final=f'{RUN}/final', sha256=s_, minutes_total_this_chunk=round((time.time() - T0) / 60, 1)),
          open(f'{RUN}/done.json', 'w'), indent=1)
if os.path.exists(f'{RUN}/state.pt'): os.remove(f'{RUN}/state.pt')
print('done', RUN, s_, flush=True)
STOP['end'] = True; os._exit(0)
