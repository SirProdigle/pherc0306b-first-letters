"""Train the glyph-likeness CNN on synthetic text composited into real negative windows.

usage: train.py OUTNAME [--n 400000] [--no-art] [--bs 128] [--epochs-split 8]
Never sees PHerc0343, PHerc0841, PHerc1447 or the zoom/deep hard negatives.
"""
import sys, time, json, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from common import *
import data


class GlyphNet(nn.Module):
    def __init__(self, cin=2, w=24):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.b1, self.b2, self.b3, self.b4 = blk(cin, w), blk(w, 2 * w), blk(2 * w, 4 * w), blk(4 * w, 8 * w)
        self.head = nn.Sequential(nn.Linear(16 * w, 64), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, a, m):
        x = torch.cat([a, m], 1)
        x = F.max_pool2d(self.b1(x), 2)      # 64
        x = F.max_pool2d(self.b2(x), 2)      # 32
        x = F.max_pool2d(self.b3(x), 2)      # 16
        x = self.b4(x)                       # 16x16 cells of 0.4 mm
        x = torch.cat([x.mean((2, 3)), x.amax((2, 3))], 1)
        return self.head(x).squeeze(1)


def load_model(path, device='cuda'):
    ck = torch.load(path, map_location=device)
    net = GlyphNet(**ck.get('arch', {})).to(device).eval()
    net.load_state_dict(ck['state'])
    return net


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--n', type=int, default=400000); ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--no-art', action='store_true'); ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--workers', type=int, default=6); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--realpos', default='', help='comma list of held-out sets used as REAL positives (LODO fine-tune)')
    ap.add_argument('--legib', action='store_true', help='add real legibility-labelled crops (other scrolls) as positives/negatives')
    ap.add_argument('--memfrac', type=float, default=0.2)
    ap.add_argument('--init', default='', help='initial checkpoint')
    ap.add_argument('--p', default='0.45,0.25,0.05', help='p_text,p_art,p_both (rest = real negatives)')
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    torch.cuda.set_per_process_memory_fraction(a.memfrac)
    dev = 'cuda'
    tr = data.load_corpus(('train',)); va = data.load_corpus(('heldneg',))
    print('train items', len(tr), 'val items', len(va), flush=True)
    ds = data.SynthDS(tr, a.n, seed=a.seed + 1, artefacts=not a.no_art, p_text=float(a.p.split(',')[0]), p_art=float(a.p.split(',')[1]), p_both=float(a.p.split(',')[2]))
    if a.legib:
        T, yl, sc = data.load_legib_cohort()
        print('legib cohort', T.shape, 'pos', int(yl.sum()), 'neg', int((yl == 0).sum()), flush=True)
        ds.set_legib(T, yl)
    if a.realpos:
        rp, rn = data.load_realpos(a.realpos.split(','))
        print('real pos items', [r['name'] for r in rp], 'real neg items', [r['name'] for r in rn], flush=True)
        ds.set_real(rp, rn)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, num_workers=a.workers,
                                     persistent_workers=False, prefetch_factor=4, multiprocessing_context='fork')
    # fixed synthetic validation set from held-out negative patches
    vds = data.SynthDS(va, 6000, seed=999, artefacts=True)
    V = list(torch.utils.data.DataLoader(vds, batch_size=128, num_workers=4, multiprocessing_context='fork'))
    net = GlyphNet().to(dev)
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location=dev)['state'])
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.n // a.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    t0 = time.time(); step = 0; run = 0
    log = []
    for x, m, y, k in dl:
        net.train()
        x, m, y = x.to(dev, non_blocking=True), m.to(dev), y.to(dev)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logit = net(x, m)
        loss = F.binary_cross_entropy_with_logits(logit.float(), y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
        step += 1
        if step % 300 == 0 or step == steps:
            net.eval(); S, Y, K = [], [], []
            with torch.no_grad():
                for vx, vm, vy, vk in V:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        S.append(net(vx.to(dev), vm.to(dev)).float().cpu()); Y.append(vy); K.append(vk)
            S, Y, K = torch.cat(S).numpy(), torch.cat(Y).numpy(), torch.cat(K).numpy()
            r = dict(step=step, loss=round(run, 4), secs=round(time.time() - t0),
                     auc_all=round(auc(S[Y == 1], S[Y == 0]), 4),
                     auc_vs_real=round(auc(S[K == 1], S[K == 0]), 4), auc_vs_art=round(auc(S[K == 1], S[K == 2]), 4))
            log.append(r); print(json.dumps(r), flush=True)
            torch.save(dict(state=net.state_dict(), arch={}, args=vars(a), log=log), f'{G}/models/{a.out}.pt')
        if step >= steps:
            break
    print('done', flush=True)
