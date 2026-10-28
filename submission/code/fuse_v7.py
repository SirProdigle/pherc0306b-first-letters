"""fuse_v7.py : the version-7 product-rule fusion, TRACE v3B x A, where A is the per-pixel mean of member v8-family maps.
The 1667 rule: a per-pixel product of two models' ink probabilities; ink must be seen by both.

  fuse_v7.py OUTDIR --trace TR.npy --members M1.npy M2.npy ... [--shuf-trace TRS.npy --shuf-members S1.npy ...] [--decoy D.npy]
             [--expect key=PATH ...]

Writes (4900 x 7780 float16, NaN where any input is NaN), with the coordinator's arithmetic exactly (m306/coord/ens_mean.py,
control_prod.py; rebuilt bit for bit on 23 Oct):
  fwd_ensA.npy   A = float16( (M1 + M2 + ...) / n ), sums in float32 in member order
  fwd_prod.npy   float16( float32(TR) * float32(A) )                         = the main image's map
  shuf_prod.npy  float16( float32(TRS) * mean_float32(S1, S2, ...) )         = the depth-shuffle control's map (claim box only)
  shuf_ensA.npy  float16( mean_float32(S1, S2, ...) )                        = A on the shuffled render (the 'A alone' view's control)
  fwd_prodd.npy  float16( float32(TR) * float32(D) )                         = the decoy control (TRACE x the decoy model)
  fuse.json      sha256 of every input and output; with --expect, whether each output is byte-identical to the given file.
The display (grey rule, median .. p99.9 of the context's valid pixels) is version 5's, applied to the product map as a single map
(make_claim_image_v2.py: the z-score of one map, then the grey levels; affine, so the same image as the raw product)."""
import os, sys, json, argparse, hashlib
import numpy as np


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()


def ld(p): return np.load(p).astype(np.float32)


def mean16(paths):
    acc = None
    for p in paths:
        a = ld(p); acc = a if acc is None else acc + a
    return (acc / len(paths)).astype(np.float16)


def save(out, name, a):
    p = f'{out}/{name}.npy'; np.save(p + '.tmp.npy', a); os.replace(p + '.tmp.npy', p); return p


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--trace', required=True); ap.add_argument('--members', nargs='+', required=True)
    ap.add_argument('--shuf-trace', default=''); ap.add_argument('--shuf-members', nargs='*', default=[]); ap.add_argument('--decoy', default='')
    ap.add_argument('--expect', nargs='*', default=[], help='key=PATH: compare an output (ensA, prod, shuf_prod, prodd) with a given file')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    rec = dict(rule='product of probabilities (TRACE v3B x A), A = mean of the member maps', inputs={}, outputs={}, expect={})
    for k, p in [('trace', a.trace)] + [(f'member{i}', m) for i, m in enumerate(a.members)] + \
                ([('shuf_trace', a.shuf_trace)] if a.shuf_trace else []) + [(f'shuf_member{i}', m) for i, m in enumerate(a.shuf_members)] + \
                ([('decoy', a.decoy)] if a.decoy else []):
        rec['inputs'][k] = dict(path=os.path.abspath(p), sha256=sha(p))
    A = mean16(a.members); out = {'ensA': save(a.out, 'fwd_ensA', A)}
    out['prod'] = save(a.out, 'fwd_prod', (ld(a.trace) * A.astype(np.float32)).astype(np.float16)); del A
    if a.shuf_trace and a.shuf_members:
        As = np.mean([ld(m) for m in a.shuf_members], 0)
        out['shuf_prod'] = save(a.out, 'shuf_prod', (ld(a.shuf_trace) * As).astype(np.float16))
        out['shuf_ensA'] = save(a.out, 'shuf_ensA', As.astype(np.float16)); del As
    if a.decoy: out['prodd'] = save(a.out, 'fwd_prodd', (ld(a.trace) * ld(a.decoy)).astype(np.float16))
    for k, p in out.items(): rec['outputs'][k] = dict(path=os.path.abspath(p), sha256=sha(p))
    ok = True
    for e in a.expect:
        k, p = e.split('=', 1)
        same = os.path.exists(p) and k in out and np.array_equal(np.load(out[k]), np.load(p), equal_nan=True)
        rec['expect'][k] = dict(path=os.path.abspath(p), identical=bool(same)); ok &= bool(same)
    json.dump(rec, open(f'{a.out}/fuse.json', 'w'), indent=1)
    print(json.dumps({k: v['identical'] for k, v in rec['expect'].items()} or {k: v['sha256'][:12] for k, v in rec['outputs'].items()}), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
