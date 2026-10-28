"""evidence_grid_dn.py [--letters] : the line-1 evidence grid (evidence_grid_line1.py, unchanged) with the DENOISE lane's models as
the first rows: TRACE x A-hn (the v7 main image), A-hn x sqrt(Reader V3), C x sqrt(Reader V3), B-hn x sqrt(Reader V3), A-hn alone.
Derived maps are written once to out/dn_maps/ (full raster, float16, NaN where an input is NaN).  Output:
CANDIDATES/quick_compare_m306_20261023/evidence_grid_line1[_letters]_with_denoised.jpg (+ .json)."""
import os, sys, shutil
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import evidence_grid_line1 as G                                                    # noqa: E402
N = G.N
RAW = f'{G.M306}/denoise/maps/raw_dnt'; DM = f'{HERE}/out/dn_maps'


def ld(p): return np.load(p).astype(np.float32)


def derive():
    os.makedirs(DM, exist_ok=True)
    tr, v3, C = ld(N.MAPS['tr']), ld(f'{G.AM}/fwd_v3.npy'), ld(N.MAPS['C'])
    ahn = np.mean([ld(f'{RAW}/hn4_s{i}_d0_fwd_t1.npy') for i in range(3)], 0)
    bhn = ld(f'{RAW}/hn4B_s0_d0_fwd_t1.npy')
    sq = lambda a: np.sqrt(np.clip(a, 0, 1))                                       # noqa: E731  (NaN stays NaN)
    out = {'Ahn': ahn, 'TxAhn': np.clip(tr, 0, 1) * ahn, 'AhnV3': ahn * sq(v3), 'CV3': C * sq(v3), 'BhnV3': bhn * sq(v3)}
    for k, a in out.items():
        p = f'{DM}/{k}.npy'
        if not os.path.exists(p): np.save(p, a.astype(np.float16))
    return {k: f'{DM}/{k}.npy' for k in out}


def main():
    G.SRC.update(derive())
    new = [('NEW: noise-reduced models', 'TRACE x A-hn (v7 main image)', ('TxAhn',)),
           ('NEW: noise-reduced models', 'A-hn x sqrt(Reader V3)', ('AhnV3',)),
           ('NEW: noise-reduced models', 'C x sqrt(Reader V3)', ('CV3',)),
           ('NEW: noise-reduced models', 'B-hn x sqrt(Reader V3)', ('BhnV3',)),
           ('NEW: noise-reduced models', 'A-hn alone (A retrained on blank papyrus)', ('Ahn',))]
    G.ROWS[:0] = new
    tmp = f'{HERE}/out/grid_tmp'; os.makedirs(tmp, exist_ok=True); G.OUT = tmp
    G.main()
    name = 'evidence_grid_line1_letters' if G.LETTERS else 'evidence_grid_line1'
    for ext in ('jpg', 'json'):
        shutil.move(f'{tmp}/{name}.{ext}', f'{N.W}/CANDIDATES/quick_compare_m306_20261023/{name}_with_denoised.{ext}')
    print('wrote', name + '_with_denoised', flush=True)


if __name__ == '__main__':
    main()
