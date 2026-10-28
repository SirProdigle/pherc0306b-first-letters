"""build_variants.py --maps DIR --ct CT --mesh MESHDIR --out DIR [--config CFG.json] [--lead VARIANT] [--layout package|flat] [--only a,b]
The registry of display variants of the package and their builder (v2 views + v3's Reader V3 views; version 5: the registry comes
from the build config).

Each variant = (pair, display rule, rung, background), all fixed in advance (I1 PREREG A1.1, A5, A6; I2's background rule).
Without --config the registry is version 4c's (below).  With --config it is the config's 'views' list, in its order; a view's
rung 'auto' must have been resolved by build_v5.py (step rungs: the A5 rule on the pre-registered blank control) before this runs,
so the shipped config holds fixed rungs only.
layout package: the lead -> OUT/<MESH>.png (+ .json), every other variant -> OUT/supplementary/<file>.png (+ .json)
layout flat:    every variant -> OUT/variants/<file>.png, and the lead copied to OUT/<MESH>.png (Docker).
--jobs N (version 7): N views at once, each in its own make_claim_image_v2.py process as before (same files)."""
import os, sys, json, shutil, argparse, subprocess

VARIANTS = {   # version 4c's registry (v3, 20 Oct 2026: Reader V3 pairs added; the v2 views stay as supplementary views)
    'v3t_grey': dict(view='1', pair='v3t', display='grey', rung='primary', bg='page', file='V3_TRACE_grey_page'),
    'v3t_secondary': dict(view='2', pair='v3t', display='threshold', rung='secondary', bg='ct', file='V3_TRACE_secondary_rung_L050'),
    'v3_grey': dict(view='3', pair='v3', display='grey', rung='primary', bg='page', file='V3_grey_page'),
    'v3q_grey': dict(view='3b', pair='v3q', display='grey', rung='primary', bg='page', file='V3_v8Q_grey_page'),
    'rt_primary': dict(view='1', pair='rt', display='threshold', rung='primary', bg='ct', file='Rv2_TRACE_primary'),
    'rt_secondary': dict(view='2', pair='rt', display='threshold', rung='secondary', bg='ct', file='Rv2_TRACE_secondary_rung_L050'),
    'rv8_secondary': dict(view='3', pair='rv8', display='threshold', rung='secondary', bg='ct', file='Rv2_v8in_secondary_rung_L050'),
    'rv8_primary': dict(view='-', pair='rv8', display='threshold', rung='primary', bg='ct', file='Rv2_v8in_primary'),
    'rv8q_secondary': dict(view='4', pair='rv8q', display='threshold', rung='secondary', bg='ct', file='Rv2_v8Q_secondary_rung_L050'),
    'rv8q_grey': dict(view='5', pair='rv8q', display='grey', rung='primary', bg='page', file='Rv2_v8Q_grey_page'),
    'rt_grey': dict(view='6', pair='rt', display='grey', rung='primary', bg='page', file='Rv2_TRACE_grey_page'),
}
PAIRS = {'rt': ('rv2', 'tr'), 'rv8': ('rv2', 'v8'), 'rv8q': ('rv2', 'v8q'), 'v3': ('v3',), 'v3t': ('v3', 'tr'), 'v3q': ('v3', 'v8q')}
LEAD = 'v3t_grey'
HERE = os.path.dirname(os.path.abspath(__file__))
FIELDS = ('view', 'pair', 'display', 'rung', 'bg', 'file')


def registry(cfg_path=''):
    """-> (VARIANTS, PAIRS, default lead) from the config, or version 4c's"""
    if not cfg_path: return VARIANTS, PAIRS, LEAD
    cfg = json.load(open(cfg_path))
    var = {v['key']: {f: v[f] for f in FIELDS} for v in cfg['views']}
    bad = [k for k, v in var.items() if v['rung'] not in ('primary', 'secondary')]
    if bad: raise SystemExit(f'unresolved rung in {bad}: run build_v5.py (step rungs) first')
    lead = next(v['key'] for v in cfg['views'] if v.get('lead'))
    return var, {k: tuple(p) for k, p in cfg['pairs'].items()}, lead


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--maps', required=True); ap.add_argument('--ct', required=True)
    ap.add_argument('--mesh', required=True); ap.add_argument('--out', required=True); ap.add_argument('--lead', default='')
    ap.add_argument('--layout', default='package', choices=('package', 'flat')); ap.add_argument('--only', default='')
    ap.add_argument('--config', default=''); ap.add_argument('--jobs', type=int, default=1)
    a = ap.parse_args(); mesh = os.path.basename(os.path.normpath(a.mesh))
    var, pairs, lead = registry(a.config); a.lead = a.lead or lead
    if a.lead not in var: raise SystemExit(f'--lead {a.lead}: not in the registry {list(var)}')
    todo = [k for k in var if not a.only or k in a.only.split(',')]
    have = {k for k in todo if all(os.path.exists(f'{a.maps}/fwd_{m}.npy') for m in pairs[var[k]['pair']])}
    jobs = []                                                                       # version 7: --jobs N runs N views at once (same subprocess per view)
    for k in todo:
        if k not in have: print('skip (maps missing):', k); continue
        v = var[k]
        if a.layout == 'package':
            dest, name = (a.out, mesh) if k == a.lead else (f'{a.out}/supplementary', v['file'])
        else:
            dest, name = f'{a.out}/variants', v['file']
        os.makedirs(dest, exist_ok=True)
        cmd = [sys.executable, f'{HERE}/make_claim_image_v2.py', '--maps', a.maps, '--ct', a.ct, '--mesh', a.mesh, '--out', dest, '--name', name,
               '--pair', v['pair'], '--rung', v['rung'], '--display', v['display'], '--bg', v['bg']] + (['--config', a.config] if a.config else [])
        jobs.append((k, v, dest, name, cmd))

    def finish(k, v, dest, name, rc, err):
        if rc: print(err[-2000:]); raise SystemExit(f'{k} failed')
        j = json.load(open(f'{dest}/{name}.json')); j['variant'] = dict(key=k, **v, lead=k == a.lead); json.dump(j, open(f'{dest}/{name}.json', 'w'), indent=1, ensure_ascii=False)
        print(f'view {v["view"]:>2} {k:15s} -> {dest}/{name}.png  {j["image_sha256"][:12]}', flush=True)

    if a.jobs <= 1:
        for k, v, dest, name, cmd in jobs:
            r = subprocess.run(cmd, capture_output=True, text=True); finish(k, v, dest, name, r.returncode, r.stderr)
    else:
        import time
        run, todo_j = [], list(jobs)
        while todo_j or run:
            while todo_j and len(run) < a.jobs:
                k, v, dest, name, cmd = todo_j.pop(0)
                import tempfile
                ef = tempfile.TemporaryFile(mode='w+')                                 # a file, not a pipe: a chatty child can never block
                run.append((k, v, dest, name, subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=ef, text=True), ef))
            for item in list(run):
                if item[4].poll() is not None:
                    run.remove(item); item[5].seek(0); finish(*item[:4], item[4].returncode, item[5].read()); item[5].close()
            time.sleep(0.5)
    if a.layout == 'flat' and a.lead in have:
        f = var[a.lead]['file']
        shutil.copy2(f'{a.out}/variants/{f}.png', f'{a.out}/{mesh}.png'); shutil.copy2(f'{a.out}/variants/{f}.json', f'{a.out}/{mesh}.json')


if __name__ == '__main__':
    main()
