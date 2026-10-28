#!/usr/bin/env python
"""build_v7.py CONFIG.json [--map FWD.npy] [--shuf-map SHUF.npy] [--out DIR] [--steps a,b,...] [--skip a,b] [--jobs N] [--preflight]

Version 7 of the lead-A claim package (PHerc0306B w074, mesh V5Rc), for the M306 finalist: one config (make_config_v7.py) gives the
whole package in version 5's layout (claim image, views, controls, letter-free geometry, README, CHANGES) plus the model's method
records, the exclusion-zone audit of its training tiles and the owner-only form text.  CPU only.
Version 5's kit (P3/v5) is copied, never edited; v6's base package is version 5's C3 package (P3/v5/build/v5_C3).
Run it as
  gaming-mode run -- systemd-run --user --scope -q -p MemoryMax=8G nice -n 19 <python> build_v7.py CONFIG --map FWD.npy --jobs 4

Steps (default: all, in this order):
  prep      the provenance guard (checkpoint sha256; the finalist map's json: same checkpoint, TTA, stride, window, render), maps dir of
            symlinks, preflight (files, shapes, header widths) -> <out>.work/guard.json, preflight.json
  rungs .. tables   version 5's steps, unchanged (views run --jobs N at once)
  exclusion the training-tile audit: SELF lane = every PHerc0306B training tile recomputed from the labels and masks exactly as
            train_self.py builds them, checked with aclick/p306/exclusion/check.py (centre + 4 corners at pad 0.7 mm as in training,
            and a 5 x 5 grid over each tile at pad 0: every input pixel >= 2.2 mm in 3-D from the zone); TRAIN lane = no PHerc0306B
            source at all -> methodology/model_v7/exclusion_audit.json (+ training_tiles.npz).  A failed audit stops the build.
  method    methodology/model_v7/: training config, run config, checkpoint metas + sha256 of every stage, train log, tile and label
            statistics, teacher recipes, data sources and licences, the exclusion check code + zone; evidence/heldout_v7/ (held-out
            known-text numbers from the M306 evaluator)
  assemble  the rest of the package from the base package (version 5's files, kept), v6 code/ (+ config_v7.json, training code
            copies in code/training/) and docker/, SHA256SUMS
  text      README.md (version 7 section + version 5's README, kept), CHANGES.md, methodology/model_v7/MODEL.md from the templates;
            the owner-only form text -> <out>.work/text/FORM_owner_only.md; prints every send blocker
  sums      SHA256SUMS again (after any later edit; also part of 'all')
A config with "dry_run": true is built the same way, but every image footer, the README and the form say DRY RUN.
Layout: <out>/ holds the package files only; <out>.work/ (beside it) holds maps symlinks, logs, tables, form images and text."""
import os, sys, json, time, shutil, argparse, subprocess, hashlib, glob, re
HERE = os.path.dirname(os.path.abspath(__file__)); CODE = f'{HERE}/code'
P3 = os.path.dirname(HERE); W = '/home/liam/projects/vesuvius-work'
sys.path.insert(0, CODE)
os.environ.setdefault('OMP_NUM_THREADS', '4'); os.environ.setdefault('MKL_NUM_THREADS', os.environ['OMP_NUM_THREADS'])
os.environ.setdefault('OPENBLAS_NUM_THREADS', os.environ['OMP_NUM_THREADS'])
STEPS = ['prep', 'rungs', 'views', 'controls', 'shuffle', 'blank', 'boxcount', 'mesh', 'letters', 'side', 'mirror', 'panel', 'overlay', 'tables',
         'exclusion', 'method', 'assemble', 'text', 'sums']
DEFAULT = list(STEPS)
RAST = (4900, 7780)
V5RC = f'{W}/x_refine/trials/P1_surface_refine/renders/P1V5Rc.zarr'
SHUF_RENDER = f'{W}/x_req/20261020/A10/v4c_work/shuf_render.zarr'
EXCL = f'{P3}/aclick/p306/exclusion'
EVAL = f'{P3}/m306/eval'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()


def log(B, step, msg, **kw):
    line = dict(t=time.strftime('%F %T'), step=step, msg=msg, **kw); print(f"[{line['t']}] {step}: {msg}", flush=True)
    with open(f'{B.work}/build_log.jsonl', 'a') as f: f.write(json.dumps(line) + '\n')


def rd(p, default=None):
    return json.load(open(p)) if os.path.exists(p) else ({} if default is None else default)


class Build:
    def __init__(self, cfg_path, out, jobs=1):
        self.cfg_path = os.path.abspath(cfg_path); self.cfg = json.load(open(cfg_path)); self.out = os.path.abspath(out)
        self.tag = self.cfg['tag']; self.mesh = self.cfg['mesh']['name']; self.jobs = jobs
        self.work = f'{self.out}.work'
        self.maps = f'{self.work}/maps'; self.shuf = f'{self.work}/shuf_maps'; self.resolved = f'{self.work}/config_resolved.json'
        self.F = self.cfg['finalist']; self.dry = bool(self.cfg.get('dry_run'))
        self.MD = f'{self.out}/methodology/model_v7'
        os.makedirs(self.work, exist_ok=True)

    def rcfg(self):
        return json.load(open(self.resolved)) if os.path.exists(self.resolved) else self.cfg

    def views(self):
        return self.rcfg()['views']

    def lead(self):
        return next(v for v in self.views() if v.get('lead'))

    def ctx(self):
        import controls_v5 as CV
        return CV.Ctx(self.rcfg(), self.maps, f'{self.maps}/ct20.png', self.shuf if os.path.isdir(self.shuf) else '')


# ---------------------------------------------------------------- the provenance guard
def guard(B):
    """the finalist's checkpoint(s) and maps must belong together; returns (problems, blockers, record)"""
    F = B.F; e = F.get('expect', {}); problems, blockers = [], []
    rec = dict(finalist=F['name'], dry_run=B.dry, ckpts=[], stages=[])
    shas = set()
    for c in F['ckpts']:
        if not os.path.exists(c['path']): problems.append(f'checkpoint missing: {c["path"]}'); continue
        s = sha(c['path']); shas.add(s)
        if c.get('sha256') and s != c['sha256']: problems.append(f'checkpoint sha256 {s[:12]} != config {c["sha256"][:12]} ({c["path"]})')
        rec['ckpts'].append(dict(role=c['role'], path=c['path'], sha256=s, bytes=os.path.getsize(c['path'])))
    for c in F.get('stages', []):
        if os.path.exists(c['path']):
            s = sha(c['path']); rec['stages'].append(dict(role=c['role'], path=c['path'], sha256=s, bytes=os.path.getsize(c['path'])))
            if c.get('sha256') and s != c['sha256']: problems.append(f'stage checkpoint sha256 mismatch ({c["path"]})')
        else: blockers.append(f'stage checkpoint missing (release needs every stage): {c["path"]}')

    def check_map(p, render, kind, want=None):
        j = p[:-4] + '.json'
        if not os.path.exists(p): return [f'{kind} map missing: {p}'], None
        if not os.path.exists(j): return [f'{kind} map has no provenance json: {j}'], None
        m = json.load(open(j)); bad = []
        ms = m.get('ckpt_sha256') or m.get('weights_sha256')
        if ms not in ({want} if want else shas): bad.append(f'{kind} map made by checkpoint {str(ms)[:12]}, not by {want[:12] if want else ", ".join(s[:12] for s in shas)}')
        if int(m.get('tta', -1)) != int(e.get('tta', 4)): bad.append(f'{kind} map TTA {m.get("tta")} != {e.get("tta", 4)}')
        if float(m.get('nsig', -1)) != float(e.get('nsig', 3)): bad.append(f'{kind} map stitch +-{m.get("nsig")} sigma != {e.get("nsig", 3)}')
        if int(m.get('stride', -1)) != int(e.get('stride', 21)): bad.append(f'{kind} map stride {m.get("stride")} != 21')
        if m.get('order', 'asc') != e.get('order', 'asc'): bad.append(f'{kind} map order {m.get("order")} != {e.get("order", "asc")}')
        if os.path.realpath(m.get('zarr', '')) != os.path.realpath(render): bad.append(f'{kind} map render {m.get("zarr")} != {render}')
        if m.get('layers', '8..31') != '8..31' or int(m.get('layer0', 8)) != 8: bad.append(f'{kind} map layers {m.get("layers")} != 8..31')
        if render == V5RC and m.get('region') != e.get('region', 'ctx'): bad.append(f'{kind} map region {m.get("region")} != ctx')
        if list(m.get('full_raster', RAST)) != list(RAST): bad.append(f'{kind} map raster {m.get("full_raster")}')
        return bad, dict(path=p, json=j, sha256=sha(p), meta={k: m.get(k) for k in ('ckpt_sha256', 'tta', 'nsig', 'stride', 'order', 'region',
                                                                                       'zarr', 'layers', 'finished', 'recipe', 'tiles')})
    if F.get('fusion') == 'product':                                                 # version 7 product rule: every member map, the decoy, TRACE
        rec['members'] = []
        for mm in F['members']:
            bad, r1 = check_map(mm['map'], V5RC, f'member {mm["name"]}', mm['sha256']); problems += bad
            bad, r2 = check_map(mm['shuf_map'], SHUF_RENDER, f'member {mm["name"]} shuffle', mm['sha256']); problems += bad
            rec['members'].append(dict(name=mm['name'], ckpt_sha256=mm['sha256'], map=r1, shuf_map=r2))
        d = F['decoy']
        if not os.path.exists(d['ckpt']) or sha(d['ckpt']) != d['sha256']: problems.append(f'decoy checkpoint missing or sha256 mismatch: {d["ckpt"]}')
        bad, rec['decoy_map'] = check_map(d['map'], V5RC, 'decoy', d['sha256']); problems += bad
        c3tr = B.cfg['models']['tr']['map']
        if os.path.realpath(F['trace_map']) != os.path.realpath(c3tr): problems.append(f'TRACE map {F["trace_map"]} is not version 5\'s ({c3tr})')
        rec['trace_map'] = dict(path=F['trace_map'], sha256=sha(F['trace_map'])); rec['trace_shuf_map'] = dict(path=F['trace_shuf_map'], sha256=sha(F['trace_shuf_map']))
        if F.get('selection_note_is_draft'): blockers.append('selection sentence is a draft: the owner confirms it (candidates/<name>.json selection_note)')
        if F.get('decoy_note_is_draft'): blockers.append('decoy sentence is a draft: the owner confirms it against supplementary/ctrl_panel_v7.jpg (decoy_note)')
        if F.get('status') == 'review' and not B.cfg.get('approved'): blockers.insert(0, 'OWNER REVIEW draft: not approved; after review rebuild the text with --approved')
        return problems, blockers, rec
    bad, rec['map'] = check_map(B.cfg['models'][F['model_key']]['map'], V5RC, 'main'); problems += bad
    sp = B.cfg.get('shuffle_maps', {}).get(F['model_key'])
    if sp:
        bad, rec['shuf_map'] = check_map(sp, SHUF_RENDER, 'shuffle')
        if bad and not os.path.exists(sp): blockers += bad
        else: problems += bad
    else: blockers.append('the finalist has no depth-shuffled map yet (GPU step): the main image\'s depth-shuffle control is pending')
    if B.dry: blockers.insert(0, 'DRY RUN package: stand-in map; never send')
    return problems, blockers, rec


# ---------------------------------------------------------------- version 5's steps (unchanged arithmetic)
def step_prep(B):
    import numpy as np, cv2
    cfg = B.cfg; problems = []
    gp, blockers, grec = guard(B)
    json.dump(dict(grec, problems=gp, send_blockers=blockers), open(f'{B.work}/guard.json', 'w'), indent=1)
    if gp: raise SystemExit('provenance guard failed:\n  ' + '\n  '.join(gp))
    if B.F.get('fusion') == 'product': fuse(B)
    if os.path.isdir(B.maps): shutil.rmtree(B.maps)
    os.makedirs(B.maps)
    for k, m in cfg['models'].items():
        if not os.path.exists(m['map']): problems.append(f'model {k}: map missing {m["map"]}'); continue
        os.symlink(os.path.abspath(m['map']), f'{B.maps}/fwd_{k}.npy')
        a = np.load(m['map'], mmap_mode='r')
        if a.shape != RAST:
            if not m.get('crop'): problems.append(f'model {k}: shape {a.shape} is not the raster {RAST} and the config gives no crop')
            else:
                y0, y1, x0, x1 = m['crop']
                if (y1 - y0, x1 - x0) != a.shape: problems.append(f'model {k}: crop {m["crop"]} does not match shape {a.shape}')
                json.dump(dict(box=m['crop'], source=os.path.abspath(m['map']), note='crop of the render raster (y0, y1, x0, x1)'),
                          open(f'{B.maps}/fwd_{k}.json', 'w'))
    os.symlink(os.path.abspath(cfg['ct']), f'{B.maps}/ct20.png')
    if os.path.isdir(B.shuf): shutil.rmtree(B.shuf)
    pending = []
    if cfg.get('shuffle_maps'):
        os.makedirs(B.shuf)
        for k, p in cfg['shuffle_maps'].items():
            if p and os.path.exists(p): os.symlink(os.path.abspath(p), f'{B.shuf}/fwd_{k}.npy')
            else: pending.append(k)
    for pk, ms in cfg['pairs'].items():
        for k in ms:
            if k not in cfg['models']: problems.append(f'pair {pk}: unknown model {k}')
    leads = [v['key'] for v in cfg['views'] if v.get('lead')]
    if len(leads) != 1: problems.append(f'exactly one lead view needed, got {leads}')
    files = [v['file'] for v in cfg['views']]
    if len(set(files)) != len(files): problems.append('view file names are not unique')
    for v in cfg['views']:
        if v['pair'] not in cfg['pairs']: problems.append(f'view {v["key"]}: unknown pair {v["pair"]}')
        if v['display'] == 'grey' and v['bg'] != 'page': problems.append(f'view {v["key"]}: grey views use the plain page (v2 rule)')
    if not os.path.isdir(cfg['mesh']['dir']): problems.append(f'mesh dir missing: {cfg["mesh"]["dir"]}')
    if cfg.get('letters', {}).get('label') and not os.path.exists(cfg['letters']['label']): problems.append('papyrologist label missing')
    if not os.path.isdir(cfg['base_package']): problems.append(f'base package missing: {cfg["base_package"]}')
    import make_claim_image_v2 as C
    C.configure(cfg); cu = cfg.get('context', {}); width = len(np.arange(cu.get('u0', -30.0), cu.get('u1', 30.0), 8.64 / 1000.0)) + 2 * C.MARGIN
    worst = []
    for v in cfg['views']:
        hdr, ftr, *_ = C.texts(B.mesh, v['pair'], v['display'], v['rung'] if v['rung'] != 'auto' else 'secondary', v['bg'], 3.999)
        lv = dict(L=-10.0, H=-10.0, P=-10.0, lo_median=-10.0, hi_p999=-10.0)
        for t in hdr(lv) + ftr(lv):
            (w, _), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * 2.2, 2); worst.append((16 + w, v['key'], t[:60]))
    mx = max(worst)
    if mx[0] > width: problems.append(f'header too wide for view {mx[1]}: {mx[0]} px > {width} px ("{mx[2]}...")')
    sums = {k: dict(file=os.path.realpath(m['map']), sha256=sha(m['map'])) for k, m in cfg['models'].items() if os.path.exists(m['map'])}
    json.dump(dict(maps=sums, ct=dict(file=os.path.realpath(cfg['ct']), sha256=sha(cfg['ct'])), header_max_px=mx[0], image_width_px=width,
                   shuffled_maps_pending=pending, problems=problems), open(f'{B.work}/preflight.json', 'w'), indent=1)
    if problems: raise SystemExit('preflight failed:\n  ' + '\n  '.join(problems))
    shutil.copy2(B.cfg_path, B.resolved)
    log(B, 'prep', f'guard OK (finalist {B.F["name"]}, map sha {(grec.get("map") or {}).get("sha256", "product: " + str(len(B.F.get("members", []))) + " members")[:24]}); maps {sorted(cfg["models"])}, header max {mx[0]} '
                   f'of {width} px, shuffled maps pending {pending}; send blockers so far: {len(blockers)}')


def fuse(B):
    """product rule: code/fuse_v7.py makes A (mean of the member maps), TRACE x A, its depth-shuffle map and TRACE x decoy into the spec's
    fused_dir; with the spec's expect_maps every output must be byte-identical to the map the candidate was judged on"""
    F = B.F; t = time.time()
    cmd = [sys.executable, f'{CODE}/fuse_v7.py', F['fused_dir'], '--trace', F['trace_map'], '--members'] + [m['map'] for m in F['members']] + \
          ['--shuf-trace', F['trace_shuf_map'], '--shuf-members'] + [m['shuf_map'] for m in F['members']] + ['--decoy', F['decoy']['map']] + \
          (['--expect'] + [f'{k}={v}' for k, v in F.get('expect_maps', {}).items()] if F.get('expect_maps') else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    shutil.copy2(f'{F["fused_dir"]}/fuse.json', f'{B.work}/fuse.json')
    if r.returncode: print(r.stdout[-2000:], r.stderr[-2000:]); raise SystemExit('fuse failed or a fused map differs from the expected one (fuse.json)')
    log(B, 'prep', f'fused maps rebuilt in {time.time() - t:.0f} s: {r.stdout.strip()}')


def step_rungs(B):
    import controls_v5 as CV
    cfg = json.load(open(B.cfg_path)); auto = [v for v in cfg['views'] if v['display'] == 'threshold' and v['rung'] == 'auto']
    dec = CV.resolve_rungs(B.ctx(), auto) if auto else {}
    for v in cfg['views']:
        if v['key'] in dec:
            v['rung'] = dec[v['key']]['rung']
            if v.get('file_by_rung'): v['file'] = v['file_by_rung'][v['rung']]
    cfg['rung_decisions'] = dec
    json.dump(cfg, open(B.resolved, 'w'), indent=1, ensure_ascii=False)
    log(B, 'rungs', f'{len(dec)} auto rung(s) resolved', decisions={k: d['rung'] for k, d in dec.items()})


def step_views(B):
    cfg = B.rcfg(); t = time.time()
    cmd = [sys.executable, f'{CODE}/build_variants.py', '--maps', B.maps, '--ct', f'{B.maps}/ct20.png', '--mesh', cfg['mesh']['dir'],
           '--out', B.out, '--config', B.resolved, '--layout', 'package', '--jobs', str(B.jobs)]
    r = subprocess.run(cmd, capture_output=True, text=True); print(r.stdout[-4000:])
    if r.returncode: print(r.stderr[-4000:]); raise SystemExit('views failed')
    log(B, 'views', f'{len(cfg["views"])} views in {time.time() - t:.0f} s ({B.jobs} at once)')


def step_controls(B):
    import controls_v5 as CV
    cfg = B.rcfg(); cx = B.ctx(); E = f'{B.out}/evidence'; os.makedirs(E, exist_ok=True); t = time.time()
    cv_ = cfg.get('context_view'); ctx = dict(pair=cv_['pair'], out=f'{B.out}/supplementary/{cv_["file"]}.png') if cv_ else None
    CV.views_controls(cx, f'{E}/controls_{B.tag}_views_same_levels.json', ctx, views_dir=f'{B.work}/ctrl_views' if cfg.get('controls', {}).get('save_views') else '')
    CV.variant_controls(cx, f'{E}/blank_strip_variants_{B.tag}.json')
    log(B, 'controls', f'done in {time.time() - t:.0f} s')


def step_shuffle(B):
    import controls_v5 as CV
    E = f'{B.out}/evidence'; os.makedirs(E, exist_ok=True); t = time.time()
    _, missing = CV.shuffle_variants(B.ctx(), B.views(), f'{E}/depth_shuffle_variants_{B.tag}.json')
    json.dump(dict(pending=missing, note='views whose models lack a depth-shuffled map (GPU run needed; README_v7_build.md)'),
              open(f'{B.work}/shuffle_pending.json', 'w'), indent=1)
    log(B, 'shuffle', f'done in {time.time() - t:.0f} s; pending (no shuffled map): {missing}')


def step_blank(B):
    import controls_v5 as CV
    E = f'{B.out}/evidence'; os.makedirs(E, exist_ok=True); t = time.time()
    CV.blank_views(B.ctx(), B.views(), f'{E}/blank_control_views_{B.tag}.json')
    log(B, 'blank', f'done in {time.time() - t:.0f} s')


def step_boxcount(B):
    import box_count_v5 as BC
    cfg = B.rcfg(); bc = cfg['box_count']; E = f'{B.out}/evidence'; os.makedirs(E, exist_ok=True)
    main = bc.get('main') or cfg['pairs'][B.lead()['pair']]
    boxes = bc.get('boxes') or {B.tag: cfg['box']}
    res, lv = BC.box_count(B.maps, main, bc['check'], boxes, bc.get('lost_vs', ''), f'{E}/box_count_{B.tag}.json', f'{B.work}/marks_{B.tag}.json')
    first = next(iter(boxes))
    log(B, 'boxcount', f'{first}: S {res[first]["S"]}, stray {res[first]["stray"]}, cut {res[first]["letter_cut"]}', levels=lv)


def step_mesh(B):
    import mesh_box_v5 as MB
    cfg = B.rcfg(); r = MB.mesh_box(cfg['mesh']['dir'], B.out, cfg['box'], B.mesh)
    log(B, 'mesh', f'flat {r["flat_cm2"]} cm2, mesh {r["mesh3d_cm2"]} cm2, stretched > 5 %: {r["stretch_off_5pct"]}')


def step_letters(B):
    """version 5's letter-dimension table (the papyrologist's 16 Oct tracing, MEASUREMENT ONLY: never used to train or choose)"""
    cfg = B.rcfg()
    r = subprocess.run([sys.executable, f'{CODE}/letter_table_v5.py', B.out, cfg['letters']['label'], B.mesh], capture_output=True, text=True)
    if r.returncode: print(r.stderr[-3000:]); raise SystemExit('letters failed')
    log(B, 'letters', 'letter_dimensions.json / .md written')


def step_side(B):
    import side_by_side_v5 as SB
    cfg = B.rcfg(); byk = {v['key']: (B.mesh if v.get('lead') else f'supplementary/{v["file"]}') for v in cfg['views']}
    views = [(byk[f[1:]] if f.startswith('@') else f.replace('<MESH>', B.mesh), lab) for f, lab in cfg['side_by_side']]
    log(B, 'side', f'views_side_by_side.jpg {SB.side_by_side(B.out, views)}')


def step_mirror(B):
    cfg = B.rcfg()
    if not cfg.get('mirror_control'): log(B, 'mirror', 'not asked for by the config'); return
    import numpy as np, cv2, claimlib_v2 as L
    png = f'{B.out}/{B.mesh}.png'; side = json.load(open(png[:-4] + '.json')); b = side['claim_box_image_px']; ip = side['image_px']
    m = int(round(3 * L.PPMX))
    img = cv2.cvtColor(cv2.imread(png), cv2.COLOR_BGR2RGB)
    y0, y1 = max(ip['map_top'], b['y0'] - m), min(ip['map_top'] + ip['map_h'], b['y1'] + m)
    x0, x1 = max(ip['map_left'], b['x0'] - m), min(ip['map_left'] + ip['map_w'], b['x1'] + m)
    crop = np.ascontiguousarray(img[y0:y1, x0:x1][:, ::-1])
    lv = side['display']['levels']; kind = side['display'].get('kind', 'threshold')
    lvt = (f'L {lv["L"]:.2f}, p99 {lv["H"]:.2f}' if kind == 'threshold' else f'grey: median {lv["lo_median"]:.2f} .. p99.9 {lv["hi_p999"]:.2f}')
    fn = f'{B.out}/supplementary/ctrl_mirror_{side.get("variant", {}).get("file", "lead")}.jpg'
    t1 = (f'CONTROL: MIRROR (left-right) of the claim box + 3 mm of {os.path.basename(png)} ({" + ".join(side["ink"]["models"])}, '
          f'same pixels, same rule: {lvt}). A reading control, not a claim.' + (' DRY RUN.' if B.dry else ''))
    out = L.frame_image(crop, [t1], [], margin=40)
    cv2.imwrite(fn, cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    log(B, 'mirror', f'{os.path.relpath(fn, B.out)} {out.shape}')


def step_panel(B):
    """version 7: one sheet of the main image's controls (config controls_panel): the main image, its depth shuffle at the same levels,
    the decoy model's image, version 5's main image -> supplementary/<file>.jpg (+ evidence/<file>.json)"""
    cfg = B.rcfg(); cp = cfg.get('controls_panel')
    if not cp: log(B, 'panel', 'no controls panel in this config'); return
    import controls_panel_v7 as CP
    fn = f'{B.out}/supplementary/{cp["file"]}.jpg'
    res = CP.panel(B.ctx(), [tuple(r) for r in cp['rows']], fn, cfg['blank_control'],
                   title=('DRY RUN. ' if B.dry else '') + 'Controls of the main image: claim box u -15..+23 x v -6.84..+2.72 mm, reading orientation')
    os.makedirs(f'{B.out}/evidence', exist_ok=True); shutil.move(fn[:-4] + '.json', f'{B.out}/evidence/{cp["file"]}.json')
    log(B, 'panel', f'{os.path.relpath(fn, B.out)}: ' + json.dumps({k: {a: v[a] for a in v if a.endswith(('half_tone', 'shuffled'))} for k, v in res.items()}))


def step_overlay(B):
    import overlay_v5 as OV
    cfg = B.rcfg(); o = cfg['overlay']
    r = OV.overlays(B.maps, list(o.get('keys') or cfg['pairs'][B.lead()['pair']]), o['models_txt'] + (' (DRY RUN)' if B.dry else ''),
                    cfg['box'], f'{B.work}/form', o.get('crop'), o.get('inset'), B.mesh, o.get('expect_levels'))
    log(B, 'overlay', f'form overlays {list(r["files"])}', levels=r['levels'])


def pct(x, nd=2): return f'{100 * x + 1e-9:.{nd}f} %'


def step_tables(B):
    cfg = B.rcfg(); E = f'{B.out}/evidence'; T = f'{B.work}/tables'; os.makedirs(T, exist_ok=True)
    bv = rd(f'{E}/blank_control_views_{B.tag}.json'); sv = rd(f'{E}/depth_shuffle_variants_{B.tag}.json'); bc = rd(f'{E}/box_count_{B.tag}.json')
    ba = rd(f'{B.out}/mesh/box_area.json'); db = rd(f'{B.out}/mesh/distortion_in_box.json')
    names = {k: m['name'] for k, m in cfg['models'].items()}; lead = B.lead(); rep = list(cfg.get('blank_reported', {}))
    ctl = cfg['blank_control'].get('short', 'blank control')
    rows = ['| view | file | ink | display | ' + ' | '.join(f'{c}, same levels' for c in [ctl] + [r.replace('_', ' ') for r in rep]) +
            ' | box | depth shuffle, box, same levels |', '|---' * (7 + len(rep)) + '|']
    for v in cfg['views']:
        b = bv.get('views', {}).get(v['key'])
        if not b: continue
        f = f"`{B.mesh}.png` (main image)" if v.get('lead') else f"`supplementary/{v['file']}.png`"
        ink = ' + '.join(names[k] for k in cfg['pairs'][v['pair']])
        if v['display'] == 'threshold':
            disp = f"threshold rule, {v['rung']} rung (L = {0.35 if v['rung'] == 'primary' else 0.5} × p99), pale CT"
            cell = lambda s: f"{pct(s['painted'])} painted ({pct(s['half_tone'])} at half tone)"                         # noqa: E731
            box = f"{pct(b['box']['painted'], 1)} painted"; m = 'painted'
        else:
            disp = 'grey rule (median .. p99.9 of the shown area, no threshold), no CT'
            cell = lambda s: f"p99 tone {s['p99_tone']:.2f}; {pct(s['half_tone'])} darker than half tone"               # noqa: E731
            box = f"p99 tone {b['box']['p99_tone']:.2f}; {pct(b['box']['half_tone'], 1)} darker than half tone"; m = 'half_tone'
        s = sv.get(v['key'])
        sh = (f"{pct(s['original'][m], 1)} → {pct(s['shuffled'][m])} {'painted' if m == 'painted' else 'at half tone'}; p99 tone "
              f"{s['original']['p99_tone']:.2f} → {s['shuffled']['p99_tone']:.2f}") if s else 'pending (no shuffled map for every model of this view)'
        rows.append(f"| {v['view']} | {f} | {ink} | {disp} | {cell(b['regions']['control'])} | " + ' | '.join(cell(b['regions'][r]) for r in rep) + f" | {box} | {sh} |")
    open(f'{T}/views_table.md', 'w').write('\n'.join(rows) + '\n')
    if os.path.exists(f'{B.out}/letter_dimensions.md'): shutil.copy2(f'{B.out}/letter_dimensions.md', f'{T}/letters.md')
    lb = bv.get('views', {}).get(lead['key'], {}); first = next(iter(cfg['box_count'].get('boxes') or {B.tag: 0}))
    c3 = bv.get('views', {}).get('c3_grey', {})
    summ = dict(tag=B.tag, lead=lead['key'], lead_models=' + '.join(names[k] for k in cfg['pairs'][lead['pair']]),
                lead_png_sha256=sha(f'{B.out}/{B.mesh}.png') if os.path.exists(f'{B.out}/{B.mesh}.png') else None,
                lead_levels=lb.get('levels'), box_mm=cfg['box'], box_size_mm=ba.get('box_size_mm'), box_flat_cm2=ba.get('flat_cm2'),
                box_mesh_cm2=ba.get('mesh3d_cm2'), stretch_off_5pct=db.get('share_cells_stretch_off_more_than_5pct'),
                blank_control_half_tone=lb.get('regions', {}).get('control', {}).get('half_tone'),
                blank_control_p99_tone=lb.get('regions', {}).get('control', {}).get('p99_tone'),
                reported_half_tone={r: lb.get('regions', {}).get(r, {}).get('half_tone') for r in rep},
                box_half_tone=lb.get('box', {}).get('half_tone'),
                c3_blank_control_half_tone=c3.get('regions', {}).get('control', {}).get('half_tone'),
                c3_box_half_tone=c3.get('box', {}).get('half_tone'),
                shuffle_lead=sv.get(lead['key']), shuffle_c3=sv.get('c3_grey'),
                box_count={k: bc.get(first, {}).get(k) for k in ('S', 'stray', 'unconf', 'letter_cut')},
                rung_decisions={k: d['rung'] for k, d in cfg.get('rung_decisions', {}).items()},
                threshold_a5_on_control={k: b['a5_on_control'] for k, b in bv.get('views', {}).items() if 'a5_on_control' in b})
    json.dump(summ, open(f'{B.work}/summary_{B.tag}.json', 'w'), indent=1, ensure_ascii=False)
    log(B, 'tables', 'tables/*.md and summary written')


# ---------------------------------------------------------------- version 7: the model's records
def step_exclusion(B):
    """code/exclusion_audit_v7.py recompute (the training tiles rebuilt from the labels and checked in 3-D), then the meshes the check
    used are copied into methodology/model_v7/exclusion/meshes/ so the Docker image (mode 'exclusion') can re-verify the tile list"""
    import exclusion_audit_v7 as XA
    os.makedirs(B.MD, exist_ok=True); F = B.F; t = time.time()
    rec = XA.recompute(F['train_config'], B.MD, F['lane'], P3)
    zj = json.load(open(f'{EXCL}/zone.json'))
    rec.update(finalist=F['name'], zone=dict(json=zj, zone_npz_sha256=sha(f'{EXCL}/zone.npz'), check_py_sha256=sha(f'{EXCL}/check.py')),
               rule=('a training tile is allowed iff its 3-D points are >= 2.0 mm (zone buffer) + 0.2 mm (render slab) + pad from every point of '
                     'every surface shown in any lead-A claim image (versions 1-7: mesh V5Rc u -30..+30 x v -20..+15 mm and the v1 mesh window, '
                     '+2 mm in-plane margin), on any winding'))
    if F.get('fusion') == 'product' and os.path.exists(f'{B.MD}/training_tiles.npz'):   # seeds 1, 2 trained from the cloud packs: same tiles?
        import numpy as np
        T_ = np.load(f'{B.MD}/training_tiles.npz'); tl = T_['tiles']; chk = {}
        for ri, r in enumerate(list(T_['regions'])):
            pk = f'{P3}/m306/self/packs/306_{r}.npz'
            if not os.path.exists(pk): chk[r] = 'pack missing'; continue
            org = np.load(pk)['org_main'].astype(np.int64); bb = np.load(f'{P3}/m306/self/labels/{r}_labels.npz')['bbox']
            pt = np.stack([org[:, 0] - 64 + int(bb[0]), org[:, 1] - 64 + int(bb[2])], 1)
            mine = tl[tl[:, 0] == ri][:, 1:].astype(np.int64)
            same = len(pt) == len(mine) and np.array_equal(pt[np.lexsort(pt.T[::-1])], mine[np.lexsort(mine.T[::-1])])
            chk[r] = dict(pack=os.path.basename(pk), pack_sha256=sha(pk), pack_tiles=int(len(pt)), identical_to_audited_tiles=bool(same))
            rec['all_allowed'] &= bool(same)
        rec['cloud_packs'] = dict(note='seeds 1 and 2 (and the decoy) were trained on rented GPUs from packs/306_<r>.npz (m306/self/code/make_packs.py, '
                                       'built with train_self.src_306); their main-variant tile lists must equal the audited list', regions=chk)
    json.dump(rec, open(f'{B.MD}/exclusion_audit.json', 'w'), indent=1, ensure_ascii=False)
    if not rec['all_allowed']: raise SystemExit(f'EXCLUSION AUDIT FAILED: see {B.MD}/exclusion_audit.json')
    if os.path.exists(f'{B.MD}/tile_regions.json'):
        TR = json.load(open(f'{B.MD}/tile_regions.json')); MR = f'{B.MD}/exclusion/meshes'
        for r, w in TR['where'].items():
            d = f'{MR}/{r}'; os.makedirs(d, exist_ok=True)
            for f in ('x.tif', 'y.tif', 'z.tif', 'meta.json'): shutil.copy2(f'{w["mesh"]}/{f}', f'{d}/{f}')
    log(B, 'exclusion', f'all training tiles allowed ({rec.get("tiles_total", "no 0306B tiles")}) in {time.time() - t:.0f} s')


def known_rows(B):
    """held-out known-text numbers from the M306 evaluator (cands/<name>/known.json; TTA 1, +-3 sigma stitch)"""
    rows = {}
    for nm, p in ((B.F['eval_name'], f'{EVAL}/cands/{B.F["eval_name"]}/known.json'), ('base v8-in', f'{EVAL}/cands/base/known.json'),
                  ('organisers\' LOO-w062 fine-tune (1447 w058 + w060 only)', f'{P3}/m306/self/ref_loo/known.json')):
        k = rd(p)
        if k: rows[nm] = dict(source=p, trained_on=k.get('trained_on'), tta=k.get('tta', 1), scores=k.get('scores', {}))
    return rows


def step_method(B):
    F = B.F; os.makedirs(B.MD, exist_ok=True); t = time.time(); g = rd(f'{B.work}/guard.json')
    run = F.get('run_dir', '')
    cp = lambda s, d: (os.path.exists(s) and (shutil.copy2(s, f'{B.MD}/{d}') or True))                                   # noqa: E731
    cp(F['train_config'], 'training_config.json'); cp(f'{run}/config_train.json', 'run_config.json'); cp(f'{run}/train_log.jsonl', 'train_log.jsonl')
    cp(f'{run}/tiles.json', 'training_sources.json')
    for c in F['ckpts'] + F.get('stages', []):
        d = os.path.dirname(c['path']); tag = f'{os.path.basename(os.path.dirname(d))}_{os.path.basename(d)}'     # <run>_<final | step_s>
        cp(f'{d}/meta.json', f'checkpoint_{tag}_meta.json'); cp(f'{d}/config.json', f'checkpoint_{tag}_config.json')
    if F.get('fusion') == 'product':                                                 # every member's (and the decoy's) run records
        for mm in F['members'] + [F['decoy']]:
            n = mm['name']; R_ = mm['run_dir']; os.makedirs(f'{B.MD}/members/{n}', exist_ok=True)
            for f in ('config_train.json', 'train_log.jsonl', 'done.json', 'tiles.json', 'cloud_train.log'):
                if os.path.exists(f'{R_}/{f}'): shutil.copy2(f'{R_}/{f}', f'{B.MD}/members/{n}/{f}')
            if os.path.exists(mm['train_config']): shutil.copy2(mm['train_config'], f'{B.MD}/members/{n}/training_config.json')
            if os.path.exists(f'{R_}/final/meta.json'): shutil.copy2(f'{R_}/final/meta.json', f'{B.MD}/members/{n}/final_meta.json')
        for f, d in ((f'{B.work}/fuse.json', 'fusion_record.json'),): shutil.copy2(f, f'{B.MD}/{d}')
    if F['lane'] == 'self':
        S = f'{P3}/m306/self'
        cp(f'{S}/labels/stats.json', 'pseudo_labels_stats.json'); cp(f'{S}/pool/regions_pool.json', 'pool_regions.json')
        for p in glob.glob(f'{S}/teach/*_teach.json') + glob.glob(f'{S}/pool/*_masks.json'): shutil.copy2(p, f'{B.MD}/{os.path.basename(p)}')
    ck = dict(base=dict(name='v8-in base (HF YoussefMoNader/ink-8um-v8in)', sha256='3b94548d7f9ba9f99bc4b0ad8cafcec6d42585c06304e1bd11e47b5d0f88c4cd'),
              finalist=g.get('ckpts'), stages=g.get('stages'), map=g.get('map'), shuffle_map=g.get('shuf_map'))
    json.dump(ck, open(f'{B.MD}/checkpoints.json', 'w'), indent=1)
    DS = rd(f'{HERE}/data_sources.json'); json.dump({k: DS[k] for k in F.get('data_sources', []) + ['v8in_base'] if k in DS},
                                                   open(f'{B.MD}/data_sources.json', 'w'), indent=1, ensure_ascii=False)
    X = f'{B.MD}/exclusion'; os.makedirs(X, exist_ok=True)
    for f in ('check.py', 'build_zone.py', 'zone.json', 'zone.npz', 'README.md', 'renders.json'): shutil.copy2(f'{EXCL}/{f}', f'{X}/{f}')
    E = f'{B.out}/evidence/heldout_v7'; os.makedirs(E, exist_ok=True)
    json.dump(known_rows(B), open(f'{E}/known_text.json', 'w'), indent=1, ensure_ascii=False)
    for k_, d_ in (('known_product', 'known_text_product_rule.json'), ('ens_known', 'known_text_ensemble.json')):
        if F.get(k_) and os.path.exists(F[k_]): shutil.copy2(F[k_], f'{E}/{d_}')
    lb = rd(f'{EVAL}/LEADERBOARD.json'); keep = {k: v for k, v in lb.items() if k in (F['eval_name'], 'base', F.get('decoy_eval_name'), F.get('nopl_eval_name'))}
    json.dump(keep, open(f'{E}/evaluator_rows.json', 'w'), indent=1, ensure_ascii=False)
    log(B, 'method', f'methodology/model_v7 and evidence/heldout_v7 written in {time.time() - t:.0f} s')


def training_code(F):
    """the code that made the finalist (copied byte-identical into code/training/, with its original path and sha256)"""
    L = [f'{P3}/aclick/p306/exclusion/check.py', f'{P3}/aclick/bench/infer.py', f'{P3}/m306/eval/code/render_leadA.py',
         f'{P3}/m306/train/v8306lib.py', f'{P3}/m306/train/train_v8306.py']
    if F['lane'] == 'self':
        L += [f'{P3}/m306/self/code/{f}' for f in ('pool.py', 'masks.py', 'teachers.py', 'labels.py', 'train_self.py')]
        L += [f'{W}/x_req/20261020/V3/run_maps.py']
    if F.get('fusion') == 'product':                                                 # seeds 1, 2 (and the decoy) ran on rented GPUs from packs
        S = f'{P3}/m306/self'
        L += [f'{S}/code/train_cloud.py', f'{S}/code/make_packs.py', f'{S}/cloud_box.sh', f'{P3}/m306/coord/ens_mean.py', f'{P3}/m306/coord/control_prod.py']
        L += [mm['train_config'] for mm in F['members'] + [F['decoy']]]
        L += sorted({mm['trainer_path'] for mm in F['members'] + [F['decoy']] if mm.get('trainer_path')})
    L += F.get('extra_training_code', [])                                         # version 7: the hard-negative trainer and its configs
    return list(dict.fromkeys(p for p in L if os.path.exists(p)))


def step_assemble(B):
    cfg = B.rcfg(); base = cfg['base_package']; made = set()
    for p in glob.glob(f'{B.out}/**/*', recursive=True):
        if os.path.isfile(p): made.add(os.path.relpath(p, B.out))
    skip_prefix = ('code/', 'docker/', 'README.md', 'CHANGES.md', 'SHA256SUMS')
    copied = 0
    for p in glob.glob(f'{base}/**/*', recursive=True):
        rel = os.path.relpath(p, base)
        if not os.path.isfile(p) or rel in made or rel.startswith(skip_prefix) or '__pycache__' in rel: continue
        if rel == f'{B.mesh}.png' or rel == f'{B.mesh}.json': continue                       # version 5's main image comes back as view 5-1
        d = f'{B.out}/{rel}'; os.makedirs(os.path.dirname(d), exist_ok=True); shutil.copy2(p, d); copied += 1
    shutil.rmtree(f'{B.out}/code', ignore_errors=True); shutil.copytree(CODE, f'{B.out}/code', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.rmtree(f'{B.out}/docker', ignore_errors=True); shutil.copytree(f'{HERE}/docker', f'{B.out}/docker', ignore=shutil.ignore_patterns('__pycache__', 'ctx*'))
    if os.path.exists(f'{base}/code/config_v5.json'): shutil.copy2(f'{base}/code/config_v5.json', f'{B.out}/code/config_v5.json')   # v5's, kept
    rc = json.load(open(B.resolved))
    json.dump(rc, open(f'{B.out}/code/config_v7.json', 'w'), indent=1, ensure_ascii=False)
    TD = f'{B.out}/code/training'; os.makedirs(TD, exist_ok=True); src = []
    for p in training_code(B.F):
        rel = os.path.relpath(p, W); d = f'{TD}/{rel}'; os.makedirs(os.path.dirname(d), exist_ok=True); shutil.copy2(p, d)
        src.append(f'{sha(p)}  {rel}')
    open(f'{TD}/SOURCES.sha256', 'w').write('\n'.join(src) + '\n')
    open(f'{TD}/README.md', 'w').write(
        '# Training code of the version-7 ink model (copies, byte-identical)\n\n'
        'Each file is the exact file that ran, at its path under our working tree (`W` = the repository root; the lanes live under\n'
        '`W/.claude/worktrees/leadA-push3/x_push3/`). The scripts address data by those absolute paths; the Docker image recreates\n'
        'the same layout (`docker/README.md`, mode `train`). `SOURCES.sha256` lists each file\'s sha256. `methodology/model_v7/MODEL.md`\n'
        'gives the order in which they ran.\n')
    n = step_sums(B, quiet=True)
    log(B, 'assemble', f'{copied} files from {base}; {len(src)} training code files; {n} files in SHA256SUMS')


# ---------------------------------------------------------------- text
def fill(t, vals, flags):
    for fl, on in flags.items():                                                     # <!--if:FLAG--> ... <!--endif:FLAG-->
        t = re.sub(rf'<!--if:{fl}-->(.*?)<!--endif:{fl}-->', (lambda m: m.group(1)) if on else '', t, flags=re.S)
        t = re.sub(rf'<!--ifnot:{fl}-->(.*?)<!--endif:{fl}-->', '' if on else (lambda m: m.group(1)), t, flags=re.S)
    for k, v in vals.items():
        if v is not None: t = t.replace('{{' + k + '}}', str(v))
    return t


def f3(x): return f'{x:.3f}' if isinstance(x, (int, float)) and x == x else '—'


def text_values(B):
    F = B.F; cfg = B.rcfg(); S = rd(f'{B.work}/summary_{B.tag}.json'); g = rd(f'{B.work}/guard.json'); X = rd(f'{B.MD}/exclusion_audit.json')
    K = rd(f'{B.out}/evidence/heldout_v7/known_text.json'); LS = rd(f'{P3}/m306/self/labels/stats.json'); tc = rd(F['train_config'])
    pc = lambda x, nd=2: f'{100 * x + 1e-9:.{nd}f}' if isinstance(x, (int, float)) else None                       # noqa: E731
    sh = S.get('shuffle_lead') or {}; s3 = S.get('shuffle_c3') or {}
    fin = K.get(F['eval_name'], {}).get('scores', {}); base = K.get('base v8-in', {}).get('scores', {})
    loo = next((v['scores'] for k, v in K.items() if k.startswith('organisers')), {})

    def krow(name, sc, sets=('f9h', 'n841h')):
        cells = []
        for s in sets:
            v = sc.get(s)
            cells.append(f'{f3(v["auc"])} / {f3(v["ap"])} / {f3(v["fi_skel80"])} / {f3(v["faint_recall"])} / {f3(v["fwhm_mm"])}' if v else 'not scored')
        return f'| {name} | ' + ' | '.join(cells) + ' |'
    ktab = '\n'.join(['| model | PHerc0009B 20250919125754, organiser human labels v2 (8.64 µm, 116 keV): AUC / AP / false ink at 80 % recall / '
                      'faint-stroke recall / stroke width (mm) | PHerc0841 20260220213127, organiser human labels v2 (9.37 µm, 113 keV): same |',
                      '|---|---|---|', krow(f'{F["display"]} (this image\'s model)', fin), krow('v8-in base (the organisers\' public model)', base)]
                     + ([krow('the organisers\' PHerc1447 fine-tune (LOO-w062: w058 + w060 only)', loo)] if loo else []))
    w62 = fin.get('w062a') or {}
    regs = X.get('regions', {})
    pool_rows = '\n'.join(f'| {r} | {v["tiles"]} | {v["min_distance_mm_tile_grid"]} mm | {v["median_distance_mm_tile_grid"]} mm |' for r, v in regs.items())
    lab_tot = LS and {k: round(sum(LS[r]['main'][k] for r in LS if not r.startswith('_')), 1) for k in ('ink_mm2', 'bg_mm2', 'ink_blobs')}
    pool_cm2 = round(sum(LS[r]['valid_cm2'] for r in LS if not r.startswith('_')), 1) if LS else None
    Tr = tc.get('training', {})
    ck = g.get('ckpts') or [{}]
    vals = dict(DATE=time.strftime('%-d %b %Y'), TAG=B.tag, LEAD_MODELS=S.get('lead_models'), FIN=F['display'], FIN_RECIPE=F['recipe_short'],
                FIN_SHA=ck[0].get('sha256'), FIN_SHA12=(ck[0].get('sha256') or '')[:12], MAP_SHA=(g.get('map') or {}).get('sha256'),
                MAP_TTA=(g.get('map') or {}).get('meta', {}).get('tta'),
                BLANK_HALF=pc(S.get('blank_control_half_tone')), BLANK_HALF_C3=pc(S.get('c3_blank_control_half_tone')),
                BOX_HALF=pc(S.get('box_half_tone'), 1), BOX_HALF_C3=pc(S.get('c3_box_half_tone'), 1),
                BOX_FLAT=f"{S['box_flat_cm2']:.3f}" if S.get('box_flat_cm2') else None, BOX_MESH=f"{S['box_mesh_cm2']:.3f}" if S.get('box_mesh_cm2') else None,
                SHUF_BEFORE=pc(sh.get('original', {}).get('half_tone'), 1), SHUF_AFTER=pc(sh.get('shuffled', {}).get('half_tone'), 2),
                SHUF_C3_BEFORE=pc(s3.get('original', {}).get('half_tone'), 1), SHUF_C3_AFTER=pc(s3.get('shuffled', {}).get('half_tone'), 2),
                BOXCOUNT_S=(S.get('box_count') or {}).get('S'), BOXCOUNT_STRAY=(S.get('box_count') or {}).get('stray'),
                THRESH_RUNG=(S.get('rung_decisions') or {}).get('c6_threshold'),
                KNOWN_TABLE=ktab, F9H_AUC=f3((fin.get('f9h') or {}).get('auc')), F9H_AUC_BASE=f3((base.get('f9h') or {}).get('auc')),
                F9H_FI=f3((fin.get('f9h') or {}).get('fi_skel80')), F9H_FI_BASE=f3((base.get('f9h') or {}).get('fi_skel80')),
                N841_AUC=f3((fin.get('n841h') or {}).get('auc')), N841_AUC_BASE=f3((base.get('n841h') or {}).get('auc')),
                N841_FI=f3((fin.get('n841h') or {}).get('fi_skel80')), N841_FI_BASE=f3((base.get('n841h') or {}).get('fi_skel80')),
                W062_R_LOO=w62.get('r_loo'), W062_R_BASE=w62.get('r_base'),
                TILES_TOTAL=(f"{X['tiles_total']:,}" if X.get('tiles_total') else None), POOL_ROWS=pool_rows, POOL_CM2=pool_cm2,
                MIN_DIST=min((v['min_distance_mm_tile_grid'] for v in regs.values()), default=None),
                PSEUDO_INK_MM2=(lab_tot or {}).get('ink_mm2'), PSEUDO_BG_MM2=(lab_tot or {}).get('bg_mm2'), PSEUDO_BLOBS=(lab_tot or {}).get('ink_blobs'),
                STEPS=Tr.get('steps'), BATCH=(int(Tr.get('micro', 4)) * int(Tr.get('accum', 4))) if Tr else None, LR=Tr.get('lr'),
                SEED=Tr.get('seed'), W306=Tr.get('w306'), MODE=F['mode'],
                N_VIEWS=len(cfg['views']), ZONE_SHA12=X.get('zone', {}).get('zone_npz_sha256', '')[:12],
                SELECTION=F.get('selection_note') or ('DRY RUN: no selection made.' if B.dry else None),
                DECOY=(F.get('decoy_note') or ('DRY RUN: not run' if B.dry else '')).rstrip('.') or None)
    if F.get('fusion') == 'product':                                                 # version 7 product rule: the claim fusion's own numbers
        KP = rd(F.get('known_product', '')).get('scores', {}); EN = rd(F.get('ens_known', '')).get(F.get('ens_known_key', ''), {})
        prow = KP.get("TRACE v3B x A (product; the main image's rule)", {}); arow = KP.get('A = mean of the three member seeds', {}) or EN
        trow = KP.get('TRACE v3B', {})
        order = [("TRACE v3B x A (product; the main image's rule)", f'**TRACE v3B x {F["display"]} (product: the main image\'s rule)**'),
                 ('TRACE v3B', 'TRACE v3B alone'), ('A = mean of the three member seeds', f'{F["display"]} alone (the mean of the three seeds)'),
                 ('TRACE v3B x model A before the hard-negative retraining (product)', 'TRACE v3B x model A before the hard-negative retraining (product)'),
                 ('TRACE v3B x base v8-in (product)', 'TRACE v3B x v8-in base (product)'), ('base v8-in', 'v8-in base (the organisers\' public model)')]
        hdr = vals['KNOWN_TABLE'].split('\n')[:2]
        vals['KNOWN_TABLE'] = '\n'.join(hdr + [krow(lab, KP[k]) for k, lab in order if k in KP] + ([krow('the organisers\' PHerc1447 fine-tune (LOO-w062: w058 + w060 only)', loo)] if loo else []))
        vals.update(F9H_AUC=f3((arow.get('f9h') or {}).get('auc')), F9H_AP=f3((arow.get('f9h') or {}).get('ap')), F9H_FI=f3((arow.get('f9h') or {}).get('fi_skel80')),
                    N841_AUC=f3((arow.get('n841h') or {}).get('auc')), N841_AP=f3((arow.get('n841h') or {}).get('ap')), N841_FI=f3((arow.get('n841h') or {}).get('fi_skel80')),
                    F9H_AP_BASE=f3((base.get('f9h') or {}).get('ap')), N841_AP_BASE=f3((base.get('n841h') or {}).get('ap')),
                    P_F9H=' / '.join(f3((prow.get('f9h') or {}).get(a)) for a in ('auc', 'ap', 'fi_skel80')),
                    P_N841=' / '.join(f3((prow.get('n841h') or {}).get(a)) for a in ('auc', 'ap', 'fi_skel80')),
                    T_F9H=' / '.join(f3((trow.get('f9h') or {}).get(a)) for a in ('auc', 'ap', 'fi_skel80')),
                    T_N841=' / '.join(f3((trow.get('n841h') or {}).get(a)) for a in ('auc', 'ap', 'fi_skel80')),
                    RULE=F.get('rule'), N_MEMBERS={2: 'two', 3: 'three', 4: 'four', 5: 'five'}.get(len(F['members']), len(F['members'])), MEMBER_TTA=F.get('expect', {}).get('tta'),
                    MEMBERS_TABLE='\n'.join(['| member | seed | sha256 of the weights | trained with | training config |', '|---|---|---|---|---|'] +
                                            [f'| {m["name"]} | {m["seed"]} | `{m["sha256"]}` | {m["trainer"]} | `members/{m["name"]}/training_config.json` |'
                                             for m in F['members']]),
                    DECOY_SHA=F['decoy']['sha256'], FIN_SHA=', '.join(m['sha256'][:12] + '...' for m in F['members']),
                    MAP_SHA=rd(f'{B.work}/fuse.json').get('outputs', {}).get('prod', {}).get('sha256'), MAP_TTA=F.get('expect', {}).get('tta'),
                    PANEL='supplementary/' + (cfg.get('controls_panel') or {}).get('file', '') + '.jpg')
    T = f'{B.work}/tables'
    if os.path.exists(f'{T}/views_table.md'): vals['VIEWS_TABLE'] = open(f'{T}/views_table.md').read().strip()
    if os.path.exists(f'{T}/letters.md'): vals['LETTERS_TABLE'] = re.sub(r'^#.*\n+', '', open(f'{T}/letters.md').read(), flags=re.M).strip()
    return vals


def step_text(B):
    F = B.F; vals = text_values(B); rev = F.get('status') == 'review' and not B.cfg.get('approved')
    flags = dict(self=F['lane'] == 'self', train=F['lane'] == 'train', fused=F['mode'] == 'fused', alone=F['mode'] == 'alone', dry=B.dry,
                 product=F['mode'] == 'product', review=rev, single=F['mode'] != 'product')
    T = f'{B.work}/text'; os.makedirs(T, exist_ok=True); left = {}
    base_readme = open(f'{B.rcfg()["base_package"]}/README.md').read()
    demoted = re.sub(r'^(#+) ', lambda m: '#' + m.group(1) + ' ', base_readme, flags=re.M)        # version 5's README, one heading level down
    for tpl, dest in (('README_v7_TEMPLATE.md', f'{B.out}/README.md'), ('CHANGES_v7_TEMPLATE.md', f'{B.out}/CHANGES.md'),
                      ('MODEL_v7_TEMPLATE.md', f'{B.MD}/MODEL.md'), ('FORM_v7_TEMPLATE.md', f'{T}/FORM_owner_only.md')):
        t = fill(open(f'{HERE}/{tpl}').read(), vals, flags)
        if tpl.startswith('README'): t = t.replace('{{V5_README}}', demoted)
        t = re.sub(r'<!--NOTE.*?-->\n?', '', t, flags=re.S)                          # template notes never reach the package
        os.makedirs(os.path.dirname(dest), exist_ok=True); open(dest, 'w').write(t)
        left[tpl] = dict(numbers=sorted(set(re.findall(r'\{\{[A-Z0-9_]+\}\}', t))), owner_text=len(re.findall(r'\[\[', t)))
    g = rd(f'{B.work}/guard.json'); bl = [b for b in g.get('send_blockers', []) if (rev or not b.startswith('OWNER REVIEW'))
                                       and (F.get('selection_note_is_draft') or not b.startswith('selection sentence'))
                                       and (F.get('decoy_note_is_draft') or not b.startswith('decoy sentence'))]
    if any(v['numbers'] for v in left.values()): bl.append(f'unfilled numbers: {[(k, v["numbers"]) for k, v in left.items() if v["numbers"]]}')
    if not os.path.isdir(f'{B.out}/evidence/docker_test_v7'): bl.append('Docker test of the v7 image not in the package yet (README_v7_build.md, Docker)')
    lb = rd(f'{EVAL}/LEADERBOARD.json')
    if F.get('decoy_eval_name') and F['decoy_eval_name'] not in lb and not F.get('decoy_note'): bl.append(f'decoy control ({F["decoy_eval_name"]}, labels moved) not evaluated yet')
    json.dump(dict(left=left, send_blockers=bl), open(f'{T}/text_status.json', 'w'), indent=1)
    step_sums(B, quiet=True)
    log(B, 'text', f'README.md, CHANGES.md, MODEL.md, FORM_owner_only.md written; send blockers: {len(bl)}')
    for b in bl: print('   SEND BLOCKER:', b, flush=True)


def step_sums(B, quiet=False):
    files = sorted(p for p in glob.glob(f'{B.out}/**/*', recursive=True)
                   if os.path.isfile(p) and not p.endswith('SHA256SUMS') and '/__pycache__/' not in p and not p.endswith('.pyc')
                   and not os.path.basename(p).startswith(('build_log', 'summary_')))
    open(f'{B.out}/SHA256SUMS', 'w').write(''.join(f'{sha(p)}  {os.path.relpath(p, B.out)}\n' for p in files))
    if not quiet: log(B, 'sums', f'{len(files)} files in SHA256SUMS')
    return len(files)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('config'); ap.add_argument('--out', default='')
    ap.add_argument('--map', default='', help="the finalist's TTA-4 lead-A map (render_leadA.py output; its .json beside it)")
    ap.add_argument('--shuf-map', default='', help="the finalist's map of the depth-shuffled render (claim box or context)")
    ap.add_argument('--steps', default='all'); ap.add_argument('--skip', default='')
    ap.add_argument('--jobs', type=int, default=1, help='views rendered at once (each ~2 GB RAM); 1 while gaming')
    ap.add_argument('--preflight', action='store_true', help='only the prep step (guard + preflight; no map statistic, no image)')
    ap.add_argument('--approved', action='store_true', help='the owner approved an OWNER REVIEW build: run --steps text,sums to drop the review banner')
    a = ap.parse_args()
    cfg = json.load(open(a.config)); out = a.out or f'{HERE}/build/{cfg["tag"]}'
    req = list(DEFAULT) if a.steps == 'all' else a.steps.split(',')
    steps = ['prep'] if a.preflight else [s for s in req if s and s not in a.skip.split(',')]
    for s in steps:
        if s not in STEPS: raise SystemExit(f'unknown step {s}; steps: {STEPS}')
    cfg_path = a.config
    if a.approved: cfg['approved'] = True
    if a.map or a.shuf_map or a.approved:                                            # the command line names the maps: written into the input config
        fk = cfg['finalist']['model_key']
        if a.map: cfg['models'][fk]['map'] = os.path.abspath(a.map); cfg['finalist']['map'] = os.path.abspath(a.map)
        if a.shuf_map: cfg.setdefault('shuffle_maps', {})[fk] = os.path.abspath(a.shuf_map); cfg['finalist']['shuf_map'] = os.path.abspath(a.shuf_map)
        os.makedirs(f'{out}.work', exist_ok=True); cfg_path = f'{out}.work/config_input.json'; json.dump(cfg, open(cfg_path, 'w'), indent=1, ensure_ascii=False)
    B = Build(cfg_path, out, a.jobs); t0 = time.time()
    log(B, 'start', f'config {os.path.abspath(a.config)} (sha256 {sha(a.config)[:12]}), steps {steps}, jobs {a.jobs}, OMP_NUM_THREADS {os.environ["OMP_NUM_THREADS"]}'
                    + (' -- DRY RUN' if B.dry else ''))
    for s in STEPS:
        if s in steps:
            t = time.time(); globals()[f'step_{s}'](B); log(B, s, f'step done in {time.time() - t:.0f} s')
    log(B, 'end', f'all steps done in {time.time() - t0:.0f} s' + (' -- DRY RUN: never send' if B.dry else ''))


if __name__ == '__main__':
    main()
