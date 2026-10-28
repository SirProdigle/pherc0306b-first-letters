"""fleet.py : rent, run and tear down Vast boxes for lane_box.sh, ranking offers by estimated $ per lane.

  fleet.py offers [--top 15] [--disk 200]          rank current offers by est. $/lane (dry run, read-only)
  fleet.py plan [--budget 150] [--boxes 6]         dry run: lanes.json queue -> offers, est. hours and $ per lane, total
  fleet.py launch LANE_ID [--offer ID]             rent + push + start one lane   (refuses unless x_vast_plan/GO exists)
  fleet.py poll [--once] [--interval 300]          pull DONE boxes, destroy DONE / FAILED / stale / over-time / over-budget
  fleet.py status | spend | destroy LABEL

Cost model ($/lane = GPU $/h x hours + storage + download + upload). Hours per stage are priors measured on our local RTX 4090
lanes (2026-10-11) and scaled: GPU-bound stages (fit, ink) by GPU_SPEED, CPU-bound stages (data, triage) by cores, renders
(network-bound) by the offer's measured download speed. Replace GPU_SPEED and STAGE_H with the first boxes' timing.tsv.
Safety: launch needs x_vast_plan/GO ({"budget": USD, "max_boxes": N}) written by the coordinator; nothing else spends money.
The Vast key lives only in ~/.config/vastai/vast_api_key (vastai set api-key); this script never reads or prints it."""
import argparse, json, math, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
W = Path('/home/liam/projects/vesuvius-work'); X = W / 'x_recipe_erwin'
VAST = os.environ.get('VAST_BIN', str(W / 'envs/vast/bin/vastai')); KEY = str(Path.home() / '.ssh/vast_ed25519')
STATE = Path(os.environ.get('FLEET_STATE', HERE / 'state')); STATE.mkdir(exist_ok=True); BOXES = STATE / 'boxes.state'
RESULTS = Path(os.environ.get('FLEET_RESULTS', HERE / 'results'))
IMAGE = 'nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04'
SSH_FAIL = {255}                     # ssh's own exit code (refused / reset / auth); anything else came from the remote command
RSYNC_FAIL = {255, 12, 10, 30, 35}   # rsync: remote shell died, protocol stream broken, socket I/O, timeouts
HOLD_MAX_MIN = float(os.environ.get('FLEET_HOLD_MAX_MIN', 90))   # a HOLD box nobody resumes is pulled and destroyed after this

# relative throughput for the fit (torch, fp32/bf16 mix) and the ink models (bf16), RTX 3090 = 1.0 (priors; benchmark replaces)
GPU_SPEED = {'RTX 3090': 1.0, 'RTX 3090 Ti': 1.1, 'RTX A5000': 0.9, 'RTX A6000': 1.1, 'A40': 1.0, 'RTX 4080': 1.4, 'RTX 4080S': 1.45,
             'RTX 4090': 1.7, 'RTX 4090D': 1.5, 'RTX 5080': 1.6, 'RTX 5090': 2.2, 'RTX 5090D': 2.0, 'L40S': 1.9, 'RTX 6000Ada': 1.8,
             'RTX 5880Ada': 1.6, 'RTX PRO 4500': 1.5, 'RTX PRO 5000': 1.9, 'RTX PRO 6000 WS': 2.3, 'RTX PRO 6000 Max-Q': 2.1,
             'A10': 0.65, 'RTX A4500': 0.75, 'RTX 4500Ada': 1.1, 'RTX 4000Ada': 0.7, 'A100 PCIE': 1.7, 'A100 SXM4': 1.8, 'H100 PCIE': 2.4, 'H100 SXM': 2.6, 'L4': 0.6}
# stage hours on the reference box (RTX 4090 = 1.7, 16 cores, ~800 Mbps); see fit_plateau.md for the fit numbers
STAGE_H = dict(setup=0.30, data=0.60, fit_full=1.50, post_per_winding=1.0 / 60, ink_per_winding=95 / 3600, triage=0.75, pack=0.10)   # triage: tri_rank after the overlapped window_score (dry run: ~170 s/winding at 4 threads)
DL_GB = dict(setup=8, data=30, per_winding=0.8)       # download per lane: env + models, band dataset, render chunks
REF = dict(gpu=1.7, cores=16, mbps=800)


def vast(*args, raw=True):
    cmd = [VAST, *args] + (['--raw'] if raw else [])
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if out.returncode != 0:
        raise RuntimeError(f'vastai {" ".join(args[:2])} failed: {out.stderr.strip()[:300]}')
    return json.loads(out.stdout) if raw else out.stdout


def fit_host_factor(host_threads):
    """The spiral fit is NOT GPU-bound.  Measured on the 17 cloud lanes of 2026-10-12 (x_refine/R3_ops_coverage, fitspeed):
    hosts with <= 32 threads ran 30k steps at a median 3.5 it/s (2.1-5.0), many-core servers (64-256 threads) at 2.1 it/s
    (1.7-3.2), and the RTX 5090 boxes were no faster than the 4090s.  Returns fit hours relative to STAGE_H['fit_full']
    (1.5 h = 5.2 it/s on the idle local 4090)."""
    t = host_threads or 32
    return 5.2 / (3.5 if t <= 32 else 2.1)


def lane_hours(lane, gpu, cores, mbps, maxw=40, short_n=8000, host_threads=None):
    g = REF['gpu'] / GPU_SPEED.get(gpu, 1.0); c = (REF['cores'] / max(cores, 4)) ** 0.7
    fit = STAGE_H['fit_full'] * fit_host_factor(host_threads) * (1 + (2 * short_n / 30000 - short_n / 30000 if lane.get('sense') == 'AB' else 0))
    data = STAGE_H['data'] * c + DL_GB['data'] * 8000 / max(mbps, 100) / 3600
    post = maxw * STAGE_H['post_per_winding'] * min(1.0, 2 * REF['cores'] / max(cores, 8)) * max(1.0, REF['mbps'] / max(mbps, 100)) ** 0.5
    ink = maxw * STAGE_H['ink_per_winding'] * g
    return dict(setup=STAGE_H['setup'], data=data, fit=fit, post_ink=max(post, ink) + 0.1, triage=STAGE_H['triage'] * c,
                pack=STAGE_H['pack'])


def lane_cost(o, lane, disk=200, maxw=40):
    gpu = o['gpu_name']; cores = o.get('cpu_cores_effective') or 8; mbps = o.get('inet_down') or 500
    h = lane_hours(lane, gpu, cores, mbps, maxw, host_threads=o.get('cpu_cores')); H = sum(h.values())
    gpu_usd = (o.get('dph_base') or o['dph_total']) * H
    sto_usd = disk * (o.get('storage_cost') or 0.2) / 730 * H
    dl = DL_GB['setup'] + DL_GB['data'] + maxw * DL_GB['per_winding']
    net_usd = dl * (o.get('inet_down_cost') or 0) + 0.3 * (o.get('inet_up_cost') or 0)
    return dict(hours=round(H, 2), usd=round(gpu_usd + sto_usd + net_usd, 3), gpu_usd=round(gpu_usd, 3), storage_usd=round(sto_usd, 3),
                net_usd=round(net_usd, 3), dl_gb=dl, stages={k: round(v, 2) for k, v in h.items()})


def search(disk=200):
    q = (f'rentable=true verified=true num_gpus=1 gpu_ram>=23 reliability>=0.98 inet_down>=500 inet_down_cost<=0.005 '
         f'cpu_cores_effective>=16 cpu_ram>=64 disk_space>={disk} direct_port_count>=1 cuda_max_good>=13.0 compute_cap>=800')   # bf16 (Hecate): Ampere or newer
    bad = set((STATE / 'bad_hosts.txt').read_text().split()) if (STATE / 'bad_hosts.txt').exists() else set()   # hosts that failed us
    return [o for o in vast('search', 'offers', q, '-o', 'dph', '--limit', '300') if not o['gpu_name'].startswith('CMP')   # mining cards
            and (o.get('public_ipaddr') or '').strip() not in bad and str(o.get('machine_id')) not in bad]


def typical_lane():
    return dict(id='typical', sense='CW')


def cmd_offers(a):
    offs = search(a.disk); lane = typical_lane()
    rows = sorted(((lane_cost(o, lane, a.disk), o) for o in offs), key=lambda t: t[0]['usd'])
    print(f"{len(offs)} offers pass the filters (reliability>=0.98, download<=$0.005/GB, >=500 Mbps, >=16 cores, >=64 GB RAM, >={a.disk} GB disk, >=24 GB VRAM)")
    print(f"{'offer':>10} {'gpu':<14} {'$/h':>6} {'cores':>5} {'Mbps':>5} {'dl$/GB':>7} {'sto$/GB-mo':>10} {'loc':<14} {'est h':>6} {'est $/lane':>10}")
    for c, o in rows[:a.top]:
        print(f"{o['id']:>10} {o['gpu_name']:<14} {o['dph_total']:>6.3f} {o.get('cpu_cores_effective', 0):>5.0f} {o.get('inet_down', 0):>5.0f} "
              f"{o.get('inet_down_cost') or 0:>7.4f} {o.get('storage_cost') or 0:>10.3f} {str(o.get('geolocation', ''))[:14]:<14} {c['hours']:>6.2f} {c['usd']:>10.3f}")
    by = {}
    for c, o in rows:
        by.setdefault(o['gpu_name'], []).append(c['usd'])
    print('\nbest est. $/lane by GPU type:', ', '.join(f'{g} {min(v):.2f} (n={len(v)})' for g, v in sorted(by.items(), key=lambda kv: min(kv[1]))))
    return rows


def cmd_plan(a):
    lanes = [l for l in json.load(open(HERE / 'lanes.json')) if l['ready']]
    rows = cmd_offers(argparse.Namespace(disk=a.disk, top=0))
    if not rows: print('no offers'); return
    # one lane per box; take the cheapest distinct offers for the first wave, then reuse the best price for the rest
    wave = [o for _, o in rows[:a.boxes]]
    total = 0.0; print(f"\n{'lane':<20} {'sense':<5} {'offer gpu':<14} {'est h':>6} {'est $':>7}  stage hours")
    for i, l in enumerate(lanes):
        o = wave[i % len(wave)]; c = lane_cost(o, l, a.disk); total += c['usd']
        flag = '' if total <= a.budget else '  <- over budget'
        print(f"{l['id']:<20} {l['sense']:<5} {o['gpu_name']:<14} {c['hours']:>6.2f} {c['usd']:>7.2f}  {c['stages']}{flag}")
    print(f"\n{len(lanes)} ready lanes, est. total ${total:.2f} (budget ${a.budget}); {len(wave)} boxes in parallel -> "
          f"about {math.ceil(len(lanes) / len(wave))} lanes per box, wall time ~{math.ceil(len(lanes) / len(wave)) * 4.5:.0f} h")
    print('blocked lanes:', [(l['id'], l['blocker']) for l in json.load(open(HERE / 'lanes.json')) if not l['ready']])


# --------------------------------------------------------------------------------------------------------------------------
def boxes():
    return [json.loads(l) for l in open(BOXES)] if BOXES.exists() else []


def save_boxes(bs):
    BOXES.write_text(''.join(json.dumps(b) + '\n' for b in bs))


def endpoints(b):
    """SSH endpoints of a box: the direct port first, then Vast's SSH proxy (ssh_host:ssh_port). The endpoint that answered
    last goes first.  On 2026-10-12 box b2's direct port died mid-run while the proxy kept working; the driver polled a dead
    port for 40 min with the job DONE, and would have destroyed the box after 45 min without pulling its maps."""
    eps = []
    for h, p in ((b.get('host'), b.get('port')), (b.get('ssh_host'), b.get('ssh_port'))):
        if h and p and (h, int(p)) not in eps:
            eps.append((h, int(p)))
    via = tuple(b['via']) if b.get('via') else None
    if via in eps:
        eps.remove(via); eps.insert(0, via)
    return eps


def _remember(b, h, p):
    if tuple(b.get('via') or ()) != (h, p):
        if b.get('via') or (h, p) != (b.get('host'), b.get('port')):
            print(f"{b.get('label', b.get('id'))}: ssh now via {h}:{p}", flush=True)
        b['via'] = [h, p]
        if b.get('label') and b.get('id'):
            update_box(b, via=b['via'])


def _ssh_opts(p):
    return ['-i', KEY, '-p', str(p), '-o', 'StrictHostKeyChecking=accept-new', '-o', 'LogLevel=ERROR', '-o', 'ConnectTimeout=20']


def ssh(b, cmd, timeout=60):
    """run cmd on the box; fails over from the direct port to the Vast proxy on an ssh-level failure (rc 255 or timeout).
    Raises TimeoutExpired only when every endpoint timed out (callers already handle that)."""
    last = None
    for h, p in endpoints(b):
        try:
            r = subprocess.run(['ssh', *_ssh_opts(p), f'root@{h}', cmd], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            last = e; continue
        if r.returncode not in SSH_FAIL:
            _remember(b, h, p); return r
        last = r
    if isinstance(last, subprocess.TimeoutExpired) or last is None:
        raise last or subprocess.TimeoutExpired('ssh', timeout)
    return last


def rsync(b, src, dst, push=True, extra=()):
    """rsync over ssh with the same direct -> proxy failover as ssh()"""
    last = None
    for h, p in endpoints(b):
        e = 'ssh ' + ' '.join(_ssh_opts(p))
        args = ['rsync', '-az', '-e', e, *extra] + ([*src, f"root@{h}:{dst}"] if push else [f"root@{h}:{src}", dst])
        try:
            r = subprocess.run(args, capture_output=True, text=True, timeout=3600)
        except subprocess.TimeoutExpired:
            last = subprocess.CompletedProcess(args, 30, '', 'timeout'); continue
        if r.returncode not in RSYNC_FAIL:
            _remember(b, h, p); return r
        last = r
    return last if last is not None else subprocess.CompletedProcess([], 255, '', 'no endpoint')


def go_config():
    g = HERE / 'GO'
    if not g.exists():
        sys.exit('refusing to spend: x_vast_plan/GO is missing (the coordinator writes {"budget": USD, "max_boxes": N} to allow renting)')
    return json.loads(g.read_text())


def smoothed_umbilicus(lane):
    out = STATE / f"umb_{lane['scroll']}_s75.json"
    if not out.exists():
        subprocess.run([sys.executable, str(X / 'tools/smooth_umb.py'), lane['umbilicus'], str(out), '--sigma', '75', '--step', '25'], check=True)
    return out


def push(b, lane):
    P = W / 'third_party/pherc343-first-letters'; V = '/home/liam/projects/vesuvius-work'
    ssh(b, f'mkdir -p {V}/fl {V}/x_research2/speed {V}/CANDIDATES/_tools {V}/x_vast_plan {X}/spiral_old {X}/vesuvius/src {P}/mesh /root/lane')
    steps = [
        ([str(W / 'fl/scripts'), str(W / 'fl/remote'), str(W / 'fl/scrolls.json'), str(W / 'fl/aws_fast.cfg')], f'{V}/fl/'),
        ([str(W / 'CANDIDATES/_tools/card.py')], f'{V}/CANDIDATES/_tools/'),
        ([str(HERE / f) for f in ('lane_box.sh', 'ink_lane.py', 'cards_from_tri.py', 'patch_fitter.py', 'snap_guard.py')], f'{V}/x_vast_plan/'),
        ([str(X / 'remote'), str(X / 'tools'), str(X / 'native')], f'{X}/'),
        ([str(W / 'wt/recipe-erwin/volume-cartographer/scripts/spiral') + '/'], f'{X}/spiral_old/'),
        ([str(W / 'wt/recipe-erwin/vesuvius/src/vc3d_fiber_format')], f'{X}/vesuvius/src/'),
        ([str(P / 'mesh/snap_meshes.py')], f'{P}/mesh/'),
        ([str(smoothed_umbilicus(lane))], '/root/umb.json'),
    ]
    for src, dst in steps:
        r = rsync(b, src, dst, extra=('--exclude', '__pycache__', '--exclude', 'build', '--exclude', 'tests'))
        if r.returncode != 0:
            raise RuntimeError(f'push {src[0]} failed: {r.stderr[:300]}')
    r = rsync(b, [str(W / 'x_research2/speed') + '/'], f'{V}/x_research2/speed/', extra=('--include', '*.py', '--exclude', '*'))   # code only (the dir holds ~1 GB of maps)
    if r.returncode != 0:
        raise RuntimeError(f'push speed code failed: {r.stderr[:300]}')
    if lane['tracks'].startswith('local:'):
        src = lane['tracks'].split(':', 1)[1]
        ssh(b, f"mkdir -p /root/lane/ds/{lane['scroll']}")
        r = rsync(b, [src], f"/root/lane/ds/{lane['scroll']}/")
        if r.returncode != 0:
            raise RuntimeError(f'push tracks failed: {r.stderr[:300]}')
    env = dict(SCROLL=lane['scroll'], Z0=lane['z0'], Z1=lane['z1'], TB='L' + lane['scroll'][5:].lower(), UM=lane['um'],
               SENSE=lane['sense'], TRACKS='local' if lane['tracks'].startswith('local:') else 'published', UMB='/root/umb.json',
               MAXW=lane.get('maxw', 40), WCENTER=lane.get('wcenter', ''), QUALITY_THR=lane.get('quality_thr', ''), SHORT_N=8000, NEED_GB=150, CACHE_GB=30)
    ssh(b, "cat > /root/lane.env <<'EOF'\n" + ''.join(f'{k}={v}\n' for k, v in env.items()) + 'EOF')


def cmd_launch(a):
    g = go_config(); bs = [b for b in boxes() if b.get('state') in ('running', 'starting')]
    if len(bs) >= g.get('max_boxes', 1):
        sys.exit(f'max_boxes {g.get("max_boxes", 1)} reached')
    credit = vast('show', 'user').get('credit', 0)
    spent = sum(box_spend(b) for b in boxes())
    if spent >= g['budget'] or credit < 3:
        sys.exit(f'budget guard: spent ${spent:.2f} of ${g["budget"]}, credit ${credit:.2f}')
    lane = next(l for l in json.load(open(HERE / 'lanes.json')) if l['id'] == a.lane)
    if not lane['ready']:
        sys.exit(f"lane {a.lane} is blocked: {lane['blocker']}")
    offer = a.offer
    if not offer:
        rows = sorted(((lane_cost(o, lane), o) for o in search() if not a.gpu or o['gpu_name'] == a.gpu), key=lambda t: t[0]['usd'])
        if not rows: sys.exit('no matching offer')
        offer, est = rows[0][1]['id'], rows[0][0]
        print(f"cheapest est.: offer {offer} {rows[0][1]['gpu_name']} ${est['usd']:.2f} / {est['hours']:.1f} h")
    label = f"vesuvius-lane-{a.lane}"
    r = vast('create', 'instance', str(offer), '--image', IMAGE, '--disk', '200', '--ssh', '--direct', '--label', label)
    b = dict(label=label, lane=a.lane, id=r['new_contract'], offer=offer, state='starting', t0=time.time())
    save_boxes(boxes() + [b]); print('rented', b)
    for _ in range(80):
        i = next((x for x in vast('show', 'instances') if x['id'] == b['id']), None)
        p = ((i or {}).get('ports') or {}).get('22/tcp') or [{}]
        if i and i.get('actual_status') == 'running' and p[0].get('HostPort'):
            b.update(host=(i.get('public_ipaddr') or '').strip(), port=int(p[0]['HostPort']), dph=i.get('dph_total'), gpu=i.get('gpu_name'),
                     ssh_host=i.get('ssh_host'), ssh_port=i.get('ssh_port'))
            update_box(b, **b)
            try:   # attach our key explicitly: on some hosts the account key never reaches the container ('Permission denied (publickey)')
                vast('attach', 'ssh', str(b['id']), (Path(KEY + '.pub')).read_text().strip(), raw=False)
            except Exception as e:
                print('attach ssh:', str(e)[:160])
            break
        time.sleep(15)
    else:
        update_box(b, state='failed_start', t_end=time.time()); vast('destroy', 'instance', str(b['id']), '-y', raw=False); sys.exit('instance never came up; destroyed')
    for _ in range(30):
        if ssh(b, 'true').returncode == 0: break
        time.sleep(10)
    start_lane(b, lane)


def start_lane(b, lane, tries=6):
    for k in range(tries):
        try:
            time.sleep(60 if k else 30)                       # let sshd and the account key settle after the box's onstart (slow hosts: ~3-5 min)
            if k == 3 and b.get('ssh_host'):              # some hosts reset direct SSH: fall back to Vast's SSH proxy
                b.update(host=b['ssh_host'], port=b['ssh_port']); print('switching to the SSH proxy', b['host'], b['port'])
            push(b, lane)
            alive = ssh(b, 'rm -f /root/lane/FAILED; p=$(cat /root/lane/pid 2>/dev/null); [ -n "$p" ] && kill -0 $p 2>/dev/null && echo ALIVE')
            if 'ALIVE' not in alive.stdout:          # start once; the remote shell can hold the session, so don't wait on it
                try:
                    h, p = endpoints(b)[0]                   # the endpoint push() just used
                    subprocess.run(['ssh', *_ssh_opts(p), f"root@{h}", f'cd /root && setsid nohup bash {W}/x_vast_plan/lane_box.sh > /root/lane_box.out 2>&1 < /dev/null &'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
                except subprocess.TimeoutExpired:
                    pass
                time.sleep(15)
            chk = ssh(b, 'p=$(cat /root/lane/pid 2>/dev/null); [ -n "$p" ] && kill -0 $p 2>/dev/null && echo ALIVE $(cat /root/lane/cur_stage 2>/dev/null)')
            if 'ALIVE' not in chk.stdout:
                raise RuntimeError(f'lane_box not running after start: {chk.stdout.strip()[:120]} {chk.stderr.strip()[:120]}')
            b['state'] = 'running'; b.pop('hold', None); update_box(b, **b, hold=None); print('started', b['label'], b.get('gpu'), f"${b.get('dph')}/h", chk.stdout.strip()); return
        except Exception as e:
            print(f"push/start attempt {k + 1} failed: {str(e)[:160]}")
    with open(STATE / 'bad_hosts.txt', 'a') as f:   # don't rent this host again
        f.write(f"{b.get('host', '')}\n")
    update_box(b, state='push_failed', t_end=time.time()); vast('destroy', 'instance', str(b['id']), '-y', raw=False); sys.exit(f"{b['label']}: push failed {tries}x; destroyed")


def box_by_label(label):
    """latest entry for a label, preferring a live one (relaunched lanes reuse labels)"""
    m = [b for b in boxes() if b['label'] == label]
    live = [b for b in m if b.get('state') in ('running', 'starting')]
    return (live or m)[-1]


def cmd_restart(a):
    """stop the box's lane_box process group, re-push code + lane.env from lanes.json, start again (resumes at the first unfinished stage)"""
    b = box_by_label(a.label)
    r = ssh(b, 'p=$(cat /root/lane/pid 2>/dev/null); [ -n "$p" ] && kill -TERM -- -$p 2>/dev/null; sleep 3; '
               '[ -n "$p" ] && kill -0 $p 2>/dev/null && kill -KILL -- -$p 2>/dev/null; rm -f /root/lane/pid; echo stopped $p')
    print(a.label, r.stdout.strip())
    cmd_resume(a)


def cmd_resume(a):
    """push + start the lane on an already-rented box (e.g. after a failed push)"""
    b = box_by_label(a.label)
    i = next(x for x in vast('show', 'instances') if x['id'] == b['id'])
    p = (i.get('ports') or {}).get('22/tcp') or [{}]
    b.update(host=(i.get('public_ipaddr') or '').strip(), port=int(p[0]['HostPort']), dph=i.get('dph_total'), gpu=i.get('gpu_name'),
             ssh_host=i.get('ssh_host'), ssh_port=i.get('ssh_port'))
    update_box(b, **b)
    lane = next(l for l in json.load(open(HERE / 'lanes.json')) if l['id'] == b['lane'])
    start_lane(b, lane)


def update_box(b, **kw):
    bs = boxes()
    for x in bs:
        if x['id'] == b['id']: x.update(kw)
    save_boxes(bs)


_INV = {'t': 0, 'rows': []}


def invoices():
    if time.time() - _INV['t'] > 120:
        try: _INV.update(t=time.time(), rows=vast('show', 'invoices'))
        except Exception: pass
    return _INV['rows']


def box_spend(b):
    """$ charged to this instance so far: invoice lines if present, else dph x elapsed (an upper-bound estimate)"""
    try:
        inv = invoices()
        s = sum(float(r.get('amount') or 0) for r in inv if str(r.get('instance_id') or '') == str(b['id']) or str(b['id']) in (r.get('description') or ''))
        if s > 0: return s
    except Exception:
        pass
    # estimate: a box that left 'running'/'starting' without a t_end (push_failed / failed_start before the 2026-10-12 fix)
    # was destroyed right away, so don't let its estimate grow with the clock (it would trip the global budget guard)
    live = b.get('state') in ('running', 'starting')
    t1 = b.get('t_end') or (time.time() if live else (b.get('last_ok') or b['t0'] + 900))
    return (b.get('dph') or 0.5) * (t1 - b['t0']) / 3600


def pull_and_destroy(b, why, tries=3, wait=60):
    dst = RESULTS / b['lane']; dst.mkdir(parents=True, exist_ok=True)
    for k in range(tries if why == 'done' else 1):   # a DONE box answered seconds ago: a failed pull is a network blip, retry
        r = rsync(b, '/root/lane/results.tar', str(dst) + '/', push=False)
        if r.returncode not in RSYNC_FAIL:
            break
        print(f"{b['label']}: results pull failed rc={r.returncode} ({(r.stderr or '').strip()[:120]}), try {k + 1}", flush=True)
        if k + 1 < tries and why == 'done':
            time.sleep(wait)
    if r.returncode != 0:     # no tarball (failure): bring the logs and stage markers
        rsync(b, '/root/lane/logs', str(dst) + '/', push=False); rsync(b, '/root/lane/timing.tsv', str(dst) + '/', push=False)
        rsync(b, '/root/lane/FAILED', str(dst) + '/', push=False); rsync(b, '/root/lane_box.out', str(dst) + '/', push=False)
        if why == 'done' or r.returncode in RSYNC_FAIL:   # leave a trace: the box is destroyed with its results unpulled
            (dst / 'PULL_FAILED').write_text(f"{int(time.time())} {why} rc={r.returncode} endpoints={endpoints(b)}\n")
    else:
        subprocess.run(['tar', '-C', str(dst), '-xf', str(dst / 'results.tar')])
        # keep the fused ink maps (float16, rsync -z compresses the empty off-sheet areas) so lanes can be re-triaged
        # locally later (glyph+masked-row fusion, reading-face gate, any better scorer) without re-renting
        r2 = rsync(b, '/root/lane/rv2/', str(dst / 'maps') + '/', push=False,
                   extra=('--include', '*/', '--include', '*_d0_*_z.npy', '--include', 'quality_tiles.npy', '--exclude', '*', '--prune-empty-dirs'))
        print(f"{b['label']}: maps pull rc={r2.returncode}")
    vast('destroy', 'instance', str(b['id']), '-y', raw=False)
    update_box(b, state=why, t_end=time.time()); print(f"{b['label']}: {why}, pulled to {dst}, destroyed")


def cmd_poll(a):
    g = json.loads((HERE / 'GO').read_text()) if (HERE / 'GO').exists() else {'budget': 0}
    while True:
        live = [b for b in boxes() if b.get('state') == 'running']
        for b in live:
          try:
            r = None
            try:
                r = ssh(b, 'cat /root/lane/DONE 2>/dev/null && echo @DONE; cat /root/lane/FAILED 2>/dev/null && echo @FAILED; cat /root/lane/heartbeat 2>/dev/null')
                out = r.stdout
            except subprocess.TimeoutExpired:
                out = ''
            now = time.time(); hours = (now - b['t0']) / 3600; usd = box_spend(b)
            hb = next((l.split() for l in out.splitlines() if l.split() and l.split()[0].isdigit() and len(l.split()[0]) >= 9), None)   # '<epoch> <stage>'
            stale = (now - float(hb[0])) / 60 if hb else (now - b.get('last_ok', b['t0'])) / 60
            if r is not None and r.returncode == 0 and hb: update_box(b, last_ok=now, stage=hb[1] if len(hb) > 1 else '')
            print(f"{b['label']}: {hours:.1f} h ${usd:.2f} stage {hb[1] if hb and len(hb) > 1 else '?'} heartbeat {stale:.0f} min ago")
            if '@DONE' in out: pull_and_destroy(b, 'done')
            elif '@FAILED' in out:
                st = (out.split('@DONE')[-1].split('@FAILED')[0].strip().split(':')[0]) or ''
                if st[:2].isdigit() and int(st[:2]) >= 4:   # the fit is done: keep the box for a fix + `fleet.py resume`
                    if b.get('hold') != st:
                        update_box(b, hold=st, hold_t=now); print(f"{b['label']}: FAILED at {st} after the fit -> HOLD (not destroyed); fix and resume")
                        with open(STATE / 'HOLD', 'a') as f: f.write(f"{int(time.time())} {b['label']} {st}\n")
                    elif (now - (b.get('hold_t') or now)) / 60 > a.hold_max_min:   # nobody resumed it: stop paying for it
                        pull_and_destroy(b, 'hold_expired')
                else:
                    pull_and_destroy(b, 'failed')
            elif stale > 45: pull_and_destroy(b, 'stale')
            elif hours > a.max_hours: pull_and_destroy(b, 'over_time')
            elif usd > a.max_usd: pull_and_destroy(b, 'over_budget')
          except Exception as e:
            print(f"{b.get('label')}: poll error {str(e)[:200]}")
        total = sum(box_spend(b) for b in boxes())
        with open(STATE / 'spend.log', 'a') as f:
            f.write(json.dumps(dict(t=int(time.time()), total=round(total, 3), boxes={b['label']: round(box_spend(b), 3) for b in boxes()})) + '\n')
        if g.get('budget') and total >= g['budget']:
            for b in [b for b in boxes() if b.get('state') == 'running']:
                pull_and_destroy(b, 'global_budget')
        if a.once or not [b for b in boxes() if b.get('state') == 'running']:
            break
        time.sleep(a.interval)


def cmd_status(a):
    for b in boxes():
        print(json.dumps({k: b.get(k) for k in ('label', 'id', 'state', 'gpu', 'dph', 'stage')}), f"spend ${box_spend(b):.2f}")
    print('instances on the account:', [(i['id'], i.get('label'), i.get('actual_status')) for i in vast('show', 'instances')])


def cmd_spend(a):
    print(f"credit ${vast('show', 'user').get('credit', 0):.2f}; boxes: " + ', '.join(f"{b['label']} ${box_spend(b):.2f}" for b in boxes()))


def cmd_destroy(a):
    b = box_by_label(a.label); pull_and_destroy(b, 'manual')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('offers'); p.add_argument('--top', type=int, default=15); p.add_argument('--disk', type=int, default=200)
    p = sp.add_parser('plan'); p.add_argument('--budget', type=float, default=150); p.add_argument('--boxes', type=int, default=6); p.add_argument('--disk', type=int, default=200)
    p = sp.add_parser('launch'); p.add_argument('lane'); p.add_argument('--offer'); p.add_argument('--gpu', default='RTX 4090')
    p = sp.add_parser('poll'); p.add_argument('--once', action='store_true'); p.add_argument('--interval', type=int, default=300)
    p.add_argument('--max-hours', type=float, default=10); p.add_argument('--max-usd', type=float, default=8)
    p.add_argument('--hold-max-min', type=float, default=HOLD_MAX_MIN)
    sp.add_parser('status'); sp.add_parser('spend')
    p = sp.add_parser('resume'); p.add_argument('label')
    p = sp.add_parser('restart'); p.add_argument('label')
    p = sp.add_parser('destroy'); p.add_argument('label')
    a = ap.parse_args()
    dict(offers=cmd_offers, plan=cmd_plan, launch=cmd_launch, poll=cmd_poll, status=cmd_status, spend=cmd_spend, destroy=cmd_destroy, resume=cmd_resume, restart=cmd_restart)[a.cmd](a)
