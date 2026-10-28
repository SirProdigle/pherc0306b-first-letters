"""drive.py BOXNAME JOBS_FILE [--gpu "RTX 5090,RTX 4090"] [--max-usd 4.5] [--max-hours 9]

Rents ONE Vast box for x_deepread/box.sh, pushes code + meshes + the 0343 control, starts it, polls every 5 min, pulls
/root/dr/out (maps + ct20), logs and control.json into x_deepread/results/BOXNAME/, then destroys the box (-y).
Uses x_vast_plan/fleet.py helpers (search filters incl. cuda>=13, bad_hosts, ssh/rsync) but keeps its OWN state file
(x_deepread/state/BOXNAME.json) so the lane poller never touches these boxes.
JOBS_FILE lines: name<TAB>local_tifxyz_dir<TAB>faces[<TAB>scroll]   (meshes are pushed to x_deepread/push/meshes/<name>/;
  scroll = a key of fl/scrolls.json, default PHerc0306B)
"""
import sys, os, json, time, subprocess, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'x_vast_plan'))   # the fleet.py next to this file (ssh/rsync with proxy failover)
import fleet as F                                                    # noqa: E402
W = Path('/home/liam/projects/vesuvius-work'); D = Path(os.environ.get('DR_DIR', W / 'x_deepread'))   # DR_DIR: test sandbox
POLL_S = int(os.environ.get('DR_POLL_S', 300))
ap = argparse.ArgumentParser(); ap.add_argument('box'); ap.add_argument('jobs')
ap.add_argument('--gpu', default='RTX 5090,RTX 4090'); ap.add_argument('--max-usd', type=float, default=4.5)
ap.add_argument('--max-hours', type=float, default=9); ap.add_argument('--min-cores', type=int, default=24)
ap.add_argument('--max-dph', type=float, default=0.8)          # never rent above this $/h (13 Oct: the cheapest match was once $2.00)
ap.add_argument('--exclude-geo', default='CN,China,HK,Hong Kong')   # setup needs GitHub/HF/S3 us-east-1: slow or blocked from these
a = ap.parse_args()
ST = D / 'state'; ST.mkdir(exist_ok=True); SF = ST / f'{a.box}.json'; RES = D / 'results' / a.box; RES.mkdir(parents=True, exist_ok=True)
log = lambda s: print(f'{time.strftime("%H:%M:%S")} {a.box}: {s}', flush=True)
b = json.load(open(SF)) if SF.exists() else {}
save = lambda: json.dump(b, open(SF, 'w'), indent=1)


def sh(cmd, timeout=90):
    via = b.get('via')
    try:
        return F.ssh(b, cmd, timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess([], 255, '', 'timeout')
    finally:
        if b.get('via') != via:
            save(); log(f"ssh endpoint now {b.get('via')}")      # puller.sh reads the state file: it follows the switch


def destroy(why):
    try:
        F.vast('destroy', 'instance', str(b['id']), '-y', raw=False)
    except Exception as e:
        log(f'destroy error {e}')
    b.update(state=why, t_end=time.time()); save(); log(f'destroyed ({why})')


def pull(tries=1):
    for k in range(tries):
        r = F.rsync(b, '/root/dr/out/', str(RES / 'out') + '/', push=False, extra=('-z',))
        if r.returncode not in F.RSYNC_FAIL: break
        log(f'pull try {k + 1} failed rc={r.returncode}'); k + 1 < tries and time.sleep(60)
    for f in ('/root/dr/logs', '/root/dr/control.json', '/root/dr/FAILED', '/root/dr/DONE', '/root/dr_box.out'):
        F.rsync(b, f, str(RES) + '/', push=False)
    log(f'pulled rc={r.returncode} via {b.get("via") or (b.get("host"), b.get("port"))}')
    if r.returncode != 0:
        (RES / 'PULL_FAILED').write_text(f'{int(time.time())} rc={r.returncode} endpoints={F.endpoints(b)}\n')
    return r.returncode


jobs = [l.rstrip('\n').split('\t') for l in open(a.jobs) if l.strip()]
if not b.get('id'):
    gpus = [g.strip() for g in a.gpu.split(',')]
    offers = [o for o in F.search(disk=150) if o['gpu_name'] in gpus and (o.get('cpu_cores_effective') or 0) >= a.min_cores
              and o['dph_total'] <= a.max_dph
              and not any(g.strip() and g.strip() in (o.get('geolocation') or '') for g in a.exclude_geo.split(','))]
    offers.sort(key=lambda o: o['dph_total'])
    if not offers: sys.exit(f'no offer at <= ${a.max_dph}/h')
    o = offers[0]; log(f"offer {o['id']} {o['gpu_name']} ${o['dph_total']:.3f}/h cores {o.get('cpu_cores_effective')} {o.get('geolocation')}")
    r = F.vast('create', 'instance', str(o['id']), '--image', F.IMAGE, '--disk', '150', '--ssh', '--direct', '--label', f'vesuvius-deepread-{a.box}')
    b.update(id=r['new_contract'], offer=o['id'], dph=o['dph_total'], gpu=o['gpu_name'], t0=time.time(), state='starting'); save()


def wait_running():
    for _ in range(80):
        try:
            i = next((x for x in F.vast('show', 'instances') if x['id'] == b['id']), None)
        except Exception as e:                       # the Vast API times out now and then
            log(f'show instances: {str(e)[:80]}'); time.sleep(20); continue
        p = ((i or {}).get('ports') or {}).get('22/tcp') or [{}]
        if i and i.get('actual_status') == 'running' and p[0].get('HostPort'):
            b.update(host=(i.get('public_ipaddr') or '').strip(), ip=(i.get('public_ipaddr') or '').strip(), port=int(p[0]['HostPort']),
                     ssh_host=i.get('ssh_host'), ssh_port=i.get('ssh_port')); save()
            try:
                F.vast('attach', 'ssh', str(b['id']), Path(F.KEY + '.pub').read_text().strip(), raw=False)
            except Exception as e:
                log(f'attach: {str(e)[:120]}')
            return True
        time.sleep(15)
    return False


if b.get('id') and not b.get('host'):
    if not wait_running():
        destroy('never_came_up'); sys.exit(1)

if b.get('state') == 'starting':
    ok = False
    for k in range(8):
        time.sleep(60 if k else 30)
        if k == 4 and b.get('ssh_host'):
            b.update(host=b['ssh_host'], port=b['ssh_port']); save(); log('switching to the ssh proxy')
        if sh('true').returncode != 0:
            continue
        sh(f'mkdir -p {W}/fl {W}/x_research2/speed {W}/x_zoom {W}/x_deepread/push/meshes')
        steps = [([str(W / 'fl/remote')], f'{W}/fl/'), ([str(W / 'fl/scrolls.json'), str(W / 'fl/aws_fast.cfg')], f'{W}/fl/'),
                 ([str(D / f) for f in ('ink_full.py', 'control343.py', 'box.sh')], f'{W}/x_deepread/'),
                 ([str(D / 'push/control343.npz')], f'{W}/x_deepread/push/')]
        good = all(F.rsync(b, s, d, extra=('--exclude', '__pycache__')).returncode == 0 for s, d in steps)
        good &= F.rsync(b, [str(W / 'x_research2/speed') + '/'], f'{W}/x_research2/speed/', extra=('--include', '*.py', '--exclude', '*')).returncode == 0
        good &= F.rsync(b, [str(W / 'x_zoom') + '/'], f'{W}/x_zoom/', extra=('--include', '*.py', '--exclude', '*')).returncode == 0
        lines = []
        for job in jobs:
            name, src, faces = job[:3]; scroll = job[3] if len(job) > 3 else 'PHerc0306B'; rec = job[4] if len(job) > 4 else 'mq'
            good &= F.rsync(b, [src.rstrip('/') + '/'], f'{W}/x_deepread/push/meshes/{name}/').returncode == 0
            lines.append(f'{name} {W}/x_deepread/push/meshes/{name} {faces} {scroll} {rec}')
        sh(f"cat > {W}/x_deepread/push/jobs.txt <<'EOF'\n" + '\n'.join(lines) + '\nEOF')
        if not good:
            log(f'push attempt {k + 1} failed'); continue
        try:
            subprocess.run(['ssh', '-i', F.KEY, '-p', str(b['port']), '-o', 'StrictHostKeyChecking=accept-new', '-o', 'LogLevel=ERROR', f"root@{b['host']}",
                            f'cd /root && setsid nohup bash {W}/x_deepread/box.sh > /root/dr_box.out 2>&1 < /dev/null &'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        except subprocess.TimeoutExpired:
            pass
        time.sleep(15)
        c = sh('p=$(cat /root/dr/pid 2>/dev/null); [ -n "$p" ] && kill -0 $p 2>/dev/null && echo ALIVE')
        if 'ALIVE' in c.stdout:
            ok = True; break
        log(f'start attempt {k + 1}: not alive')
    if not ok:
        with open(F.STATE / 'bad_hosts.txt', 'a') as f: f.write(f"{b.get('ip') or b.get('host', '')}\n")   # the box's IP, not the ssh proxy
        destroy('push_failed'); sys.exit(1)
    b['state'] = 'running'; save(); log(f"started on {b.get('gpu')} ${b.get('dph'):.3f}/h")

last_ok = time.time()
while b.get('state') == 'running':
    r = sh('cat /root/dr/DONE 2>/dev/null && echo @DONE; cat /root/dr/FAILED 2>/dev/null && echo @FAILED; cat /root/dr/heartbeat 2>/dev/null; '
           'ls /root/dr/out 2>/dev/null | wc -l; ls /root/dr/out/*/.done 2>/dev/null | wc -l')
    hours = (time.time() - b['t0']) / 3600; usd = hours * (b.get('dph') or 0.6)
    if r.returncode == 0: last_ok = time.time()
    log(f"{hours:.2f} h ~${usd:.2f} | {' | '.join(r.stdout.split()[-4:])}")
    if '@DONE' in r.stdout:
        pull(tries=3); destroy('done'); break
    if '@FAILED' in r.stdout:
        pull(); destroy('failed'); break
    if usd > a.max_usd or hours > a.max_hours:
        pull(); destroy('over_budget'); break
    if time.time() - last_ok > 45 * 60:   # both the direct port and the proxy failed for 45 min: last pull attempt, then stop paying
        pull(); destroy('unreachable'); break
    time.sleep(POLL_S)
try:
    inv = F.vast('show', 'invoices'); spent = sum(float(x.get('amount') or 0) for x in inv if str(b['id']) in (x.get('description') or '') or str(x.get('instance_id')) == str(b['id']))
    b['invoiced_usd'] = round(spent, 3); save(); log(f'invoiced so far ${spent:.2f}')
except Exception as e:
    log(f'invoices: {e}')
