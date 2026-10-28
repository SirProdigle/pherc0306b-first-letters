#!/bin/bash
# review2.sh : gallery of unreviewed patches with row score >= 3, plus 6 random unreviewed ones; marks all scored patches reviewed
W=/home/liam/projects/vesuvius-work; touch $W/fl/reviewed.txt
python3 - <<'PY'
import json, glob, random, subprocess, os
W='/home/liam/projects/vesuvius-work'
rev=set(l.strip() for l in open(f'{W}/fl/reviewed.txt'))
shown=set(l.strip() for l in open(f'{W}/fl/shown.txt')) if os.path.exists(f'{W}/fl/shown.txt') else set()
hi=[]; rest=[]
for f in glob.glob(f'{W}/fl/*/screen2/maps/*/spectral.json'):
    d=os.path.dirname(f)+'/'
    if '/PHerc0343/' in d: continue
    try: j=json.load(open(f))
    except Exception: continue
    if j.get('skipped'): rev.add(d); continue
    if j.get('version') not in (2, 3): continue
    if j.get("max_ink",-9)>=4.5 or (j.get('version')==2 and (j.get("max_aligned",-9)>=4.5 or j.get("max",-9)>=9)):
        if d not in shown: hi.append((j.get('max',-9),d))
    elif d not in rev: rest.append((j.get('max',-9),d))
hi.sort(reverse=True); random.shuffle(rest)
pick=[d for _,d in hi][:12]+[d for _,d in rest][:6]
print(f'high={len(hi)} rest={len(rest)} showing {len(pick)}')
if pick:
    subprocess.run([f'{W}/envs/vcpy/bin/python',f'{W}/fl/scripts/gallery.py',f'{W}/fl/viz/rev2_latest.jpg',*[p.rstrip('/') for p in pick],'--tile','430','--cols','2','--sort'],stderr=subprocess.DEVNULL)
for _,d in hi+rest: rev.add(d)
open(f'{W}/fl/reviewed.txt','w').write('\n'.join(sorted(rev))+'\n')
open(f'{W}/fl/shown.txt','a').write(''.join(p+'\n' for p in pick))
PY
