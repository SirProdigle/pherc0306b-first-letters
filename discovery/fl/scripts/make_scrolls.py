import json, subprocess, urllib.request
B='https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com'
elig=json.load(open('/home/liam/projects/villa/scrollprize.org/src/data/prizeEligibility.json'))['first-letters-2027']
def ls(prefix):
    out=subprocess.run(['aws','s3','ls','--no-sign-request',f's3://vesuvius-challenge-open-data/{prefix}'],capture_output=True,text=True).stdout
    return [l.split()[-1] for l in out.splitlines() if l.strip()]
cfg={}
for e in elig:
    s,v=e['scroll'],e['volume']
    vols=[x.strip('/') for x in ls(f'{s}/volumes/') if x.startswith(v)]
    surf=[x.strip('/') for x in ls(f'{s}/representations/predictions/surfaces/') if x.startswith(v) and x.strip('/').endswith('.zarr')]
    ng=[x.strip('/') for x in ls(f'{s}/representations/predictions/surfaces/') if x.startswith(v) and 'normal-grids' in x]
    if not (vols and surf):
        print('skip',s,vols,surf); continue
    vurl=f'{B}/{s}/volumes/{vols[0]}'
    za=json.load(urllib.request.urlopen(vurl+'/0/.zarray'))
    um=float(vols[0].split('-')[1].replace('um',''))
    cfg[s]=dict(name=s,volume_url=vurl,surface_url=f'{B}/{s}/representations/predictions/surfaces/{surf[0]}',
                ngrid_url=(f'{B}/{s}/representations/predictions/surfaces/{ng[0]}' if ng else ''),voxel_um=um,shape=za['shape'])
    print(s,um,za['shape'],'ngrid' if ng else 'NO-NGRID')
json.dump(cfg,open('/home/liam/projects/vesuvius-work/fl/scrolls.json','w'),indent=1)
