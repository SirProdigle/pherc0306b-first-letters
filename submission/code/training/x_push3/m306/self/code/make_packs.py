"""make_packs.py [306|anchors|all] : SELF cloud data packs.  Everything a cloud box needs to train the SELF
arms, already zone-checked and cut locally, so the box needs no meshes, renders, zone file or labels beyond these arrays.
  packs/306_<r>.npz : img (h, w, 24) uint8 = train_self.src_306's padded input (CT zeroed < 2.2 mm from the zone, clip 200, stored
                      order), and per variant v in main / sc / decoy: lab_<v> (h, w) uint8 0 bg / 1 ink / 255 ignored (padded),
                      org_<v> (n, 2) tile top-lefts, ink_<v> (n,) bool; every tile centre + 4 corners passed check.assert_allowed(pad 0.7)
  packs/a_<name>.npz : one anchor source of TRAIN's v8306lib (src_1447 w058 / w060, src_0343, src_human PHerc0139 clusters):
                      img, lab (float16), keep, org, group, info (json)
  packs/MANIFEST.json : files, sha256, sizes, sources and licences."""
import sys, os, json, hashlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import train_self as T, v8306lib as L                                            # noqa: E402
PK = f'{T.SELF}/packs'; os.makedirs(PK, exist_ok=True)
REGIONS = ['A6', 'w052', 'w053', 'w059', 'w109']; VARIANTS = ['main', 'sc', 'decoy']
H0139 = ['20250108000004', '20250108000005', '20250831000000', '20260108000000', '20260112000000', '20260115000000',
         '20260302000000', '20260317000000']


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()


def pack_306(r):
    out = f'{PK}/306_{r}.npz'
    if os.path.exists(out): return
    arrs, infos = {}, {}
    for v in VARIANTS:
        d = T.src_306(r, v)
        if 'img' not in arrs: arrs['img'] = d['img']
        else: assert np.array_equal(arrs['img'][::97, ::97], d['img'][::97, ::97])
        lab8 = np.where(d['keep'] > 0, (d['lab'] > 0.5).astype(np.uint8), 255).astype(np.uint8)
        arrs[f'lab_{v}'] = lab8; arrs[f'org_{v}'] = d['org']; arrs[f'ink_{v}'] = d['ink']; infos[v] = d['info']
        print(r, v, d['info']['tiles'], 'tiles', d['info']['ink_tiles'], 'ink tiles', flush=True)
    np.savez(out + '.tmp.npz', info=json.dumps(infos), **arrs); os.replace(out + '.tmp.npz', out)


def save_anchor(name, d, group):
    out = f'{PK}/a_{name}.npz'
    np.savez(out + '.tmp.npz', img=d['img'], lab=d['lab'].astype(np.float16), keep=d['keep'].astype(np.uint8), org=d['org'],
             group=group, info=json.dumps(d['info'], default=str))
    os.replace(out + '.tmp.npz', out); print('anchor', name, group, len(d['org']), 'tiles', flush=True)


def anchors():
    for w in ('w058', 'w060'):
        if not os.path.exists(f'{PK}/a_{w}.npz'): save_anchor(w, L.src_1447(w), '1447')
    if not os.path.exists(f'{PK}/a_0343.npz'): save_anchor('0343', L.src_0343(), '0343')
    for seg in H0139:
        if any(f.startswith(f'a_h0139_{seg}_') for f in os.listdir(PK)): continue
        for d in L.src_human('PHerc0139', seg):
            save_anchor(f'h0139_{seg}_{d["name"].split("_")[-1]}', d, 'h0139')


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('306', 'all'):
        for r in REGIONS: pack_306(r)
    if what in ('anchors', 'all'): anchors()
    man = {}
    for f in sorted(os.listdir(PK)):
        if f.endswith('.npz') and '.tmp' not in f: man[f] = dict(bytes=os.path.getsize(f'{PK}/{f}'), sha256=sha(f'{PK}/{f}'))
    json.dump(dict(files=man, licences={'PHerc0306B / PHerc1447 / PHerc0139 scans and organiser labels': 'CC BY-NC 4.0 (Vesuvius Challenge)',
                                        'PHerc0343 jeff-j-chen tracings': 'MIT (TRAIN DATA.md)',
                                        'SELF pseudo-labels': 'derived here from cleared models (v8-in MIT, Reader V3 MIT, TRACE v3B, Reader v2)'}),
              open(f'{PK}/MANIFEST.json', 'w'), indent=1)
    print('manifest', len(man), 'files', sum(v['bytes'] for v in man.values()) / 1e9, 'GB')
