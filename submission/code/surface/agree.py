"""agree.py CARRIER_DIR TRACK_DIR : per pre-registered region, where do the surfaces agree along the carrier normal?
Offsets: A, big, R1 crossings (carrier.npz), V2 (CT-tracked sheet + delta), V3 (m7-tracked sheet), compared on the sheet level
(V2 - delta = CT sheet peak; for A/big/R1/V3 the surface itself; the CT peak sits ~delta below the face).  Share of nodes within
4 vx (same sheet layer) and within 10 vx (same sheet) of each other."""
import sys, json
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
REG = {'claim rect': (-19.9, 12.3, -7.8, 4.6), 'L line band': (-15, 12, -2.5, 2.5), 'RA rows above': (-20, 15, -8, -2.5),
       'CA column above': (-30, 25, -20, -8), 'RB rows below': (-20, 15, 2.5, 8), 'CB column below': (-30, 25, 8, 20),
       'LL line row left': (-35, -20, -4, 4), 'LR line row right': (12, 30, -4, 4)}         # P1's pre-registered regions (evalP1.REG)
U0, V0 = -36.0, -21.0
C = np.load(f'{sys.argv[1]}/carrier.npz'); T = np.load(f'{sys.argv[2]}/track.npz'); T3 = np.load(f'{sys.argv[2]}/track_m7.npz')
U, V = C['u'], C['v']; d = float(T['delta'])
S = {'A': C['x_A'], 'big': C['x_big'], 'R1': C['x_R1'], 'CTtrack': T['V2_trackP'] - d + 2.3 - 2.3, 'm7track': T3['V2_trackP']}
S['CTtrack'] = T['V2_trackP'] - d          # CT sheet peak
pairs = [('A', 'big'), ('A', 'R1'), ('A', 'm7track'), ('big', 'm7track'), ('R1', 'm7track'), ('A', 'CTtrack'), ('m7track', 'CTtrack')]
res = {}
print(f'delta {d:.2f}; median (m7track - CTpeak) over the line band:', end=' ')
band = (np.abs(V) <= 2.5) & (U >= -15) & (U <= 12)
print(np.nanmedian((S['m7track'] - S['CTtrack'])[band]).round(2))
for k, (u0, u1, v0, v1) in REG.items():
    m = (U >= u0) & (U < u1) & (V >= v0) & (V < v1); r = {}
    for a, b in pairs:
        x = S[a][m] - S[b][m]; f = np.isfinite(x)
        r[f'{a}-{b}'] = (round(float(f.mean()), 2), round(float((np.abs(x[f]) <= 4).mean()), 2) if f.any() else None,
                         round(float((np.abs(x[f]) <= 10).mean()), 2) if f.any() else None)
    res[k] = r
    print(k); [print(f'   {p:18s} both {v[0]:.2f}  |d|<=4 {v[1]}  |d|<=10 {v[2]}') for p, v in r.items()]
json.dump(res, open(f'{sys.argv[2]}/agree.json', 'w'), indent=1)
