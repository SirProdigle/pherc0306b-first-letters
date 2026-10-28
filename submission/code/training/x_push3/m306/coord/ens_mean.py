"""ens_mean.py OUT.npy MAP1 MAP2 ... : deep-ensemble map = per-pixel mean of member probability maps (NaN where any member is NaN)."""
import sys, json
import numpy as np
out, maps = sys.argv[1], sys.argv[2:]
acc = None
for m in maps:
    a = np.load(m).astype(np.float32); acc = a if acc is None else acc + a
acc /= len(maps)
np.save(out, acc.astype(np.float16)); json.dump(dict(kind='ensemble_mean', members=maps), open(out[:-4] + '.json', 'w'), indent=1)
print('wrote', out)
