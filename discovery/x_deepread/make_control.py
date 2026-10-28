"""make_control.py OUT.npz : the 0343 positive-control crop for control343.py (layers 23..43 of the validate343 crop)."""
import sys
import numpy as np, zarr
W = '/home/liam/projects/vesuvius-work'
Y0, Y1, X0, X1 = 450, 2350, 2400, 5300
vol = np.asarray(zarr.open(f'{W}/fl/0343/concat_66.zarr', mode='r')['0'][23:44, Y0:Y1, X0:X1])
lab = np.load(f'{W}/x_px343/label_render.npy')[Y0:Y1, X0:X1]; msk = np.load(f'{W}/x_px343/mask_render.npy')[Y0:Y1, X0:X1]
np.savez_compressed(sys.argv[1], vol=vol, lab=lab.astype(np.uint8), msk=msk.astype(np.uint8))
print(vol.shape, lab.sum(), msk.sum())
