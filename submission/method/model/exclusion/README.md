# Exclusion zone

The zone holds every point of PHerc. 0306B that we showed in any image of the claimed line, across all seven versions of our claim. `check.py` measures how far a render pixel lies from it in 3-D and decides whether the pixel may be used for training.

| file | what it holds |
|---|---|
| `zone.npz` | the zone: 2,386,014 points, in voxels (x, y, z) of volume `20250521133212` |
| `zone.json` | how the zone was built: its sources, margins and checksums |
| `build_zone.py` | the code that built it |
| `check.py` | the distance test |
| `renders.json`, `make_registry.py` | our renders and the meshes they came from, and the script that lists them |
| `meshes/` | the meshes of the five training surfaces |
| `region_fracs.json`, `scratch/` | the measurements below, and the scripts that made them |

The audit of all 23,331 PHerc. 0306B training tiles is `../exclusion_audit.json` (all allowed, the nearest 2.925 mm from the zone), and every tile's position is in `../training_tiles.npz`.

## What the zone covers

The zone comes from two meshes. Both are densified 4× by bilinear interpolation inside valid quads, so neighbouring points are at most 0.043 mm (5 render pixels) apart.

1. The claim mesh, V5Rc: every point whose render pixel falls in the claim image's window, u −30..+30 × v −20..+15 mm, plus 2 mm on the surface. That comes to 1,254,679 points. The claim frame is u = −36 + x / 115.741 and v = −21 + y / 115.741 mm.
2. The mesh of our first version (`w074own`, auto_grown_20261012144243662, 14 Oct): the window u −34..+30 × v −20..+18 mm, which contains its view and its 60 × 36 mm context image, plus 2 mm. That comes to 1,131,335 points. Its 4 cm² area lies within 0.43 mm of the V5Rc surface (median 0.02 mm), so the two meshes describe the same place.

## The rule

`check.py` maps a render pixel to 3-D by bilinear lookup in its tifxyz mesh, using the `vc_render_tifxyz --scale 1` convention (render pixel = grid / scale). A pixel is allowed when its 3-D point is at least 2.0 mm + 0.2 mm + `pad_mm` from every zone point. The 0.2 mm covers the render slab's half-depth (0.17 mm) and the point spacing. A pixel with no valid mesh quad is rejected. Since the distance is measured in 3-D, the rule covers the claimed sheet and every other sheet that passes within reach of it.

`pad_mm` stands for the size of whatever the pixel feeds: half the diagonal of the model's input tile. That is 0.39 mm for a 64 px v8-in tile and 1.6 mm for a 256 px Reader tile. The A-hn training audit checked each tile's centre and corners at pad 0.7 mm, and a 5 × 5 grid of points at pad 0.

`allowed_mask` samples every `step` pixels and adds a tolerance of 0.71 × `step` pixels, so it never allows a pixel that the exact test rejects.

## Use

```python
import sys; sys.path.insert(0, 'method/model/exclusion'); import check

check.allowed(zarr_or_mesh, ys, xs, pad_mm=0.7)                # bool per pixel, True = allowed
check.assert_allowed(zarr_or_mesh, ys, xs, pad_mm=0.7)         # raises if any pixel is rejected
check.allowed_mask(zarr, roi=(y0, y1, x0, x1), pad_mm=0.7)     # full-resolution bool mask
check.distance_mm(zarr_or_mesh, ys, xs)                        # distance to the zone in mm
```

From the shell:

```bash
python check.py ZARR y,x [y,x ...]                             # exit 1 if any point is rejected
python check.py ZARR --mask OUT.npz [--roi y0 y1 x0 x1] [--pad 0.4]
python check.py register ZARR MESH [--offset y x]              # add a render to renders.json
```

`renders.json` lists our renders by their local paths. To test your own render, register it with its mesh or pass `mesh=`. A crop zarr with a `crop.json` (`render`, and `box` = y0, y1, x0, x1) resolves through its parent render. `check.py selftest` reads our local renders, so it only runs on our machine.

## Measured distances

| region | allowed | distance to the zone |
|---|---|---|
| 4,000 random pixels of the claim window on V5Rc | 0 of the 3,924 that have a mesh | |
| our first version's render, 48 × 48 mm around the line | 4.2 % (its far corners) | |
| the screening lane's render of winding 74 | 12.6 % | |
| another candidate row, on lane winding 50 (y 1000–1650, x 3000–6600) | 96.6 % | min 2.08 mm, median 3.10 mm |
| the same row on its own GrowPatch (y 2400–3900, x 1800–4500) | 96.2 % | min 1.89 mm, median 3.10 mm |
| that GrowPatch, whole render | 86.1 % | min 0.18 mm, where its far edge reaches the claimed sheet |
| w052, lower row | 99.8 % | min 2.29 mm |
| w053 rows | 97.9 % | min 1.90 mm |
| w059 band | 23.9 % | min 0.015 mm, median 1.69 mm: most of the band is inside the buffer |
| w109, two rows | 99.97 % | min 3.84 mm |
