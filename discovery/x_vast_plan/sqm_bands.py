"""sqm_bands.py [--width 1500] : clearest z band per scroll from the Scan-Quality-Map atlas (x_vast_plan/sqm_atlas.json,
docs/atlas.json of github.com/jcooperkai-sys/Scan-Quality-Map, MIT, fetched 2026-10-12; data only, none of its code run).

Profile bins are 768 slices; a band is the run of bins covering --width slices whose block-weighted median quality is
highest, among bins holding >= 50 % of the scroll's peak block count (so the band sits inside the papyrus, not at an end).
Quality is only comparable within one scan protocol (8.64 um/116 keV vs 9.362 um/113 keV)."""
import json, sys, os, math
W = int(sys.argv[sys.argv.index('--width') + 1]) if '--width' in sys.argv else 1500
A = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sqm_atlas.json')))


def best_band(s, width=W):
    pr = s['profile']; mb = max(p['blocks'] for p in pr); nb = math.ceil(width / 768)
    best = None
    for frac in (0.5, 0.25, 0.0):
        for i in range(len(pr) - nb + 1):
            run = pr[i:i + nb]
            if min(p['blocks'] for p in run) < frac * mb: continue
            q = sum(p['median_quality'] * p['blocks'] for p in run) / max(1, sum(p['blocks'] for p in run))
            if best is None or q > best[0]: best = (q, run[0]['z'], run[-1]['z'] + 768)
        if best: break
    q, z0, z1 = best
    c = (z0 + z1) // 2; z0 = c - width // 2; z0 -= z0 % 100        # centre a width-slice band, round to 100
    return dict(scroll=s['scroll'], um=s['voxel_um'], median=round(s['median_quality'], 3), clear=round(s['share_clear'], 2),
                hazy=round(s['share_hazy'], 2), band=[z0, z0 + width], band_q=round(q, 3), atlas_best_z=s['best_band_z'],
                z_extent=pr[-1]['z'] + 768)


def band_quality(s, z0, z1):
    """block-weighted median quality of the profile bins overlapping [z0, z1)"""
    run = [p for p in s['profile'] if p['z'] < z1 and p['z'] + 768 > z0]
    return round(sum(p['median_quality'] * p['blocks'] for p in run) / max(1, sum(p['blocks'] for p in run)), 3) if run else None


if __name__ == '__main__':
    for s in sorted(A, key=lambda s: (s['voxel_um'], -s['median_quality'])):
        b = best_band(s)
        print(f"{b['scroll']:<11} {b['um']:<6} med {b['median']:.2f} clear {b['clear']:.2f} hazy {b['hazy']:.2f} "
              f"band z {b['band'][0]}-{b['band'][1]} (q {b['band_q']:.2f}; atlas best z {b['atlas_best_z']}; extent {b['z_extent']})")
