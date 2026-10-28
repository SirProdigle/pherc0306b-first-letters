"""neutral_controls.py I1_CONTROLS_DIR OUTDIR : I1's control panels (Reader v2 + TRACE, threshold rule A1.1, primary rung, each window's
own levels) re-framed with neutral headers (lane I2).  The pixels come from I1's saved arrays q4_<name>_{d,bg,valid}.npy
(controls/, written by I1's build_q4_controls.py) through the same claimlib_v2.q4_rgb; only the header text differs."""
import os, sys, json
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import claimlib_v2 as L                                                            # noqa: E402

TXT = {  # name -> (output file, neutral description)
    'blank_A_V0': ('ctrl_blank_papyrus', 'blank papyrus of the same sheet of PHerc0306B (window u -11.2..+28.8 x v +5.8..+20.8 mm of the claim frame, '
                   'chosen by a rule fixed in advance); its own top 1 % is painted by construction'),
    'shuffle_A_orig': ('ctrl_depth_shuffle_original_order', 'the line\'s area of PHerc0306B, layers in their true order (Reader v2 + TRACE at depth 0, no TTA)'),
    'shuffle_A_shuf1': ('ctrl_depth_shuffle_shuffled', 'the same area with the layers the models read permuted (seed 1), same models and recipe'),
    'known_1447_w060': ('ctrl_known_1447_w060', 'known text: PHerc1447 winding w060 (public ink labels exist), shown as stored (upright)'),
    'known_1447_w058': ('ctrl_known_1447_w058', 'known text: PHerc1447 winding w058 (public ink labels exist), shown as stored (upright)'),
    'known_0343_PANTOGI': ('ctrl_known_0343_PANTOGI', 'known text: PHerc0343, the traced PANTOGINON crop, reading orientation'),
    'known_0343_lane': ('ctrl_known_0343_lane', 'known text: PHerc0343 winding w048, a window away from the traced crop, reading orientation'),
    'artefact_0813_atlas': ('ctrl_artefact_0813_atlas', 'known ARTEFACT: the PHerc0813 gap-pocket line (9.36 um scan; no CT layer shown)'),
    'artefact_0826_hook': ('ctrl_artefact_0826_hook', 'known ARTEFACT: the PHerc0826 hook (9.36 um scan; no CT layer shown)'),
}


def main():
    src, out = sys.argv[1:3]; os.makedirs(out, exist_ok=True)
    lv = json.load(open(f'{src}/q4_controls.json'))
    for name, (fn, desc) in TXT.items():
        d = np.load(f'{src}/q4_{name}_d.npy').astype(np.float32); bg = np.load(f'{src}/q4_{name}_bg.npy').astype(np.float32)
        v = np.load(f'{src}/q4_{name}_valid.npy')
        l = lv[name]['levels']
        hdr = [f'CONTROL | Reader v2 + TRACE v3B | display rule fixed in advance, primary rung, this window\'s own levels '
               f'(L {l["L"]:.2f}, p99 {l["H"]:.2f}, p99.9 {l["P"]:.2f}); dark = ink',
               desc]
        img = L.frame_image(L.q4_rgb(d, bg, v), hdr, [], margin=40)
        cv2.imwrite(f'{out}/{fn}.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(fn, img.shape, round(float((d[v] > 0).mean()), 4))


if __name__ == '__main__':
    main()
