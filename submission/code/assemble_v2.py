"""assemble_v2.py [DOCKER_TEST_DIR] : fill the v2 DRAFT package around its image, mesh and letter table (lane I2).
Re-runnable.  Copies (never moves) from the lanes' dirs and from v1's package (read only):
  methodology/ evidence/ supplementary/ code/ docker/ README.md DRAFT_STATUS_owner_only.md OTHER_INFORMATION_form_text_owner_only.md
then writes SHA256SUMS.  The image, mesh/ and letter_dimensions.* come from make_claim_image_v2.py, make_mesh_v2.py, letter_table_v2.py.
DOCKER_TEST_DIR: the pipeline test's /out (compare_*.json, render_vs_P1.json, logs), summarised into evidence/ and docker/README.md."""
import os, sys, json, glob, shutil, hashlib
W = '/home/liam/projects/vesuvius-work'
HERE = f'{W}/x_refine/I2_package'; TX = f'{HERE}/text'
PKG = os.environ.get('I2_PKG', f'{W}/x_refine/trials/I2_package/SUBMISSION_v2_DRAFT_owner_only')   # I2_PKG: a copy (dry runs)
V1 = f'{W}/CANDIDATES/SUBMISSION_FINAL_PHerc0306B'                                   # read only
I1S = f'{W}/x_refine/trials/I1_display_read/claim_v2_owner_only/supp'                 # read only
V1D = f'{W}/x_submit/docker_final'                                                    # read only
MESH = 'PHerc0306B_20250521133212_w074own_touchup_V5Rc'
SUPP_I1 = {}                                                                          # neutral_controls.py writes the control panels
CODE = ['make_claim_image_v2.py', 'claimlib_v2.py', 'build_variants.py', 'ink_v2.py', 'ink_v8.py', 'compare_v2.py', 'letter_table_v2.py',
        'make_mesh_v2.py', 'blank_strip.py', 'variant_controls.py', 'shuffle_control.py', 'v8q_known_text.py', 'known_text_panels.py',
        'mirror_control.py', 'supp_views.py', 'neutral_controls.py', 'set_lead.sh', 'assemble_v2.py', 'compare_views.py', 'letter_table_label3.py']
MODELS = ['MODEL_CARD.md', 'training_done.json', 'training_log.json', 'judge_v8Q.json', 'bar_A_line.json', 'train_v8.py', 'v8lib.py', 'p4a_judge.py']
SURF = ['p1geom.py', 'carrier.py', 'm7prof.py', 'track.py', 'touchup.py', 'reflatten.py', 'metric.py', 'recto.py', 'adopt_check.py', 'agree.py',
        'build_surface.sh', 'DECISION_touchup.json']
DOCK = ['Dockerfile', 'Dockerfile.from_v1', 'run.sh', 'make_context.sh', 'build.sh']


def sha(p):
    h = hashlib.sha256(); f = open(p, 'rb')
    for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def cp(a, b):
    os.makedirs(os.path.dirname(b), exist_ok=True); shutil.copy2(a, b)


def repro_text(T):
    if not T or not os.path.exists(f'{T}/compare_rt.json'): return '(the GPU test has not run yet)'
    rt = json.load(open(f'{T}/compare_rt.json')); rp = json.load(open(f'{T}/render_vs_P1.json')); ri = sum(v['identical'] == 1.0 for v in rp.values())
    ex = {k: json.load(open(f'{T}/{k}_box_exact.json')) for k in ('v8', 'v8q') if os.path.exists(f'{T}/{k}_box_exact.json')}
    cv = json.load(open(f'{T}/compare_variants.json')) if os.path.exists(f'{T}/compare_variants.json') else {}
    m = lambda k: rt['models'][k]                                                     # noqa: E731
    f = lambda x: f"max |diff| {x['max_abs_x255']}/255, identical {100 * x['identical_frac']:.2f} %, r {x['pearson_r']}"   # noqa: E731
    lines = [f"- **Render**, in the container from the mesh, offline (`VOLUME` = a local copy of the CT chunks it needs): {ri} of 41 layers are "
             f"byte-identical to the render the shipped maps were made from (4900 × 7780 px).",
             f"- **Reader v2** and **TRACE**, over the shown area: byte-identical files to the shipped maps ({f(m('rv2'))}; {f(m('tr'))}).",
             f"- **CT surface layer**: identical ({100 * rt['ct_layer20']['identical_frac']:.2f} % of pixels)."]
    for k, nm in (('v8', 'v8-in base'), ('v8q', 'v8Q')):
        if k in ex:
            e = ex[k]; lines.append(f"- **{nm}**, over the 4 cm² box ({e['box_px']:,} px, box-only pass): {e['differing_px']} pixels differ from the shipped "
                                    f"context map, each by one float16 step (max |diff| {e['max_abs']:.6f} = {e['max_abs_x255']}/255).")
    if cv:
        same = [n[:-4] for n, v in cv.items() if v['png_identical']]; diff = {n[:-4]: v for n, v in cv.items() if not v['png_identical']}
        lines.append(f"- **Every view** rebuilt from the container's own maps (v8-in and v8Q new inside the box, shipped outside): byte-identical PNGs for "
                     f"{', '.join(same)}" + ('; ' + '; '.join(f"{n}: {v['pixels_differing']} pixel(s) differ by at most {v['max_abs_diff']} grey level"
                                                             for n, v in diff.items()) if diff else '') + '.')
    lines.append("- Files: `evidence/docker_test/` (`compare_*.json`, `*_box_exact.json`, `render_vs_P1.json`, logs).")
    return '\n'.join(lines)


def views_table(lead):
    """the views of the package with their controls at their own levels (variant_controls.json, shuffle_variants.json)"""
    import importlib.util
    spec = importlib.util.spec_from_file_location('bv', f'{HERE}/build_variants.py'); bv = importlib.util.module_from_spec(spec); spec.loader.exec_module(bv)
    T = f'{W}/x_refine/trials/I2_package'
    vc = json.load(open(f'{T}/variant_controls.json'))
    sv = json.load(open(f'{T}/shuffle_variants.json')) if os.path.exists(f'{T}/shuffle_variants.json') else {}
    names = {'rt': 'Reader v2 + TRACE v3B', 'rv8': 'Reader v2 + v8-in', 'rv8q': 'Reader v2 + v8Q'}
    out = ['| view | file | ink | display | blank strip, same levels | box | depth shuffle, box, same levels |', '|---|---|---|---|---|---|---|']
    for k, v in bv.VARIANTS.items():
        ck = f"{v['pair']}_{v['display']}" + (f"_{v['rung']}" if v['display'] == 'threshold' else '')
        c = vc.get(ck); sh = sv.get(k)
        f = f"`{MESH}.png` (main image)" if k == lead else f"`supplementary/{v['file']}.png`"
        if v['display'] == 'threshold':
            disp = f"threshold rule, {'primary' if v['rung'] == 'primary' else 'secondary'} rung (L = {0.35 if v['rung'] == 'primary' else 0.5} × p99), pale CT"
            bl = f"{100 * c['strip']['painted']:.2f} % painted ({100 * c['strip']['half_tone']:.2f} % at half tone)" if c else ''
            bx = f"{100 * c['box']['painted']:.1f} % painted" if c else ''
            shs = (f"{100 * sh['original']['painted']:.1f} % → {100 * sh['shuffled']['painted']:.2f} % painted" if sh else 'pending')
        else:
            disp = 'grey rule (median .. p99.9 of the shown area, no threshold), ' + ('pale CT' if v['bg'] == 'ct' else 'no CT')
            bl = f"p99 tone {c['strip']['p99_tone']:.2f}; {100 * c['strip']['half_tone']:.2f} % darker than half tone" if c else ''
            bx = f"p99 tone {c['box']['p99_tone']:.2f}; {100 * c['box']['half_tone']:.1f} % darker than half tone" if c else ''
            shs = (f"half tone {100 * sh['original']['half_tone']:.1f} % → {100 * sh['shuffled']['half_tone']:.2f} %; p99 tone {sh['original']['p99_tone']:.2f} → "
                   f"{sh['shuffled']['p99_tone']:.2f}" if sh else 'pending')
        out.append(f"| {v['view']} | {f} | {names[v['pair']]} | {disp} | {bl} | {bx} | {shs} |")
    return '\n'.join(out) + ('\n\nThe bar for a threshold view, fixed in advance: at most 1 % of the blank strip painted at that view\'s own levels. '
                             'View 3 and the Reader v2 + v8-in primary rung are shown for comparison only.')


def evidence_tables():
    """{{BLANK_TABLE}}, {{SHUFFLE_TABLE}}, {{KNOWN_TABLE}} of evidence/README.md"""
    import importlib.util
    spec = importlib.util.spec_from_file_location('bv', f'{HERE}/build_variants.py'); bv = importlib.util.module_from_spec(spec); spec.loader.exec_module(bv)
    T = f'{W}/x_refine/trials/I2_package'; vc = json.load(open(f'{T}/variant_controls.json'))
    sv = json.load(open(f'{T}/shuffle_variants.json')) if os.path.exists(f'{T}/shuffle_variants.json') else {}
    kt = json.load(open(f'{T}/known_text/known_text_panels.json')) if os.path.exists(f'{T}/known_text/known_text_panels.json') else {}
    names = {'rt': 'Reader v2 + TRACE', 'rv8': 'Reader v2 + v8-in', 'rv8q': 'Reader v2 + v8Q'}
    b = ['| view | pair, rule | blank strip | 4 cm² box | bar |', '|---|---|---|---|---|']
    s_ = ['| view | pair, rule | box, unshuffled | box, shuffled | fused z r, shuffled vs unshuffled |', '|---|---|---|---|---|']
    for k, v in bv.VARIANTS.items():
        ck = f"{v['pair']}_{v['display']}" + (f"_{v['rung']}" if v['display'] == 'threshold' else ''); c = vc.get(ck)
        rule = (f"threshold, {v['rung']} rung" if v['display'] == 'threshold' else 'grey, no CT')
        if c:
            if v['display'] == 'threshold':
                ok = c['strip']['painted'] <= 0.01
                b.append(f"| {v['view']} | {names[v['pair']]}, {rule} | {100 * c['strip']['painted']:.2f} % painted, {100 * c['strip']['half_tone']:.2f} % at half tone | "
                         f"{100 * c['box']['painted']:.1f} % painted | {'pass' if ok else '**fail**'} (≤ 1 %) |")
            else:
                b.append(f"| {v['view']} | {names[v['pair']]}, {rule} | p99 tone {c['strip']['p99_tone']:.2f}, {100 * c['strip']['half_tone']:.2f} % darker than half tone | "
                         f"p99 tone {c['box']['p99_tone']:.2f}, {100 * c['box']['half_tone']:.1f} % darker than half tone | (no fixed bar) |")
        sh = sv.get(k)
        if sh:
            m = 'painted' if v['display'] == 'threshold' else 'half_tone'; lab = 'painted' if m == 'painted' else 'darker than half tone'
            s_.append(f"| {v['view']} | {names[v['pair']]}, {rule} | {100 * sh['original'][m]:.2f} % {lab}; p99 tone {sh['original']['p99_tone']:.2f} | "
                      f"{100 * sh['shuffled'][m]:.3f} % {lab}; p99 tone {sh['shuffled']['p99_tone']:.2f} | {sh['fused_z_r']} |")
    tests = [f"- {k[8:]} vs {v['vs']}: shuffled half-tone share ratio {v['ratio'] if v['ratio'] is not None else 'n/a (both 0)'}; bar ≤ {v['bar']}: "
             f"{'**pass**' if v['passes'] else '**fail**'}" for k, v in sv.items() if k.startswith('q4_test_')]
    shuffle = '\n'.join(s_) + ('\n\n' + '\n'.join(tests) if tests else '') if sv else '(pending: the v8Q run on the shuffled render)'
    k_ = ['| window | Reader v2 + v8Q, view 4 rule (own levels) | view 5 rule (own levels) | note |', '|---|---|---|---|']
    for n, r in kt.items():
        k_.append(f"| {n} | {100 * r['view4']['painted']:.1f} % painted, {100 * r['view4']['half_tone']:.1f} % at half tone | p99 tone {r['view5']['p99_tone']:.2f}, "
                  f"{100 * r['view5']['half_tone']:.1f} % darker than half tone | {r['label']} |")
    known = '\n'.join(k_) if kt else '(pending: the v8Q runs on the known-text windows)'
    return '\n'.join(b), shuffle, known


def main():
    T = sys.argv[1] if len(sys.argv) > 1 else ''
    # methodology, evidence, supplementary
    cp(f'{TX}/METHODOLOGY.md', f'{PKG}/methodology/METHODOLOGY.md')
    E = f'{PKG}/evidence'; os.makedirs(E, exist_ok=True)
    cp(f'{W}/x_refine/trials/I2_package/blank_strip.json', f'{E}/blank_strip_same_levels.json')
    TT = f'{W}/x_refine/trials/I2_package'
    for a_, b_ in (('shuffle_control.json', 'depth_shuffle_same_levels.json'), ('variant_controls.json', 'blank_strip_variants.json'),
                   ('shuffle_variants.json', 'depth_shuffle_variants.json'), ('known_text/known_text_panels.json', 'known_text_panels_v8Q.json')):
        if os.path.exists(f'{TT}/{a_}'): cp(f'{TT}/{a_}', f'{E}/{b_}')
    cp(f'{W}/x_refine/research/Q4_legibility_image/m6b_v8fixed.json', f'{E}/heldout_Rv2_v8in_metrics.json')
    for f in ('heldout_PHerc1447_metrics.json', 'heldout_PHerc1447_metrics_readerv2_trace.json', 'heldout_PHerc1447_w058_labels_vs_fused.jpg',
              'depth_shuffle_metrics.json', 'depth_shuffle_control.jpg'):
        cp(f'{V1}/evidence/{f}', f'{E}/{f}')
    if T and os.path.exists(f'{T}/compare_rt.json'):
        for f in ('compare_rt.json', 'compare_v8_box.json', 'compare_v8q_box.json', 'compare_variants.json', 'render_vs_P1.json', 'run_info.json',
                  'v8_box_exact.json', 'v8q_box_exact.json'):
            if os.path.exists(f'{T}/{f}'): cp(f'{T}/{f}', f'{E}/docker_test/{f}')
        for f in glob.glob(f'{T}/*.log'):
            t = open(f, errors='replace').read().replace(T, '/out')
            os.makedirs(f'{E}/docker_test', exist_ok=True); open(f'{E}/docker_test/{os.path.basename(f)}', 'w').write(t)
    rep = repro_text(T)
    bt, st_, kt_ = evidence_tables()
    open(f'{E}/README.md', 'w').write(open(f'{TX}/evidence_README.md').read().replace('{{REPRO}}', rep).replace('{{BLANK_TABLE}}', bt)
                                      .replace('{{SHUFFLE_TABLE}}', st_).replace('{{KNOWN_TABLE}}', kt_))
    S = f'{PKG}/supplementary'
    for a, b in SUPP_I1.items(): cp(f'{I1S}/{a}', f'{S}/{b}')
    cp(f'{TX}/supplementary_README.md', f'{S}/README.md')
    # code
    shutil.rmtree(f'{PKG}/code', ignore_errors=True); C = f'{PKG}/code'
    cp(f'{V1}/code/LICENSE', f'{C}/LICENSE')
    for f in CODE:
        if os.path.exists(f'{HERE}/{f}'): cp(f'{HERE}/{f}', f'{C}/{f}')
    for f in MODELS: cp(f'{HERE}/models/v8Q/{f}', f'{C}/models/v8Q/{f}')
    for f in SURF: cp(f'{HERE}/surface/{f}', f'{C}/surface/{f}')
    for f in glob.glob(f'{HERE}/surface/inputs/*'): cp(f, f'{C}/surface/inputs/{os.path.basename(f)}')
    for f in ('__init__.py', 'modeling.py', 'inference.py'): cp(f'{W}/third_party/v8in/ink8um/{f}', f'{C}/third_party/v8in/ink8um/{f}')
    cp(f'{W}/third_party/v8in/config.json', f'{C}/third_party/v8in/config.json')
    cp(f'{V1D}/code/x_zoom/zoomlib.py', f'{C}/x_zoom/zoomlib.py'); cp(f'{V1D}/code/x_research2/speed/fastink.py', f'{C}/x_research2/speed/fastink.py')
    # docker
    shutil.rmtree(f'{PKG}/docker', ignore_errors=True); D = f'{PKG}/docker'
    for f in DOCK: cp(f'{HERE}/docker/{f}', f'{D}/{f}')
    for f in ('constraints.txt', 'requirements.txt', 'checkpoints.tsv'): cp(f'{V1D}/{f}', f'{D}/{f}')
    cp(f'{V1D}/vendor/libz.so.1.zlib-ng-2.3.3', f'{D}/vendor/libz.so.1.zlib-ng-2.3.3')
    dres = open(f'{HERE}/docker/RESULTS.md').read() if os.path.exists(f'{HERE}/docker/RESULTS.md') else '(pending)'
    open(f'{D}/README.md', 'w').write(open(f'{TX}/docker_README.md').read().replace('{{DOCKER_RESULTS}}', dres))
    # README (letter rows from letter_dimensions.json) and the form texts
    L = json.load(open(f'{PKG}/letter_dimensions.json'))['letters']; rows = []
    for r in L:
        if not r['measured']: rows.append(f"| {r['position_in_line']} | {r.get('letter', '')} | – | – | – | no ink at half tone |"); continue
        nb = [s for s, t in (('left', r['runs_into_left_neighbour']), ('right', r['runs_into_right_neighbour'])) if t]
        rows.append(f"| {r['position_in_line']} | {r.get('letter', '')} | {r['x_px']}, {r['y_px']} | {r['w_px']} × {r['h_px']} | {r['w_mm']} × {r['h_mm']} | "
                    f"{'; '.join(t for t in (('joined to its ' + ' and '.join(nb) + ' neighbour') if nb else '', r.get('note', '')) if t)} |")
    side = json.load(open(f'{PKG}/{MESH}.json')); lk = side['variant']['key']; vt = json.load(open(f'{TX}/variants.json'))[lk]
    ok = [r for r in L if r['measured']]
    summ = (f"At half tone the {len(ok)} letter boxes are {min(r['w_mm'] for r in ok):.1f}–{max(r['w_mm'] for r in ok):.1f} mm wide and "
            f"{min(r['h_mm'] for r in ok):.1f}–{max(r['h_mm'] for r in ok):.1f} mm tall; "
            f"{sum(1 for r in ok if r['runs_into_left_neighbour'] or r['runs_into_right_neighbour'])} of them join a neighbour at half tone.")
    txt = open(f'{TX}/README.md').read()
    for k_, v_ in (('{{LETTER_ROWS}}', '\n'.join(rows)), ('{{LETTER_SUMMARY}}', summ), ('{{LEAD_VIEW}}', side['variant']['view']),
                   ('{{LEAD_INK}}', vt['ink']), ('{{LEAD_RULE}}', vt['rule']), ('{{LEAD_BG}}', vt['bg']), ('{{VIEWS_TABLE}}', views_table(lk))):
        txt = txt.replace(k_, v_)
    open(f'{PKG}/README.md', 'w').write(txt)
    for f in ('OTHER_INFORMATION_form_text_owner_only.md', 'DRAFT_STATUS_owner_only.md'):
        if os.path.exists(f'{TX}/{f}'): cp(f'{TX}/{f}', f'{PKG}/{f}')
    # repo-relative paths in every JSON sidecar
    for j in glob.glob(f'{PKG}/**/*.json', recursive=True):
        t = open(j).read(); t2 = t.replace(W + '/', '')
        if t2 != t: open(j, 'w').write(t2)
    files = sorted(p for p in glob.glob(f'{PKG}/**/*', recursive=True) if os.path.isfile(p) and not p.endswith('SHA256SUMS'))
    open(f'{PKG}/SHA256SUMS', 'w').write(''.join(f'{sha(p)}  {os.path.relpath(p, PKG)}\n' for p in files))
    print('files', len(files), 'MB', round(sum(os.path.getsize(p) for p in files) / 1e6, 1))


if __name__ == '__main__':
    main()
