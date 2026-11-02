"""Check frozen Tucker provenance, exact-evaluation results and saved figures."""
import re
from pathlib import Path

import numpy as np
from PIL import Image
from run_tucker_gallery import CASES, current_folder, digest, read

from ftde.direct_training import write_record

ROOT = Path('artifacts/tucker_gallery')


def conversion_check(folder, selection):
    from ftde import load_tucker, squared_integral_tt, squared_integral_tucker
    from ftde.tucker_operations import tucker_to_tt
    g = load_tucker(folder/selection['selected']/'fit.npz')
    tt = tucker_to_tt(g)
    box = np.asarray(g.domain)
    points = np.random.default_rng(812).uniform(box[:, 0], box[:, 1], (16, len(box)))
    a, b = np.asarray(g(points)), np.asarray(tt(points))
    value_error = float(np.max(np.abs(a-b))/max(float(np.max(np.abs(a))), 1e-300))
    norm_error = abs(float(squared_integral_tt(tt)/squared_integral_tucker(g))-1)
    if not np.isfinite([value_error, norm_error]).all() or max(value_error, norm_error) > 1e-9:
        raise RuntimeError('Diagnostic TT conversion changed the fitted Tucker function')
    return {'relative_value_error': value_error, 'relative_normalization_error': norm_error}


def audit():
    expected, selections, conversions = set(), {}, {}
    if (ROOT/'current.json').exists():
        pointer = ROOT/'current_selection.json'
        if pointer.exists():
            source = read(pointer)
            if digest(source['path']) != source['sha256']:
                raise RuntimeError('Published selection record changed')
            capacity = read(source['path'])
        else:
            capacity = read('artifacts/tucker_capacity/selection.json')
        if read(ROOT/'current.json') != capacity['selected']:
            raise RuntimeError('Published sources disagree with capacity selection')
        for case, candidates in capacity['candidates'].items():
            rule = capacity.get('selection_rules', {}).get(case)
            if rule:
                import jax

                from ftde import TuckerTrainingProblem, load_tucker
                from ftde.noisy_model_selection import select_by_paired_noisy_nll
                if rule['rule'] != 'paired_one_standard_error':
                    raise RuntimeError('Unknown Tucker selection rule')
                logs, roughness = [], []
                for row in candidates:
                    folder = Path(row['folder'])
                    fit = folder/read(folder/'selection.json')['selected']
                    logs.append(np.load(fit/'validation.npy'))
                    problem = TuckerTrainingProblem(load_tucker(fit/'fit.npz'), strength=0., marginal_strength=1.)
                    roughness.append(float(jax.jit(problem.marginal_penalty)(problem.params)))
                    jax.clear_caches()
                chosen, details = select_by_paired_noisy_nll(np.stack(logs), roughness)
                if candidates[chosen]['folder'] != capacity['selected'][case] or details['selected_index'] != rule['details']['selected_index']:
                    raise RuntimeError('Tucker selection disagrees with paired one-SE rule')
                np.testing.assert_allclose(roughness, [r['roughness'] for r in rule['details']['candidates']], rtol=1e-10)
            elif min(candidates, key=lambda row: row['validation_nll'])['folder'] != capacity['selected'][case]:
                raise RuntimeError('Capacity selection does not minimize held-out NLL')
            for row in candidates:
                if digest(Path(row['folder'])/'selection.json') != row['selection_sha256']:
                    raise RuntimeError('A capacity candidate changed after selection')
    summary = {r['case']: r for r in read(ROOT/'summary.json')['cases']}
    for family, dim, rank in CASES:
        for noise in ['free', 'mixture']:
            case = f'{family}{dim}_{noise}'
            folder = current_folder(case)
            p, s, e = (read(folder/f'{name}.json') for name in ['protocol', 'selection', 'evaluation'])
            if (folder == ROOT/case and p['rank'] != rank) or e['selection_sha256'] != digest(folder/'selection.json'):
                raise RuntimeError('Stale protocol/evaluation')
            for path, sha in s['files'].items():
                if digest(path) != sha:
                    raise RuntimeError(f'Frozen file changed: {path}')
            selected = s['selected']
            if selected != min(s['scores'], key=s['scores'].get):
                raise RuntimeError('Selection does not minimize validation NLL')
            if summary[case]['selected'] != selected or not np.isfinite(e['relative_l2']) or abs(summary[case]['relative_l2']-e['relative_l2']) > 1e-12:
                raise RuntimeError('Summary and evaluation disagree')
            if noise != 'free' and read(folder/'convolution_check.json')['selection_sha256'] != digest(folder/'selection.json'):
                raise RuntimeError('Stale convolution check')
            checks = read(folder/'plot_checks.json')
            if any(not np.isfinite(row['latent_pair_relative_l2']) for row in checks.values()):
                raise RuntimeError('Invalid marginal diagnostic')
            directory = 'ginzburg_landau' if family == 'gl' else 'swirl'
            prefix = f'tucker_{"noise_free" if noise == "free" else "noisy"}_dim{dim}'
            expected.update(f'{directory}/{prefix}_{suffix}.png' for suffix in ['marg_01', 'marg_23', 'convergence'])
            selections[case] = digest(folder/'selection.json')
            conversions[case] = conversion_check(folder, s)
    cryo = ROOT/'cryoem4'
    s = read(cryo/'selection.json')
    for path, sha in s['files'].items():
        if digest(path) != sha:
            raise RuntimeError(f'Frozen cryo-EM file changed: {path}')
    selections['cryoem4'] = digest(cryo/'selection.json')
    conversions['cryoem4'] = conversion_check(cryo, s)
    if read(cryo/'convolution_check.json')['selection_sha256'] != selections['cryoem4']:
        raise RuntimeError('Stale cryo-EM convolution check')
    expected.update(['cryoem/tucker_dim4.png', 'cryoem/tucker_training.png', 'convergence/tucker_dimensional_errors.png'])
    actual = {str(p.relative_to('figs')) for p in Path('figs').rglob('tucker_*.png')}
    if actual != expected:
        raise RuntimeError(f'Missing/unexpected Tucker figures: {actual ^ expected}')
    for name in expected:
        with Image.open(Path('figs')/name) as image:
            image.verify()
    for target in re.findall(r'\]\(([^)]+)\)', Path('figs/TUCKER.md').read_text()):
        if not (Path('figs')/target).exists():
            raise RuntimeError(f'Broken Tucker link: {target}')
    record = {'case_selections': selections, 'diagnostic_conversion_checks': conversions, 'png_count': len(expected),
              'images': {name: digest(Path('figs')/name) for name in sorted(expected)},
              'summary_sha256': digest(ROOT/'summary.json'), 'index_sha256': digest('figs/TUCKER.md')}
    write_record(ROOT/'publication.json', record)
    print('TUCKER PUBLICATION AUDITED', len(selections), 'cases,', len(expected), 'PNGs')


if __name__ == '__main__':
    audit()
