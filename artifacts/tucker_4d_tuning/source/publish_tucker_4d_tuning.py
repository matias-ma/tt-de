"""Evaluate the frozen 4D tuning winners and refresh companion Tucker figures."""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from check_tucker_convolution import check
from run_tucker_gallery import digest, read
from tune_tucker_4d import BASE, ROOT, protocol_path, setup

from ftde.direct_training import write_record


def selected_fit(folder):
    return folder/read(folder/'selection.json')['selected']


def main():
    study = setup()
    path = ROOT/'selection.json'
    selection = read(path)
    if (selection['tuning_protocol_sha256'] != digest(protocol_path()) or
            selection['refinements_sha256'] != digest(ROOT/'refinements.json')):
        raise RuntimeError('Tuning protocol changed after selection')
    if read(BASE/'current.json') not in (study['prior_sources'], selection['selected']):
        raise RuntimeError('Published sources changed during this study')
    completed = {}
    for name, spec in (study['candidates'] | read(ROOT/'refinements.json')).items():
        folder = ROOT/name/f'swirl4_{spec["noise"]}'
        frozen = read(folder/'selection.json')
        for filename, sha in frozen['files'].items():
            if digest(filename) != sha:
                raise RuntimeError(f'Frozen candidate changed: {filename}')
        result = read(selected_fit(folder)/'result.json')
        logs = np.load(selected_fit(folder)/'validation.npy')
        if result['history'][-1]['step'] != spec['steps'] or abs(float(logs.mean())+result['best_nll']) > 1e-12:
            raise RuntimeError('Incomplete training or inconsistent validation score')
        if spec.get('parent') and digest(spec['parent']) != spec['parent_sha256']:
            raise RuntimeError('Warm-start checkpoint changed')
        completed[name] = {'steps': result['steps'], 'best_nll': result['best_nll'],
                           'selection_sha256': digest(folder/'selection.json')}
    write_record(ROOT/'validation.json', {'completed_candidates': completed,
                                          'tuning_protocol_sha256': digest(protocol_path())})
    rows = []
    equivalence = {}
    for noise in ['free', 'mixture']:
        case = f'swirl4_{noise}'
        source = Path(selection['selected'][case])
        previous = Path(study['prior_sources'][case])
        tt = Path('artifacts/dimension_gallery_v9')/case
        p, old, ttp = (read(folder/'protocol.json') for folder in [source, previous, tt])
        for key in ['data', 'data_sha256', 'domain', 'kernel', 'noise_weights', 'noise_covariances']:
            if not (p[key] == old[key] == ttp[key]):
                raise RuntimeError(f'Comparison changed {case}: {key}')
        if not (p['modes'] == old['modes'] == read(selected_fit(tt)/'config.json')['modes']):
            raise RuntimeError('Polynomial basis changed')
        equivalence[case] = {'data_sha256': digest(p['data']), 'domain': p['domain'],
                                 'kernel_sha256': digest(p['kernel']) if p['kernel'] else None}
        subprocess.run([sys.executable, 'scripts/plot_tucker_gallery.py', str(source)], check=True)
        if noise == 'mixture':
            check(source)
        fit, ttfit = selected_fit(source), selected_fit(tt)
        result, config = read(fit/'result.json'), read(fit/'config.json')
        logs, ttlogs = np.load(fit/'validation.npy'), np.load(ttfit/'validation.npy')
        if logs.shape != ttlogs.shape:
            raise RuntimeError('Validation vectors are not aligned')
        delta = ttlogs-logs  # Tucker NLL minus TT NLL, paired on observations.
        tte = read(tt/'evaluation.json')['models'][ttfit.name]
        e = read(source/'evaluation.json')
        rows.append({'condition': 'clean' if noise == 'free' else 'noisy', 'case': case, 'source': str(source),
                         'rank': p['rank'], 'strength': config['strength'],
                         'marginal_strength': config.get('marginal_strength', 0.),
                         'tucker_l2': e['relative_l2'], 'prior_tucker_l2': read(previous/'evaluation.json')['relative_l2'],
                         'tt_l2': tte['relative_l2'], 'tucker_nll': result['best_nll'], 'tt_nll': -float(ttlogs.mean()),
                         'paired_nll_difference': float(delta.mean()), 'paired_nll_standard_error': float(delta.std(ddof=1)/np.sqrt(len(delta))),
                         'selected_step': result['best_step'], 'final_stage_steps': result['steps']})
        prefix = f'tucker_{"noise_free" if noise == "free" else "noisy"}_dim4'
        for suffix in ['marg_01', 'marg_23', 'convergence']:
            name = f'{prefix}_{suffix}.png'
            destination = Path('figs/swirl')/name
            archive = ROOT/'previous_figs'/name
            archive.parent.mkdir(exist_ok=True)
            if not archive.exists():
                shutil.copy2(destination, archive)
            staged = source.parent/'figs/swirl'/name
            if staged.resolve() != destination.resolve():
                shutil.copy2(staged, destination)
    controls = {}
    for case, name in selection.get('minimum_nll_control', {}).items():
        folder = Path(name)
        if name != selection['selected'][case]:
            subprocess.run([sys.executable, 'scripts/plot_tucker_gallery.py', name], check=True)
        controls[case] = {'source': name, 'relative_l2': read(folder/'evaluation.json')['relative_l2'],
                          'validation_nll': read(selected_fit(folder)/'result.json')['best_nll']}
    write_record(ROOT/'comparison.json', {'rows': rows, 'minimum_nll_controls': controls, 'selection_sha256': digest(path),
                                             'paired_nll_standard_error_is_selection_unadjusted': True})
    write_record(ROOT/'data_equivalence.json', equivalence)
    for name in ['summary.json', 'publication.json', 'current.json']:
        target = ROOT/f'previous_{name}'
        if not target.exists():
            shutil.copy2(BASE/name, target)
    write_record(BASE/'current.json', selection['selected'])
    write_record(BASE/'current_selection.json', {'path': str(path), 'sha256': digest(path), 'report': str(ROOT/'report.md')})
    lines = ['# 4D Swirl Tucker tuning', '',
             'Higher ranks, a Hessian-penalty sweep, longer noisy training, and larger-batch refinement were tested on the same observations, domain, polynomial basis and prescribed noise kernel as TT. TT remains the default and its fits and figures are unchanged.', '',
             'The original experiment plan was saved before fitting. Three clean fits use rank 32, 20,000 updates and Hessian strengths 0.003, 0.01 and 0.03. After measuring noisy rank-24 runtime, the noisy plan was amended before truth evaluation: three fits now start from an exact rank expansion of the published rank-16 Tucker fit, with 4,000 additional updates at strengths 0.00001, 0.0001 and 0.001. All use 49 modes and the log-penalty objective. The interrupted noisy scratch run remains archived and is not a candidate.', '',
             'For each condition, the best held-out fit (including the prior published Tucker fit) then receives lower-learning-rate refinement: 5,000 clean updates with batch 4,096; three independent noisy branches of 2,000 updates with batch 256 and marginal-curvature strengths 0, 0.0003 and 0.001. Clean selection minimizes held-out NLL. The main noisy selection uses the same rule as TT: minimum marginal curvature among candidates within one paired standard error of the best observation NLL. The strict minimum-NLL choice is also frozen and reported below. This selection amendment was saved before evaluating any new fits against truth. Baseline truth errors were known before this development follow-up.', '',
             '| Condition | TT relative L2 | Previous Tucker | Tuned Tucker | Tuned rank |',
             '|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["condition"]} | {row["tt_l2"]:.6f} | {row["prior_tucker_l2"]:.6f} | {row["tucker_l2"]:.6f} | {row["rank"]} |')
    for case, control in controls.items():
        lines += ['', f'The noisy strict-minimum-NLL control has relative L2 {control["relative_l2"]:.6f} and held-out NLL {control["validation_nll"]:.6f}, from `{control["source"]}`. It is reported as a control, not selected by its truth error.']
    lines += ['', 'L2 values are unsquared full-dimensional latent relative errors, not errors of projected marginals. The 4D Tucker self-integral uses degree-exact blocked quadrature; the cross integral is checked by doubling path quadrature from 4,000 to 8,000 points.', '',
              '| Condition | TT held-out NLL | Tucker held-out NLL | Tucker − TT | Paired SE |',
              '|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["condition"]} | {row["tt_nll"]:.6f} | {row["tucker_nll"]:.6f} | {row["paired_nll_difference"]:+.6f} | {row["paired_nll_standard_error"]:.6f} |')
    lines += ['', 'Lower values are better. Paired standard errors describe observation-wise likelihood differences on the same 4,096 validation observations; they are not adjusted for model selection and are not an independent test-set significance claim. This compares the newly tuned Tucker fits with the existing TT gallery, not equal tuning budgets or parameter counts. Noisy observation likelihood need not order latent-density error identically.', '',
              '| New candidate | Rank | Hessian strength | Marginal strength | Best held-out NLL |',
              '|---|---:|---:|---:|---:|']
    for name, spec in (study['candidates'] | read(ROOT/'refinements.json')).items():
        folder = ROOT/name/f'swirl4_{spec["noise"]}'
        result = read(selected_fit(folder)/'result.json')
        lines.append(f'| {name} | {spec["rank"]} | {spec["strength"]:g} | {spec.get("marginal_strength", 0.):g} | {result["best_nll"]:.6f} |')
    lines += ['', 'Selected checkpoints:', '']
    for row in rows:
        lines.append(f'- {row["condition"]}: `{row["source"]}`, checkpoint {row["selected_step"]}/{row["final_stage_steps"]} in its final training stage.')
    lines += ['', 'Exactly separable noise-mixture blocks now use successive factor-sized actions instead of a dense pair matrix. The real-kernel benchmark agrees with the previous calculation to about 1e-15 in gradients and measures about a 1.25× value-and-gradient speedup at rank 24 on the allocated A100. Rank expansion zero-pads the old core and adds orthogonal factor directions; it preserves the initial represented function. These are computational changes, not convolution approximations.', '',
              '[Current figures](../../figs/TUCKER.md) · [Previous 4D figures](previous_figs) · [Frozen selection](selection.json) · [Machine-readable comparison](comparison.json)', '',
              'Reproduce with `scripts/tune_tucker_4d.py`, then `scripts/publish_tucker_4d_tuning.py`, using `scripts/run_noise_diagnosis_gpu.sh` on an allocated GPU.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    subprocess.run([sys.executable, 'scripts/summarize_tucker_gallery.py'], check=True)
    subprocess.run([sys.executable, 'scripts/audit_tucker_gallery.py'], check=True)
    subprocess.run([sys.executable, 'scripts/audit_dimension_gallery.py', '--publication'], check=True)
    write_record(ROOT/'publication.json', {'selection_sha256': digest(path), 'report_sha256': digest(ROOT/'report.md'),
                                              'gallery_manifest_sha256': digest(BASE/'publication.json')})
    print('PUBLISHED', rows, flush=True)


if __name__ == '__main__':
    main()
