"""Display matched optimization changes and post-selection latent recovery."""
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from run_marginal_regularization import CASES_RUN, read
from run_training_geometry import ROOT

LABELS = {'original': 'Baseline, seed 0', 'precondition_seed0': 'Stronger whitening',
          'curriculum_seed0': 'Coarse to fine', 'combined_seed0': 'Both changes',
          'baseline_seed1': 'Baseline, seed 1', 'precondition_seed1': 'Stronger whitening, seed 1',
          'curriculum_seed1': 'Coarse to fine, seed 1', 'combined_seed1': 'Both changes, seed 1'}


def plot():
    import plot_marginal_regularization as common
    for folder in ['swirl', 'ginzburg_landau', 'convergence']:
        Path('figs', folder).mkdir(parents=True, exist_ok=True)
    common.ROOT, common.STEM = ROOT, 'training_geometry'
    common.model_label = lambda n: LABELS[n]
    common.chosen_models = lambda _: ['original', 'precondition_seed0', 'curriculum_seed0', 'combined_seed0']
    common.plot()
    frozen = read(ROOT/'frozen.json')
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), layout='constrained')
    for row, case in enumerate(CASES_RUN):
        info = frozen['cases'][case]
        best = min(m['validation_nll'] for m in info['models'].values())
        for name, model in info['models'].items():
            history = model['result']['history']
            step = np.array([h['step'] for h in history])
            nll = np.array([h['validation_nll'] for h in history])
            rough = np.array([h['roughness_hessian'] for h in history])
            style = '--' if name.endswith('seed1') else '-'
            label = LABELS[name]
            axes[row, 0].plot(step, nll, style, label=label)
            axes[row, 1].semilogy(step, rough, style)
            config = model['config']
            schedule = config.get('mode_schedule') or [(0, 49 if case.startswith('swirl') else 25)]
            if name in ['original', 'curriculum_seed0']:
                x, y = zip(*schedule, strict=True)
                axes[row, 2].step([*x, 4000], [*y, y[-1]], where='post', label=label)
        axes[row, 0].set(title=case+' — noisy validation', ylabel='NLL (zoom)', ylim=(best-.004, best+.06))
        axes[row, 0].legend(fontsize=8)
        axes[row, 1].set(title='Amplitude roughness', ylabel='Integrated Hessian energy')
        axes[row, 2].set(title='Trainable coefficients per axis', ylabel='Active modes')
        axes[row, 2].legend(fontsize=8)
        for ax in axes[row]:
            ax.axvline(1000, color='gray', ls=':', lw=1)
            ax.axvline(2000, color='gray', ls=':', lw=1)
            ax.axvspan(0, 2000, color='gray', alpha=.07)
            ax.set_xlabel('Adam updates; selection begins at 2000')
            ax.grid(alpha=.2)
    fig.suptitle('Fixed loss penalty and 4000-update budget; optimizer coordinates and mode activation vary')
    dest = Path('figs/convergence/training_geometry_training.png')
    fig.savefig(dest, dpi=160)
    plt.close(fig)
    plot_previous_comparison(frozen)


def plot_previous_comparison(frozen):
    """Place the new choices beside the previously frozen smooth choice."""
    import hashlib

    from ftde.direct_training import write_record

    prior_root = Path('artifacts/calibrated_marginals_v7')
    prior = read(prior_root/'overall_selection.json')
    for path, sha in prior['sources'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise RuntimeError(f'Previous selection changed: {path}')
    studies = {'regularization': Path('artifacts/marginal_regularization_v5'),
               'initialization': Path('artifacts/smooth_initialization_v6'), 'calibrated': prior_root}
    smooth = read(ROOT/'smooth_selection.json')['cases']
    summary = {}
    for case in CASES_RUN:
        old = prior['cases'][case]['models'][prior['cases'][case]['smooth']]
        old_root = studies[old['study']]
        curves = np.load(ROOT/case/'univariate.npz')
        old_curves = np.load(old_root/case/'univariate.npz')
        np.testing.assert_array_equal(curves['grid'], old_curves['grid'])
        selected = frozen['cases'][case]['selected']
        selected_smooth = smooth[case]['selected']
        names = list(dict.fromkeys([selected, selected_smooth]))
        evaluated = read(ROOT/case/'evaluation.json')['models']
        summary[case] = {'primary': selected, 'smooth_choice': selected_smooth,
            'previous_smooth_choice': old, 'previous_metrics': read(old_root/case/'evaluation.json')['models'][old['name']],
            'models': evaluated, 'selection_used_truth': False}
        dim = len(curves['truth'])
        fig, axes = plt.subplots(2, (dim+1)//2, figsize=(13, 7), layout='constrained')
        for k, ax in enumerate(axes.flat):
            ax.plot(curves['grid'], curves['truth'][k], 'k', lw=2, label='Truth (evaluation only)')
            ax.plot(curves['grid'], old_curves[old['name']][k], color='C0', ls='--', lw=1.6,
                    label='Previous smooth choice (different training recipe)')
            for c, name in enumerate(names, 1):
                label = LABELS[name]+(' [min NLL]' if name == selected else '')
                label += ' [smooth choice]' if name == selected_smooth else ''
                ax.plot(curves['grid'], curves[name][k], color=f'C{c}', lw=1.5, label=label)
            ax.set(title=f'Coordinate {k}', xlabel='x', ylabel='Latent marginal density', xlim=(-2.6, 2.6))
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(f'{case}: new observation-selected fits versus the previous smooth choice')
        dest = 'swirl' if case.startswith('swirl') else 'ginzburg_landau'
        fig.savefig(f'figs/{dest}/training_geometry_recovery_1d.png', dpi=160)
        plt.close(fig)
    write_record(ROOT/'summary.json', summary)


if __name__ == '__main__':
    plot()
