"""Publish cross-study results from choices frozen before truth evaluation."""
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from evaluate_noise_diagnosis import paired_se
from finish_noisy_followup import CALIBRATED_ROOT, ROOTS
from run_marginal_regularization import CASES_RUN, read

from ftde.direct_training import write_record


def summarize():
    frozen = read(CALIBRATED_ROOT/'overall_selection.json')
    for path, digest in frozen['sources'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    summary = {}
    for case in CASES_RUN:
        info = frozen['cases'][case]
        baseline_logs = np.load(ROOTS['regularization']/case/'test_logs_original.npy')
        curves = {s: np.load(root/case/'univariate.npz') for s, root in ROOTS.items()}
        evaluated = {s: read(root/case/'evaluation.json') for s, root in ROOTS.items()}
        records = {}
        for label, model in info['models'].items():
            study, name = model['study'], model['name']
            metric = evaluated[study]['models'][name]
            delta = baseline_logs-np.load(ROOTS[study]/case/f'test_logs_{name}.npy')
            records[label] = {**model, **metric, 'test_nll_minus_old_control': float(delta.mean()),
                               'paired_test_se_vs_old_control': paired_se(delta, case)}
        best_marginal = info['best_marginal']
        choices = {'old_control': 'regularization:original', 'minimum_nll': info['primary'],
                   'smooth_choice': info['smooth'], 'best_marginal_recipe': best_marginal}
        summary[case] = {'choices': choices, 'chosen_metrics': {k: records[v] for k, v in choices.items()},
                         'all_models': records, 'selection_used_truth': False}
        labels = list(dict.fromkeys(choices.values()))
        legend = {v: ', '.join(k.replace('_', ' ') for k, x in choices.items() if x == v) for v in labels}
        truth = curves['regularization']['truth']
        grid = curves['regularization']['grid']
        dim = len(truth)
        fig, axes = plt.subplots(2, (dim+1)//2, figsize=(13, 7), layout='constrained')
        for k, ax in enumerate(axes.flat):
            ax.plot(grid, truth[k], color='black', lw=2, label='Truth (evaluation only)')
            for c, label in enumerate(labels):
                model = info['models'][label]
                ax.plot(grid, curves[model['study']][model['name']][k], color=f'C{c}', lw=1.4, label=legend[label])
            ax.set(title=f'Coordinate {k}', xlabel='x', ylabel='Latent marginal density', xlim=(-2.6, 2.6))
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(f'{case}: final comparisons selected from noisy validation observations')
        folder = 'swirl' if case.startswith('swirl') else 'ginzburg_landau'
        fig.savefig(f'figs/{folder}/noisy_recovery_1d.png', dpi=160)
        plt.close(fig)
    write_record(CALIBRATED_ROOT/'summary.json', summary)
    for case, values in summary.items():
        print(case, {k: {f: x[f] for f in ['relative_l2', 'mean_univariate_relative_l2', 'marginal_curvature',
              'validation_nll', 'test_nll_minus_old_control', 'paired_test_se_vs_old_control']}
              for k, x in values['chosen_metrics'].items()}, flush=True)


if __name__ == '__main__':
    summarize()
