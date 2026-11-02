"""Show observation-only objective, regularizer, initialization and solver controls."""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

ROOT = Path('artifacts/noise_free_v2')
GROUPS = {
    'Objective': ['baseline', 'adam_logpen', 'conditional_0.1', 'conditional_1', 'conditional_10', 'quadratic_0.0001', 'quadratic_0.001'],
    'Regularizer (Adam)': ['baseline', 'h1_0.001', 'h1_0.01', 'h1_0.1', 'hessian_0.0001', 'hessian_0.001', 'hessian_0.01'],
    'Initialization (same minibatch seed)': ['baseline', 'seed_1', 'seed_2', 'perturbation_0.001', 'perturbation_0.1'],
    'Optimizer and budget': ['baseline', 'long_adam', 'unclipped_adam', 'lbfgs_flat', 'lbfgs_continue', 'lbfgs_hessian_continue',
                             'sweep_flat', 'sweep_hessian_flat_long', 'sweep_hessian_medium_flat_long', 'sweep_hessian_strong_flat_long',
                             'adam_h1_long_continue', 'adam_hessian_long_continue'],
}


def short(name):
    if name == 'unclipped_adam':
        return 'Adam: no gradient-norm clip'
    return name.replace('hessian_', 'Hess ').replace('conditional_', 'conditional ').replace('quadratic_', 'L2 ').replace('perturbation_', 'perturb ').replace('baseline', 'baseline / seed 0').replace('_', ' ')


if __name__ == '__main__':
    fig, axes = plt.subplots(2, 4, figsize=(21, 11), layout='constrained')
    for row, case in enumerate(('swirl4', 'swirl10')):
        folder = ROOT/case
        frozen = json.loads((folder/'frozen.json').read_text())
        for ax, (title, group) in zip(axes[row], GROUPS.items(), strict=True):
            names = [n for n in group if n in frozen['names']]
            records = [json.loads((folder/n/'result.json').read_text()) for n in names]
            values = [r['best_nll'] for r in records]
            baseline = json.loads((folder/'baseline/result.json').read_text())['best_nll']
            ax.barh(range(len(names)), np.asarray(values)-baseline,
                    color=['tab:red' if n == frozen['selected'] else 'tab:blue' for n in names])
            ax.set_yticks(range(len(names)), [short(n) for n in names], fontsize=7)
            ax.invert_yaxis()
            ax.axvline(0, color='gray', lw=.8)
            ax.set(title=f'{case}: {title}', xlabel='Validation NLL minus baseline ↓')
            ax.grid(axis='x', alpha=.2)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    fig.suptitle('Noise-free swirl: direct coefficient-optimization controls', fontsize=17)
    fig.get_layout_engine().set(rect=(0, .08, 1, .84))
    fig.text(.012, .013, 'All runs: one Chebyshev TT, 49 modes/axis, maximum rank 32, elementary initialization. Bars use each run’s retained validation checkpoint.\n'
             'Default Adam: 4,000 updates; long Adam: 8,000; Adam continuation: 12,000 additional updates. L-BFGS: 300 full-data updates. Core sweeps: 6 or 20.\n'
             'Core sweeps use NLL + log(1 + λR); other runs use NLL + λR unless marked otherwise. The Adam logpen control uses the logarithmic penalty. Red: selected recipe.', fontsize=9)
    Path('figs/convergence').mkdir(exist_ok=True)
    fig.savefig('figs/convergence/noise_free_optimization.png', dpi=160)
    plt.close(fig)
