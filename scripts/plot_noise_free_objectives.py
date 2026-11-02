"""Compare objectives under both likelihood and post-fit density error."""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('artifacts/noise_free_v2')
LABELS = {'baseline': 'NLL', 'adam_logpen': 'NLL + log penalty', 'conditional_0.1': 'NLL + 0.1 conditional',
          'conditional_1': 'NLL + 1 conditional', 'conditional_10': 'NLL + 10 conditional',
          'quadratic_0.0001': 'L² loss, λ=10⁻⁴', 'quadratic_0.001': 'L² loss, λ=10⁻³'}


if __name__ == '__main__':
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout='constrained')
    for col, case in enumerate(('swirl4', 'swirl10')):
        folder = ROOT/case
        frozen = json.loads((folder/'frozen.json').read_text())
        selected = frozen['selected']
        names = [n for n in LABELS if n in frozen['names']]+[selected]
        records = [json.loads((folder/n/'result.json').read_text()) for n in names]
        evaluation = [json.loads((folder/n/'evaluation.json').read_text()) for n in names]
        labels = [LABELS.get(n, 'Selected core sweeps') for n in names]
        nll = np.asarray([r['best_nll'] for r in records])
        colors = ['tab:red' if n == selected else 'tab:blue' for n in names]
        axes[0, col].bar(range(len(names)), nll-nll.min(), color=colors)
        axes[1, col].bar(range(len(names)), [r['relative_l2'] for r in evaluation], color=colors)
        for ax in axes[:, col]:
            ax.set_xticks(range(len(names)), labels, rotation=30, ha='right', fontsize=8)
            ax.grid(axis='y', alpha=.2)
        axes[0, col].set(title=f'{case}: held-out likelihood', ylabel='Excess validation NLL ↓')
        axes[1, col].set(title=f'{case}: final full-dimensional density error', ylabel='Relative L2 ↓')
    fig.suptitle('Alternative objectives: likelihood and density error can prefer different fits', fontsize=15)
    fig.get_layout_engine().set(rect=(0, .11, 1, .81))
    fig.text(.015, .02, 'Blue: 4,000 Adam updates, same 49-mode/rank-32 Chebyshev TT and elementary initializer. Red: selected 20-sweep core-training recipe.\n'
             'Blue NLL/composite fits use axis-curvature λ=10⁻⁴; log-penalty replaces λR with log(1+λR). The red recipe also changes regularization.\n'
             'The L² training score is ∫p² − 2 mean p(observations), with p = a² / ∫a²; the current exact implementation supports 4D.\n'
             'All fitted models and primary NLL choices were frozen before truth evaluation. Supplemental objective scoring does not reselect any model.', fontsize=9)
    fig.savefig('figs/convergence/objectives.png', dpi=160)
    plt.close(fig)
