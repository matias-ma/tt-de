"""Fixed-recipe, three-seed robustness; this does not reselect the main fit."""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('artifacts/noise_free_v2')


if __name__ == '__main__':
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    summary = {}
    for col, case in enumerate(('swirl4', 'swirl10')):
        folder = ROOT/case
        frozen = json.loads((folder/'frozen.json').read_text())
        adam = [json.loads((folder/name/'result.json').read_text()) for name in ['baseline', 'seed_1', 'seed_2']]
        core = [json.loads((folder/frozen['selected']/'result.json').read_text())]
        core += [json.loads((ROOT/'initialization_check'/f'{case}_seed{s}'/'result.json').read_text()) for s in [1, 2]]
        for records, label, color in [(adam, 'Adam + original axis curvature', 'tab:blue'),
                                       (core, 'Core sweeps + selected full Hessian', 'tab:orange')]:
            axes[0, col].plot([0, 1, 2], [r['best_nll'] for r in records], 'o-', color=color, label=label)
        axes[0, col].set(title=f'{case}: same elementary initializer, different seeds', xlabel='Initialization seed', ylabel='Retained validation NLL ↓', xticks=[0, 1, 2])
        axes[0, col].legend(fontsize=8)
        for seed, record in enumerate(core):
            h = record['history']
            axes[1, col].plot([v['step'] for v in h[1:]], [v['validation_nll'] for v in h[1:]], '.-', label=f'Seed {seed}')
        axes[1, col].set(title='Core-sweep learning after the first sweep', xlabel='Full-data core sweeps', ylabel='Validation NLL ↓')
        axes[1, col].legend(fontsize=8)
        summary[case] = {label: {'nll': [r['best_nll'] for r in records], 'range': float(np.ptp([r['best_nll'] for r in records]))}
                         for label, records in [('adam', adam), ('core_sweeps', core)]}
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle('Elementary initialization: robustness of the fixed optimization recipes', fontsize=15)
    fig.get_layout_engine().set(rect=(0, .09, 1, .84))
    fig.text(.015, .015, 'Initializer: constant + 0.01 smooth random perturbation; 49 Chebyshev modes, rank 32. No observation enters initialization.\n'
             'Adam minibatch seed is fixed. The sweep recipe was frozen before this seed check; these runs do not reselect the published fit.\n'
             'Optimizer and regularizer differ between the two recipes. Three seeds establish limited empirical robustness, not seed independence.', fontsize=9)
    fig.savefig('figs/convergence/initialization.png', dpi=160)
    plt.close(fig)
    (ROOT/'initialization_check/summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
