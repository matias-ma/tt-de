"""Matched capacity controls at fixed optimizer, budget and regularizer."""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from objective_study import ROOT


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout='constrained')
    for row, case in enumerate(['gl6_free', 'swirl4_free']):
        folder = ROOT/case
        names = json.loads((folder/'capacity_selection.json').read_text())['names']
        evaluation = json.loads((folder/'evaluation.json').read_text())['models']
        generalization = json.loads((folder/'generalization.json').read_text())['models']
        for modes in [17, 25, 49]:
            records = []
            for name in names:
                config = json.loads((folder/name/'config.json').read_text())
                if config['modes'] != modes:
                    continue
                result = json.loads((folder/name/'result.json').read_text())
                selected = next(h for h in result['history'] if h['step'] == result['best_step'])
                records.append((config['rank'], selected['validation_nll'], selected['validation_mmd'],
                                evaluation[name+'/nll']['relative_l2'], generalization[name]['training_nll']))
            if not records:
                continue
            records = np.array(sorted(records))
            for col in range(3):
                axes[row, col].plot(records[:, 0], records[:, col+1], 'o-', label=f'{modes} modes', color=f'C{[17, 25, 49].index(modes)}')
                axes[row, col].set(xlabel='Requested maximum TT rank', title='Ginzburg–Landau, 6D' if case.startswith('gl') else 'Swirl, 4D', xticks=[5, 10, 20, 32])
            axes[row, 0].plot(records[:, 0], records[:, 4], 's--', markersize=4,
                              color=f'C{[17, 25, 49].index(modes)}')
        for ax, ylabel in zip(axes[row], ['NLL ↓ (solid: validation; dashed: training)', 'Held-out kernel discrepancy (MMD²) ↓', 'Full latent relative L2 ↓ (evaluation only)'], strict=True):
            ax.set_ylabel(ylabel)
            ax.grid(alpha=.25)
            ax.legend(fontsize=9)
    fig.suptitle('Capacity matters: rank and mode count at fixed NLL objective', fontsize=16, y=.985).set_in_layout(False)
    fig.get_layout_engine().set(rect=(0, .07, 1, .86))
    fig.text(.025, .01, 'All controls: near-uniform initialization; 4,000 Adam updates; full-Hessian λ=0.001; same observations and box.\n'
             'Solid curves: validation; dashed curves in the first column: training. Each point uses its minimum-validation-NLL checkpoint. Truth is evaluated afterward.', fontsize=10)
    destination = Path('figs/convergence/capacity.png')
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
