"""Display both held-out scores for every fit and both retained checkpoints."""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from objective_study import ROOT
from plot_objective_study import CASE_LABELS


def main():
    cases = json.loads((ROOT/'publication.json').read_text())['cases']
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), layout='constrained')
    colors = {'nll': 'C0', 'mmd': 'C1', 'hybrid': 'C2'}
    for ax, case in zip(axes.flat, cases, strict=True):
        folder = ROOT/case
        frozen = json.loads((folder/'frozen.json').read_text())
        rows = frozen['nll_selection']['candidates']
        minimum = min(r['validation_nll'] for r in rows)
        for name, row in zip(frozen['names'], rows, strict=True):
            run, checkpoint = name.rsplit('/', 1)
            config = json.loads((folder/run/'config.json').read_text())
            result = json.loads((folder/run/'result.json').read_text())
            step = result['best_step'] if checkpoint == 'nll' else result['kernel_step']
            history = next(r for r in result['history'] if r['step'] == step)
            x, y = row['validation_nll']-minimum, history['validation_mmd']
            ax.scatter(x, y, color=colors[config['objective']], marker='o' if checkpoint == 'nll' else 'x', s=40, alpha=.7)
            if name == frozen['selected_nll']:
                ax.scatter(x, y, marker='*', s=190, color='crimson', zorder=5)
            if name == frozen['selected_kernel']:
                ax.scatter(x, y, marker='D', s=115, facecolor='none', edgecolor='purple', linewidth=2, zorder=6)
        ax.set(title=CASE_LABELS[case], xlabel='Excess held-out NLL over best checkpoint ↓',
               ylabel='Held-out kernel discrepancy (MMD²) ↓', yscale='log')
        ax.set_xscale('symlog', linthresh=.001)
        # All excess losses are nonnegative; keep only a small amount of
        # negative padding so the star at zero stays inside the frame.
        ax.set_xlim(-.0002, max(.002, 1.3*max(r['validation_nll']-minimum for r in rows)))
        ax.grid(alpha=.25)
    handles = [Line2D([], [], color=c, marker='o', linestyle='', label=k.upper() if k != 'hybrid' else 'NLL + MMD²') for k, c in colors.items()]
    handles += [Line2D([], [], marker='*', color='crimson', linestyle='', markersize=12, label='NLL selection'),
                Line2D([], [], marker='D', color='purple', markerfacecolor='none', linestyle='', label='MMD selection')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .945), ncol=5)
    fig.suptitle('Every candidate: objectives, capacities, regularizers and checkpoint choices', y=.995, fontsize=16).set_in_layout(False)
    fig.get_layout_engine().set(rect=(0, .07, 1, .82))
    fig.text(.02, .01, 'Circles: minimum-NLL checkpoints. Crosses: minimum-kernel-score checkpoints. Stars/diamonds mark the two frozen one-SE selections.\n'
             'All fits start near uniform. Both axes use held-out observations; true latent densities do not enter this comparison.', fontsize=10)
    path = Path('figs/convergence/objective_candidates.png')
    fig.savefig(path, dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
