"""Render frozen cryo-EM direct-training comparisons; no clean truth or labels."""

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('artifacts/direct_optimization')
FIGS = Path('figs/cryoem')
CASES = ('dim4', 'path_dim4', 'dim10', 'dim20')
LABELS = {'dim4': 'Cryo-EM 4D', 'path_dim4': 'Cryo-EM path 4D',
          'dim10': 'Cryo-EM 10D', 'dim20': 'Cryo-EM 20D'}


def plot_case(case):
    root = ROOT/f'cryoem_{case}'
    selection = json.loads((root/'selection.json').read_text())
    names = selection['names']
    historical_rank = {'dim4': 32, 'path_dim4': 32, 'dim10': 64, 'dim20': 48}[case]
    labels = [f'Historical fit\nmaximum rank {historical_rank}', 'Near-uniform → Adam\nλ = 0.0001',
              'Near-uniform → Adam\nλ = 0.001', 'Selected Adam → full-data L-BFGS']
    fig, axes = plt.subplots(2, 4, figsize=(16, 9.8), layout='constrained')
    for row, pair in enumerate(('01', '23')):
        saved = [np.load(root/name/'marginals.npz') for name in names]
        vmax = max(float(m['pair'+pair].max()) for m in saved)
        for col, (name, label, marginal) in enumerate(zip(names, labels, saved, strict=True)):
            ax = axes[row, col]
            image = ax.pcolormesh(marginal['xs'+pair], marginal['ys'+pair], marginal['pair'+pair],
                                 cmap='magma', vmin=0, vmax=vmax, shading='auto')
            if row == 0:
                title = label+f"\nValidation NLL = {selection['validation_nll'][col]:.4f}"
                if name == selection['new_selected']:
                    title += '\nSelected new fit'
                ax.set_title(title, fontsize=10)
            ax.set(xlabel=f'x{pair[0]}', ylabel=f'x{pair[1]}', aspect='equal')
        fig.colorbar(image, ax=axes[row], shrink=.75, label='Deconvolved marginal density; common row scale')
    fig.suptitle(LABELS[case]+': direct Chebyshev TT optimization', fontsize=16)
    fig.get_layout_engine().set(rect=(0, .08, 1, .87))
    chosen_label = labels[names.index(selection['selected'])].replace('\n', '; ')
    fig.text(.015, .018, 'New fits: elementary near-uniform start, 49 modes, maximum rank 32, 4,000 Adam updates; selected Adam adds 40 full-data L-BFGS updates.\n'
             'Same observations and prescribed noise as the historical fit; no learned signal pilot. Historical recipe and training budget differ.\n'
             'Development-validation selection (one SE, then lower H²): '+chosen_label+'. Unknown clean-density accuracy is not measured.', fontsize=9)
    fig.savefig(FIGS/f'direct_optimization_{case}.png', dpi=160)
    plt.close(fig)


def plot_training():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout='constrained')
    for case, ax in zip(CASES, axes.flat, strict=True):
        root = ROOT/f'cryoem_{case}'
        selection = json.loads((root/'selection.json').read_text())
        for strength, color in ((.0001, 'tab:green'), (.001, 'tab:blue')):
            name = f'constant_h2_{strength:g}'
            result = json.loads((root/name/'result.json').read_text())
            h = result['history']
            ax.plot([v['step'] for v in h], [v['nll'] for v in h], '.-', color=color, label=f'Adam, λ={strength:g}')
            best = next(v for v in h if v['step'] == result['best_step'])
            ax.scatter(best['step'], best['nll'], color=color, marker='*', s=100, zorder=4)
        ax.axhline(selection['validation_nll'][0], color='gray', ls='--', label='Historical fit')
        refined = json.loads((root/'lbfgs/result.json').read_text())
        h = refined['history']
        inset = ax.inset_axes([.47, .38, .49, .35])
        inset.plot([v['step'] for v in h], [v['nll'] for v in h], '.-', color='tab:purple', ms=3)
        best = next(v for v in h if v['step'] == refined['best_step'])
        inset.scatter(best['step'], best['nll'], color='tab:purple', marker='*', s=80, zorder=4)
        inset.set_title('Full-data L-BFGS continuation', fontsize=8)
        inset.set_xlabel('L-BFGS updates', fontsize=7)
        inset.tick_params(labelsize=7)
        inset.ticklabel_format(axis='y', style='plain', useOffset=False)
        inset.grid(alpha=.2)
        ax.set(title=LABELS[case], xlabel='Adam updates from the near-uniform start', ylabel='Development-validation NLL ↓')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle('Cryo-EM: density structure learned through TT optimization', fontsize=16)
    fig.get_layout_engine().set(rect=(0, .06, 1, .89))
    fig.text(.015, .018, 'Stars identify retained checkpoints. L-BFGS begins at the selected Adam checkpoint; its updates use every training observation.\n'
             'Curves are development validation, reused from earlier studies. No independent clean-density improvement is claimed.', fontsize=10)
    fig.savefig(FIGS/'direct_optimization_training.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    for case in CASES:
        plot_case(case)
    plot_training()
