"""Show completed, frozen noise-free optimization controls."""

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from plot_direct_optimization import label, training_trace

from ftde.swirl import swirl_pair_marginal

ROOT = Path('artifacts/direct_optimization/swirl_free4')
FIGS = Path('figs/swirl')


def short(name):
    if name == 'previous_squared':
        return 'Historical'
    n = name.split('_n')[1].split('_')[0]
    r = name.split('_r')[1].split('_')[0]
    strength = name.split('_l')[1].split('_')[0]
    seed = name.split('_s')[1].split('_')[0]
    return f'n={n}, r={r}, λ={strength}, seed={seed}'


if __name__ == '__main__':
    evaluation = json.loads((ROOT/'evaluation_free.json').read_text())
    results, shown = evaluation['models'], evaluation['displayed']
    names = ['uniform_start', 'previous_squared', *[shown[k] for k in ('adam', 'lbfgs', 'signed', 'penalized')]]
    data = np.load('artifacts/swirl/swirl_free_dim4/data.npz')['train']
    grid = np.linspace(-2.6, 2.6, 201)
    for i, j in [(0, 1), (2, 3)]:
        fig, axes = plt.subplots(2, 4, figsize=(16, 10), layout='constrained')
        truth = swirl_pair_marginal(4, i, j, grid, grid)
        vmax = float(truth.max())
        axes.flat[0].pcolormesh(grid, grid, truth, shading='auto', cmap='magma', vmin=0, vmax=vmax)
        axes.flat[0].set_title('True density\nEvaluation only')
        hist, xe, ye = np.histogram2d(data[:, i], data[:, j], bins=100, range=[[-2.6, 2.6]]*2, density=True)
        axes.flat[1].pcolormesh(xe, ye, hist.T, shading='auto', cmap='magma', vmin=0, vmax=vmax)
        axes.flat[1].set_title('Noise-free observations\n100,000 training samples')
        for ax, name in zip(list(axes.flat)[2:], names, strict=True):
            marginal = np.load(ROOT/name/'marginals_free.npz')
            image = ax.pcolormesh(marginal['nodes'][i], marginal['nodes'][j], marginal[f'pair{i}{j}'],
                                 shading='auto', cmap='magma', vmin=0, vmax=vmax)
            title = label(name)+f"\nFull 4D relative L2 = {results[name]['relative_l2']:.3f}"
            if name not in ('previous_squared', 'uniform_start'):
                title += ', λ='+name.split('_l')[1].split('_')[0]
            ax.set_title(title, fontsize=9)
        for ax in axes.flat:
            ax.set_facecolor(plt.colormaps['magma'](0.))
            ax.set(xlim=(-2.6, 2.6), ylim=(-2.6, 2.6), xlabel=f'x{i}', ylabel=f'x{j}', aspect='equal')
        for ax in axes[0]:
            ax.set_xlabel('')
        fig.colorbar(image, ax=axes, shrink=.75, extend='max', label='Marginal density; common scale')
        fig.suptitle('Noise-free swirl: direct TT learning from a near-uniform start', fontsize=16)
        fig.get_layout_engine().set(rect=(0, .08, 1, .87), hspace=.12)
        fig.text(.015, .018, 'No fitted signal initializer. Original Chebyshev functional TT throughout.\n'
                 'Signed fits train without clipping; the displayed joint density is truncated and renormalized only afterward.\n'
                 'Displayed recipes minimize held-out L2 risk within each method. Truth is used only after selection; budgets and capacities differ.', fontsize=9)
        fig.savefig(FIGS/f'direct_optimization_free_dim4_marg_{i}{j}.png', dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout='constrained')
    colors = {'adam': 'tab:green', 'lbfgs': 'tab:purple', 'signed': 'tab:blue', 'penalized': 'tab:orange'}
    for key in ('adam', 'lbfgs'):
        name = shown[key]
        xs, ys, chosen_x, chosen_y = training_trace(ROOT, name)
        axes[0, 0].plot(xs, ys, marker='.', color=colors[key], label=key+': '+short(name))
        axes[0, 0].scatter(chosen_x, chosen_y, marker='*', s=100, color=colors[key], zorder=3)
    axes[0, 0].set(title='Squared TT: optimization from the flat start', xlabel='Adam updates, then L-BFGS updates (different costs)', ylabel='Validation NLL ↓')
    axes[0, 0].legend(fontsize=7)
    zoom = axes[0, 0].inset_axes([.45, .34, .51, .38])
    h = json.loads((ROOT/shown['lbfgs']/'result.json').read_text())['history']
    zoom.plot([v['step'] for v in h], [v['nll'] for v in h], color='tab:purple', marker='.', ms=3)
    selected_step = json.loads((ROOT/shown['lbfgs']/'result.json').read_text())['best_step']
    chosen = next(v for v in h if v['step'] == selected_step)
    zoom.scatter(chosen['step'], chosen['nll'], color='tab:purple', marker='*', s=90, zorder=4)
    zoom.set_title('Full-data refinement (zoom)', fontsize=8)
    zoom.tick_params(labelsize=7)
    zoom.set_xlabel('L-BFGS steps', fontsize=7)
    zoom.grid(alpha=.2)
    for key in ('signed', 'penalized'):
        name = shown[key]
        history = json.loads((ROOT/name/'result.json').read_text())['history']
        axes[0, 1].plot([h['sweep'] for h in history], [h['objective'] for h in history],
                        marker='.', color=colors[key], label=key+': '+short(name))
    axes[0, 1].set(title='Signed TT: decreasing the actual training objective', xlabel='Alternating core sweeps', ylabel='Penalized least-squares objective ↓')
    axes[0, 1].legend(fontsize=7)
    truth = np.load(ROOT/'truth_evaluation.npz')
    axes[1, 0].plot(truth['t'], truth['density'], color='black', lw=2, label='True density (evaluation only)')
    for key in ('adam', 'lbfgs', 'signed', 'penalized'):
        name = shown[key]
        profile = np.load(ROOT/name/'profile_free.npz')['density']
        crossings = results[name]['raw_tt_sign_changes_on_true_curve']
        axes[1, 0].plot(truth['t'], profile, color=colors[key], lw=1,
                        label=f'{key}; {crossings} raw-TT sign crossings')
    axes[1, 0].set(title='Joint-density ripples along the true curve', xlabel='True curve parameter (used only after fitting)', ylabel='Joint density evaluated on curve')
    axes[1, 0].legend(fontsize=7)
    controls = ['squared_n25_r16_l0.0001_b1024_s0', 'squared_n49_r32_l0.0001_b1024_s0',
                'squared_n65_r32_l0.0001_b1024_s0', 'squared_n49_r64_l0.0001_b1024_s0',
                'squared_n49_r32_l0.0001_b1024_s1', shown['lbfgs'], shown['signed'], shown['penalized']]
    ticklabels = ['25 modes\nr16', '49 modes\nr32', '65 modes\nr32', '49 modes\nr64', '49 modes\nr32, seed1',
                  'L-BFGS\n49, r32', 'Signed\nselected', 'Signed +\npenalty']
    axes[1, 1].bar(np.arange(len(controls)), [results[n]['relative_l2'] for n in controls],
                    color=['tab:green']*5+['tab:purple', 'tab:blue', 'tab:orange'])
    axes[1, 1].set_xticks(np.arange(len(controls)), ticklabels, fontsize=8)
    axes[1, 1].set(title='Capacity, seed, and objective controls', ylabel='Full-dimensional relative L2 ↓')
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle('Noise-free swirl: optimization and spikes without deconvolution', fontsize=16)
    fig.get_layout_engine().set(rect=(0, .075, 1, .875))
    fig.text(.015, .016, 'Green capacity controls: 4,000 Adam updates, batch 1,024, λ=10⁻⁴; same near-uniform initialization recipe.\n'
             'Stars: selected checkpoints. L-BFGS starts from the selected Adam checkpoint and adds 100 full-data updates.\n'
             'The signed forward model retains negative values during all updates. Clipped likelihood may be zero; it is never silently floored.', fontsize=9)
    fig.savefig(FIGS/'direct_optimization_free_dim4_training.png', dpi=160)
    plt.close(fig)
