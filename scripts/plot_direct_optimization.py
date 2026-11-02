"""Figures for completed direct TT experiments; no training or selection."""

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ftde.swirl import swirl_pair_marginal

ROOT = Path('artifacts/direct_optimization')
FIGS = Path('figs/swirl')


def training_trace(root, name):
    result = json.loads((root/name/'result.json').read_text())
    history = result['history']
    xs, ys = [v['step'] for v in history], [v['nll'] for v in history]
    offset = 0
    if name.endswith('_lbfgs'):
        parent = json.loads((root/name.removesuffix('_lbfgs')/'result.json').read_text())
        offset = parent['best_step']
        previous = [v for v in parent['history'] if v['step'] <= offset]
        xs = [v['step'] for v in previous]+[offset+v['step'] for v in history[1:]]
        ys = [v['nll'] for v in previous]+[v['nll'] for v in history[1:]]
    selected = next(v for v in history if v['step'] == result['best_step'])
    return xs, ys, offset+selected['step'], selected['nll']


def label(name):
    if name == 'previous_squared':
        return 'Historical squared TT\nKDE start'
    if name == 'uniform_start':
        return 'Near-uniform start\nNo density fitting'
    n = name.split('_n')[1].split('_')[0]
    rank = name.split('_r')[1].split('_')[0]
    if name.startswith('signed'):
        if '_neg' in name:
            eta = name.split('_neg')[-1]
            return f'Signed TT + negative penalty {eta}\n{n} modes, middle rank {rank}; final clip'
        return f'Unconstrained signed TT\n{n} modes, middle rank {rank}; final clip'
    if name.endswith('_lbfgs'):
        return f'Squared TT: Adam + full-data L-BFGS\n{n} modes, rank {rank}; uniform start'
    return f'Squared TT: Adam\n{n} modes, rank {rank}; uniform start'


def four():
    root = ROOT/'swirl4'
    evaluation = json.loads((root/'evaluation_q97.json').read_text())
    results = evaluation['models']
    selection = json.loads((root/'selection_q97.json').read_text())
    def best(names):
        return min(names, key=lambda n: selection['results'][n]['validation_nll'])
    signed = best([n for n in results if n.startswith('signed') and '_neg' not in n])
    penalized = selection['penalized_selected']
    penalty_nll_best = best([n for n in results if n.startswith('signed') and '_neg' in n])
    adam = best([n for n in results if n.startswith('squared') and '_lbfgs' not in n])
    lbfgs = best([n for n in results if n.endswith('_lbfgs')])
    displayed = ['uniform_start', 'previous_squared', adam, lbfgs, signed, penalized]
    data = np.load('artifacts/swirl/swirl_mixture_dim4/data.npz')['train']
    for i, j in [(0, 1), (2, 3)]:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8.3), layout='constrained')
        grid = np.linspace(-2.5, 2.5, 181)
        truth = swirl_pair_marginal(4, i, j, grid, grid)
        vmax = float(truth.max())
        axes.flat[0].pcolormesh(grid, grid, truth, shading='auto', cmap='magma', vmin=0, vmax=vmax)
        axes.flat[0].set_title('True clean density\nEvaluation only')
        hist, xe, ye = np.histogram2d(data[:, i], data[:, j], bins=100, range=[[-2.5, 2.5]]*2)
        hist = hist/(len(data)*np.diff(xe)[:, None]*np.diff(ye)[None])
        axes.flat[1].pcolormesh(xe, ye, hist.T, shading='auto', cmap='magma', vmin=0, vmax=vmax)
        axes.flat[1].set_title('Noisy observations\n100,000 training samples')
        for ax, name in zip(list(axes.flat)[2:], displayed, strict=True):
            marginal = np.load(root/name/'marginals_q97.npz')
            nodes = marginal['nodes']
            image = ax.pcolormesh(nodes[i], nodes[j], marginal[f'pair{i}{j}'], shading='auto',
                                 cmap='magma', vmin=0, vmax=vmax)
            ax.set_title(label(name)+f"\nRelative L2 = {results[name]['relative_l2']:.3f}", fontsize=10)
        for ax in axes.flat:
            ax.set_facecolor(plt.colormaps['magma'](0.))
            ax.set(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), xlabel=f'x{i}', ylabel=f'x{j}', aspect='equal')
        for ax in axes[0]:
            ax.set_xlabel('')
        fig.colorbar(image, ax=axes, shrink=.75, label='Marginal density; common scale', extend='max')
        fig.suptitle('4D swirl: density learned by optimizing TT coefficients', fontsize=16)
        fig.get_layout_engine().set(rect=(0, .075, 1, .885))
        fig.text(.015, .018, 'New fits start near uniform. No KDE or learned signal mixture enters their construction.\n'
                 'Signed fits are clipped and renormalized jointly before marginalization. Historical fit is a comparison only.\n'
                 'Penalized fit: least negative mass within one SE of best validation likelihood. L2 is evaluated afterward; capacities/budgets differ.', fontsize=9)
        fig.savefig(FIGS/f'direct_optimization_dim4_marg_{i}{j}.png', dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), layout='constrained')
    for name in [adam, lbfgs]:
        xs, ys, chosen_x, chosen_y = training_trace(root, name)
        line, = axes[0, 0].plot(xs, ys, marker='.', label=label(name).replace('\n', '; '))
        axes[0, 0].scatter(chosen_x, chosen_y, marker='*', s=100, color=line.get_color(), zorder=3)
    axes[0, 0].set(title='Squared TT: learning from the flat start', xlabel='Adam updates, then L-BFGS updates (different costs)', ylabel='Negative log likelihood ↓')
    axes[0, 0].legend(fontsize=7, loc='upper right')
    zoom = axes[0, 0].inset_axes([.44, .30, .53, .40])
    h = json.loads((root/lbfgs/'result.json').read_text())['history']
    zoom.plot([v['step'] for v in h], [v['nll'] for v in h], color='tab:purple', marker='.', ms=3)
    selected_step = json.loads((root/lbfgs/'result.json').read_text())['best_step']
    chosen = next(v for v in h if v['step'] == selected_step)
    zoom.scatter(chosen['step'], chosen['nll'], color='tab:purple', marker='*', s=90, zorder=4)
    zoom.set_title('Full-data refinement (zoom)', fontsize=8)
    zoom.tick_params(labelsize=7)
    zoom.set_xlabel('L-BFGS steps', fontsize=7)
    zoom.grid(alpha=.2)
    for name in [signed, penalized]:
        h = json.loads((root/name/'result.json').read_text())['history']
        axes[0, 1].plot([v['sweep'] for v in h], [v['objective'] for v in h], label=label(name).replace('\n', '; '))
    axes[0, 1].set(title='Signed TT: actual training objective', xlabel='Alternating sweeps', ylabel='Penalized objective ↓')
    axes[0, 1].legend(fontsize=8)
    names = [n for n in selection['results'] if n.startswith('signed')]
    colors = ['tab:orange' if '_neg' in n else 'tab:blue' for n in names]
    axes[1, 0].scatter([selection['results'][n]['negative_mass'] for n in names],
                       [selection['results'][n]['validation_nll'] for n in names], c=colors, s=60)
    axes[1, 0].axhline(selection['results'][adam]['validation_nll'], color='black', ls='--', label='Squared TT from uniform')
    axes[1, 0].set(title='Final clipping changes the estimator', xlabel='Negative mass removed', ylabel='Clipped noisy validation NLL ↓')
    axes[1, 0].text(.04, .94, 'Blue: unconstrained\nOrange: negative-part penalty', transform=axes[1, 0].transAxes, va='top', fontsize=9)
    axes[1, 0].legend(fontsize=8, loc='lower right')
    compare = ['previous_squared', adam, lbfgs, signed, penalized]
    labels = ['Historical', 'Adam', 'Adam +\nL-BFGS', 'Signed', 'Penalty\nselected']
    bar_colors = ['gray', 'tab:green', 'tab:purple', 'tab:blue', 'tab:orange']
    if penalty_nll_best != penalized:
        compare.append(penalty_nll_best)
        labels.append('Penalty\nbest NLL')
        bar_colors.append('goldenrod')
    axes[1, 1].bar(np.arange(len(compare)), [results[n]['relative_l2'] for n in compare], color=bar_colors)
    axes[1, 1].set_xticks(np.arange(len(compare)), labels)
    axes[1, 1].set(title='Independent clean-density evaluation', ylabel='Full-dimensional relative L2 ↓')
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle('Direct optimization: progress, negative mass, and recovery', fontsize=15)
    fig.get_layout_engine().set(rect=(0, .055, 1, .905))
    fig.text(.015, .013, 'All new learning occurs in TT optimization. Negative values remain active in the signed forward model during fitting.\n'
             'Algorithms have different objectives and costs; validation selection is frozen before clean-density evaluation.', fontsize=9)
    fig.savefig(FIGS/'direct_optimization_dim4_training.png', dpi=160)
    plt.close(fig)


def ten(noise_free=False):
    root = ROOT/('swirl_free10' if noise_free else 'swirl10')
    evaluation = json.loads((root/'evaluation.json').read_text())
    names = ['uniform_start', 'previous_squared', evaluation['selection']]
    data = np.load(f'artifacts/swirl/swirl_{"free" if noise_free else "mixture"}_dim10/data.npz')['train']
    for i, j in [(0, 1), (2, 3)]:
        fig, axes = plt.subplots(1, 5, figsize=(19, 4.9), layout='constrained')
        grid = np.linspace(-2.5, 2.5, 161)
        truth = swirl_pair_marginal(10, i, j, grid, grid)
        vmax = float(truth.max())
        axes[0].pcolormesh(grid, grid, truth, shading='auto', cmap='magma', vmin=0, vmax=vmax)
        axes[0].set_title('True clean density\nEvaluation only', fontsize=10)
        hist, xe, ye = np.histogram2d(data[:, i], data[:, j], bins=100, range=[[-2.5, 2.5]]*2)
        hist = hist/(len(data)*np.diff(xe)[:, None]*np.diff(ye)[None])
        axes[1].pcolormesh(xe, ye, hist.T, shading='auto', cmap='magma', vmin=0, vmax=vmax)
        axes[1].set_title(('Noise-free' if noise_free else 'Noisy')+' observations\n100,000 training samples', fontsize=10)
        for ax, name in zip(axes[2:], names, strict=True):
            m = np.load(root/name/'marginals.npz')
            result = evaluation['models'][name]
            image = ax.pcolormesh(m['nodes'][i], m['nodes'][j], m[f'pair{i}{j}'], shading='auto', cmap='magma', vmin=0, vmax=vmax)
            ax.set_title(label(name)+f"\nL2 = {result['relative_l2']:.3f} ± {result['relative_l2_mc_se']:.3f}", fontsize=10)
        for ax in axes:
            ax.set_facecolor(plt.colormaps['magma'](0.))
            ax.set(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), xlabel=f'x{i}', ylabel=f'x{j}', aspect='equal')
        fig.colorbar(image, ax=axes, shrink=.75, extend='max', label='Marginal density')
        fig.suptitle(('Noise-free 10D' if noise_free else '10D')+' swirl: squared TT learned from a near-uniform start', fontsize=16)
        fig.get_layout_engine().set(rect=(0, .12, 1, .83))
        description = ('Noise-free TT likelihood; no measurement noise or fitted signal initializer.' if noise_free else
                       'The new run learns solely through the prescribed-noise convolved TT likelihood. No fitted signal initializer.')
        fig.text(.015, .025, description+'\n'
                 'Full-dimensional L2 uses independent defensive importance sampling; ± is Monte Carlo SE. Historical training recipe and budget differ.', fontsize=9)
        fig.savefig(FIGS/f'direct_optimization_{"free_" if noise_free else ""}dim10_marg_{i}{j}.png', dpi=160)
        plt.close(fig)
    if noise_free:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout='constrained')
        candidates = [n for n in evaluation['models'] if n.startswith('squared')]
        truth = np.load(root/'truth_profile.npz')
        axes[1].plot(truth['t'], truth['density'], color='black', lw=2, label='True density (evaluation only)')
        for name in candidates:
            xs, ys, chosen_x, chosen_y = training_trace(root, name)
            rank = name.split('_r')[1].split('_')[0]
            title = f'Adam, rank {rank}'+(' + L-BFGS' if name.endswith('_lbfgs') else '')
            line, = axes[0].plot(xs, ys, marker='.', label=title)
            axes[0].scatter(chosen_x, chosen_y, marker='*', s=100, color=line.get_color(), zorder=3)
            profile = np.load(root/name/'profile.npz')['density']
            changes = evaluation['models'][name]['raw_tt_sign_changes_on_true_curve']
            axes[1].plot(truth['t'], profile, lw=1, label=f'{title}; {changes} amplitude sign crossings')
        axes[0].set(title='Learning from the nearly flat start', xlabel='Adam updates, then full-data L-BFGS updates', ylabel='Validation NLL ↓')
        axes[1].set(title='Joint density on the curve after fitting', xlabel='True curve parameter (evaluation only)', ylabel='Joint density')
        for ax in axes:
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.suptitle('Noise-free 10D swirl: optimization and density gaps', fontsize=15)
        fig.get_layout_engine().set(rect=(0, .10, 1, .85))
        fig.text(.02, .02, '49 Chebyshev modes, maximum ranks 32 and 64; uniform + small random perturbation.\n'
                 'Stars mark selected checkpoints; L-BFGS starts from the selected Adam checkpoint. Adam was run for 4,000 updates.', fontsize=9)
        fig.savefig(FIGS/'direct_optimization_free_dim10_training.png', dpi=160)
        plt.close(fig)


if __name__ == '__main__':
    FIGS.mkdir(exist_ok=True)
    if (ROOT/'swirl4/evaluation_q97.json').exists():
        four()
    if (ROOT/'swirl10/evaluation.json').exists():
        ten()
    if (ROOT/'swirl_free10/evaluation.json').exists():
        ten(noise_free=True)
