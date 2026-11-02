"""Publish the current direct-TT study into the existing figure folders."""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('artifacts/noise_free_v2')


def label(name):
    aliases = {'baseline': 'Direct TT baseline', 'previous_direct': 'Previous direct TT',
               'uniform_start': 'Near-uniform start', 'sweep_flat': 'Core Newton sweeps',
               'lbfgs_hessian_continue': 'Adam → L-BFGS (Hessian)', 'hessian': 'Full Hessian penalty',
               'sweep_hessian_strong_flat_long': 'Core sweeps + Hessian λ=0.01',
               'sweep_hessian_medium_flat_long': 'Core sweeps + Hessian λ=0.003',
               'sweep_hessian_flat_long': 'Core sweeps + Hessian λ=0.001',
               'sweep_hessian_continue': 'Adam → core sweeps (Hessian)'}
    if name in aliases:
        return aliases[name]
    return name.replace('hessian_', 'Full Hessian λ=').replace('h1_', 'Slope penalty λ=').replace('_', ' ')


def plot_case(case):
    folder = ROOT/case
    result = json.loads((folder/'evaluation.json').read_text())
    selected = result['selected']
    swirl = case.startswith('swirl')
    dim = int(case[5:]) if swirl else 6
    dest = Path('figs/swirl' if swirl else 'figs/ginzburg_landau')
    dest.mkdir(parents=True, exist_ok=True)
    base = Path('artifacts/swirl')/f'swirl_free_dim{dim}' if swirl else Path('artifacts/figure_refresh/gl_free_dim6')
    with np.load(base/'data.npz') as data:
        train = data['train']
    shown = ['uniform_start', 'baseline', selected]
    if selected == 'baseline':
        shown[-1] = 'hessian' if not swirl else 'hessian_0.01'
    if swirl:
        shown.insert(2, 'previous_direct')
    truth = np.load(folder/'truth.npz')
    for i, j in [(0, 1), (2, 3)]:
        fig, axes = plt.subplots(1, len(shown)+2,
                                 figsize=(3.35*(len(shown)+2), 4.5), layout='constrained')
        grids = [np.load(folder/name/'marginals.npz') for name in shown]
        grid = grids[0]['grid']
        panels = []
        panels.append(('True marginal\nEvaluation only', truth[f'pair{i}{j}']))
        hist, xe, ye = np.histogram2d(train[:, i], train[:, j], bins=100,
                                     range=[(grid[0], grid[-1])]*2, density=True)
        vmax = float(truth[f'pair{i}{j}'].max())
        axes[0].pcolormesh(grid, grid, panels[0][1], cmap='magma', vmin=0, vmax=vmax, shading='auto')
        axes[0].set_title(panels[0][0], fontsize=10)
        offset = 1
        axes[offset].pcolormesh(xe, ye, hist.T, cmap='magma', vmin=0, vmax=vmax, shading='auto')
        axes[offset].set_title(f'Observed samples\nN={len(train):,}', fontsize=10)
        for ax, name, marginal in zip(axes[offset+1:], shown, grids, strict=True):
            image = ax.pcolormesh(grid, grid, marginal[f'pair{i}{j}'], cmap='magma', vmin=0, vmax=vmax, shading='auto')
            title = label(name)+f"\nFull {dim}D relative L2 = {result['models'][name]['relative_l2']:.3f}"
            if name == selected:
                title += '\nSelected using observations'
            ax.set_title(title, fontsize=10)
        for ax in axes:
            ax.set(xlim=(grid[0], grid[-1]), ylim=(grid[0], grid[-1]), xlabel=f'x{i}', ylabel=f'x{j}', aspect='equal')
        fig.colorbar(image, ax=axes, shrink=.72, extend='max', label='Marginal density; common scale')
        fig.suptitle(f'Noise-free {"swirl" if swirl else "Ginzburg–Landau"}, {dim}D: direct Chebyshev TT optimization', fontsize=15)
        fig.get_layout_engine().set(rect=(0, .09, 1, .89))
        fig.text(.015, .015, 'Single squared, normalized Chebyshev TT; 49 amplitude modes, maximum rank 32. Initialization uses only the box and random seed.\n'
                 'Settings and checkpoints selected from held-out observations; known density enters only the final error calculation.', fontsize=9)
        fig.savefig(dest/f'noise_free_dim{dim}_marg_{i}{j}.png', dpi=160)
        plt.close(fig)
    frozen = json.loads((folder/'frozen.json').read_text())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout='constrained')
    for name in dict.fromkeys(['baseline', selected, *[n for n in ['sweep_flat', 'lbfgs_hessian_continue'] if n in frozen['names']]]):
        record = json.loads((folder/name/'result.json').read_text())
        history = record['history']
        axes[0, 0].plot([h['seconds'] for h in history], [h['validation_nll'] for h in history], '.-', label=label(name))
        row = next(h for h in history if h['step'] == record['best_step'])
        axes[0, 0].scatter(row['seconds'], row['validation_nll'], marker='*', s=90, zorder=4)
    axes[0, 0].set(title='Coefficient optimization\nStars mark retained checkpoints', xlabel='Seconds within this stage, including compilation', ylabel='Held-out joint NLL ↓')
    axes[0, 0].legend(fontsize=8)
    names = frozen['names']
    records = frozen['selection']['candidates']
    best = min(r['validation_nll'] for r in records)
    for n, r in zip(names, records, strict=True):
        axes[0, 1].scatter(r['roughness'], r['validation_nll']-best, s=65 if n == selected else 25,
                          marker='*' if n == selected else 'o', color='tab:red' if n == selected else 'tab:blue')
    axes[0, 1].set(xscale='log', yscale='symlog', title='Observation-only comparison\nAll completed runs',
                   xlabel='Common full-Hessian amplitude energy ↓', ylabel='Excess validation NLL ↓')
    axes[0, 1].set_yscale('symlog', linthresh=.005)
    axes[0, 1].annotate(label(selected), (records[names.index(selected)]['roughness'], records[names.index(selected)]['validation_nll']-best),
                         xytext=(8, 12), textcoords='offset points', fontsize=8)
    if swirl:
        axes[1, 0].plot(truth['t'], truth['profile'], color='black', label='Truth (evaluation only)', lw=1.5)
        for name in dict.fromkeys(['baseline', 'previous_direct', selected, 'sweep_flat']):
            profile = np.load(folder/name/'profile.npz')['density']
            axes[1, 0].plot(truth['t'], profile, label=label(name), lw=1)
        axes[1, 0].set(title='Joint density along the true curve', xlabel='Curve parameter (evaluation only)', ylabel='Joint density')
        axes[1, 0].legend(fontsize=8)
    else:
        axes[1, 0].axis('off')
    evalnames = [n for n in dict.fromkeys(['baseline', selected, 'hessian_0.01', 'h1_0.01',
                 'sweep_flat', 'lbfgs_hessian_continue', 'previous_direct']) if n in result['models']]
    axes[1, 1].bar(np.arange(len(evalnames)), [result['models'][n]['relative_l2'] for n in evalnames],
                   color=['tab:red' if n == selected else 'tab:blue' for n in evalnames])
    axes[1, 1].set_xticks(range(len(evalnames)), [label(n) for n in evalnames], rotation=25, ha='right', fontsize=8)
    axes[1, 1].set(title='Full-dimensional error after freezing selection', ylabel='Relative L2 ↓')
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.suptitle(f'{case}: optimization, regularization, and final density recovery', fontsize=15)
    fig.get_layout_engine().set(rect=(0, .09, 1, .84))
    fig.text(.015, .015, 'NLL differences use the same validation observations. The one-SE choice is a tuning heuristic, not a post-search confidence guarantee.\n'
             'Continuation runs start from an optimized TT; their time axis begins at that stage. No external fitted signal initializer.', fontsize=9)
    fig.savefig(dest/f'noise_free_dim{dim}_training.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    for case in ('swirl4', 'swirl10', 'gl6'):
        plot_case(case)
