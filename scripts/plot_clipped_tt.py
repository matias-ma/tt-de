"""Figures for frozen direct-TT trials; target truth is used only after fitting."""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from refresh_density_fits import read_model
from scipy.interpolate import RegularGridInterpolator

from ftde.marginals import squared_pair_grid
from ftde.swirl import curve, swirl_pair_marginal

root = Path('artifacts/clipped_tt')
base = Path('artifacts/swirl/swirl_mixture_dim4')
figs = Path('figs/swirl')
selection = json.loads((root/'selection.json').read_text())
old = read_model(base/'selected.npz')
old_l2 = json.loads((base/'integrated_l2.json').read_text())['rank64_h2_0.0001']['relative_l2']
cold = min((r for r in selection['candidates'] if '_warm_' not in r['name']), key=lambda r: r['nll'])['name']
warm = selection['warm_selected']
names = [cold, warm]
saved = [np.load(root/name/'marginals_order129.npz') for name in names]
audits = [json.loads((root/name/'audit.json').read_text())[-1] for name in names]
selected_steps = [json.loads((root/name/'result.json').read_text())['best_step'] for name in names]
summaries = []
old_valid = np.load(root/'baseline_validation.npy')
old_test = np.load(root/'baseline_test.npy')
for entry in selection['candidates']:
    directory = root/entry['name']
    if not (directory/'audit.json').exists():
        continue
    audit = json.loads((directory/'audit.json').read_text())
    final = audit[-1]
    row = {**entry, **final}
    for label, reference in [('validation', old_valid), ('test', old_test)]:
        delta = np.load(directory/f'{label}_order{final["order"]}.npy')-reference
        row[label+'_improvement'] = float(delta.mean())
        row[label+'_paired_se'] = float(delta.std(ddof=1)/np.sqrt(len(delta)))
    if len(audit) > 1:
        previous = audit[-2]
        a = np.load(directory/f'validation_order{previous["order"]}.npy')
        b = np.load(directory/f'validation_order{final["order"]}.npy')
        row['quadrature_nll_change'] = final['nll']-previous['nll']
        row['quadrature_max_log_change'] = float(np.max(np.abs(a-b)))
        row['quadrature_l2_change'] = final['relative_l2']-previous['relative_l2']
    summaries.append(row)
(root/'comparison.json').write_text(json.dumps(summaries, indent=2)+'\n')

xs = np.linspace(-2.5, 2.5, 181)
fig, axes = plt.subplots(2, 4, figsize=(16, 8), layout='constrained')
for row, (i, j) in enumerate([(0, 1), (2, 3)]):
    truth = swirl_pair_marginal(4, i, j, xs, xs)
    original = squared_pair_grid(old, i, j, xs, xs)
    grids = [truth, original]
    xx, yy = np.meshgrid(xs, xs)
    for data in saved:
        nodes = data['nodes']
        interpolation = RegularGridInterpolator((nodes[j], nodes[i]), data[f'pair{i}{j}'], bounds_error=False, fill_value=0.)
        grids.append(interpolation(np.stack([yy, xx], axis=-1)))
    titles = ['True density', f'Original squared TT\nfull 4D L2 = {old_l2:.3f}',
              f'Clipped TT, KDE start, step {selected_steps[0]}\nfull 4D L2 = {audits[0]["relative_l2"]:.3f}',
              f'Clipped TT, recovery start, step {selected_steps[1]}\nfull 4D L2 = {audits[1]["relative_l2"]:.3f}']
    peak = max(float(g.max()) for g in grids)
    for ax, grid, title in zip(axes[row], grids, titles, strict=True):
        im = ax.pcolormesh(xs, xs, grid, shading='nearest', cmap='magma', vmin=0, vmax=peak)
        ax.set(title=title, xlabel=f'x{i}', ylabel=f'x{j}', aspect='equal')
        fig.colorbar(im, ax=ax, shrink=.8)
fig.suptitle('4D noisy swirl: optimize the density TT directly, clipping inside the likelihood', fontweight='bold')
fig.supxlabel('p(x) = max(f_TT(x), 0) / integral max(f_TT, 0). No fitted signal mixture. Each initialization group selected using noisy validation likelihood.\n'
              'Marginals integrate the clipped joint density; truth and full-dimensional L2 are evaluated only after fitting. 129-node quadrature per axis.', fontsize=9)
fig.savefig(figs/'noisy_mixture_clipped_dim4.png', dpi=180)
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout='constrained')
colors = plt.cm.tab10(np.linspace(0, 1, len(selection['candidates'])))
for entry, color in zip(selection['candidates'], colors, strict=True):
    config = json.loads((root/entry['name']/'experiment.json').read_text())
    result = json.loads((root/entry['name']/'result.json').read_text())
    h = result['history']
    label = ('Recovery' if '_warm_' in entry['name'] else 'KDE')+f' start, r{config["rank"]}, H2 {config["strength"]:g}'
    if config.get('negative_penalty', 0):
        label += f', negative {config["negative_penalty"]:g}'
    if config.get('penalty_normalization', 'raw') == 'raw':
        label += ' (original penalty)'
    steps = [r['step'] for r in h]
    axes[0].plot(steps, [r['nll'] for r in h], color=color, label=label)
    axes[1].semilogy(steps, [r['negative_to_positive_mass'] for r in h], color=color)
axes[0].axhline(-old_valid.mean(), color='black', ls='--', label='Original squared TT')
axes[0].set(ylim=(3.856, 3.89), xlabel='Training step', ylabel='Noisy validation NLL', title='Optimization sees observations only')
axes[1].set(xlabel='Training step', ylabel='Discarded negative mass / retained mass', title='Clipping can hide a large negative function')
axes[0].legend(fontsize=7)
for name, color, label in [(cold, '#d95f02', 'Clipped TT, KDE start'), (warm, '#1976b9', 'Clipped TT, recovery start')]:
    row = next(r for r in summaries if r['name'] == name)
    axes[2].errorbar(['Validation', 'Independent test'], [row['validation_improvement'], row['test_improvement']],
                     yerr=[2*row['validation_paired_se'], 2*row['test_paired_se']], fmt='o-', color=color, capsize=4, label=label)
axes[2].axhline(0, color='black', ls='--')
axes[2].set(ylabel='Log-likelihood gain over original squared TT', title='Independent observations (bars: ±2 paired SE)')
axes[2].legend(fontsize=8)
for ax in axes:
    ax.grid(alpha=.15)
fig.suptitle('Direct clipped TT: training behavior and independent likelihood checks', fontweight='bold')
fig.supxlabel('The negative-mass ratio describes the unnormalized function f_TT. The fitted density is nonnegative throughout.\n'
              'Validation selects checkpoints; the 8,192-row independent test set is used only afterward. Positive gain favors the clipped model.', fontsize=9)
fig.savefig(figs/'noisy_mixture_clipped_training_dim4.png', dpi=180)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout='constrained')
t = np.linspace(0, 1, 1200)
path = curve(t, 4)
for ax, (i, j) in zip(axes, [(0, 1), (2, 3)], strict=True):
    truth = swirl_pair_marginal(4, i, j, xs, xs)
    original = squared_pair_grid(old, i, j, xs, xs)
    for grid, label, color in [(truth, 'Truth', 'black'), (original, 'Original squared TT', '#6b7280')]:
        interp = RegularGridInterpolator((xs, xs), grid, bounds_error=False, fill_value=0.)
        ax.plot(t, interp(path[:, [j, i]]), label=label, color=color)
    for data, label, color in zip(saved, ['Clipped, KDE start', 'Clipped, recovery start'], ['#d95f02', '#1976b9'], strict=True):
        interp = RegularGridInterpolator((data['nodes'][j], data['nodes'][i]), data[f'pair{i}{j}'], bounds_error=False, fill_value=0.)
        ax.plot(t, interp(path[:, [j, i]]), label=label, color=color)
    ax.set(title=f'Marginal ({i}, {j}) along the true curve', xlabel='True path parameter (post-fit diagnosis only)', ylabel='Marginal density on the curve')
    ax.legend(fontsize=8)
    ax.grid(alpha=.15)
fig.suptitle('Removing the square does not by itself guarantee a smooth recovered ridge')
fig.savefig(figs/'noisy_mixture_clipped_ridges_dim4.png', dpi=180)
plt.close(fig)
print(json.dumps(summaries, indent=2))
