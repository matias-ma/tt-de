"""Create compact cryo-EM result figures without deleting checkpoints."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from chebttge.chebtt import ChebTT
from chebttge.cores import ChebyshevCoeffCore
from cryoem_common import kde_marginal_fn, marginal_density_fn


def load_tt(path):
    saved = np.load(path)
    domain = saved['domain']
    return ChebTT(cores=tuple(ChebyshevCoeffCore(jnp.asarray(saved[f'core_{k}']), tuple(domain[k]))
                              for k in range(len(domain))), domain=jnp.asarray(domain))


def evaluate_batched(function, points, batch=512):
    evaluate = jax.jit(function)
    return np.concatenate([np.asarray(evaluate(jnp.asarray(points[i:i+batch])))
                           for i in range(0, len(points), batch)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('artifacts/cryoem'))
    parser.add_argument('--keep-workfiles', action='store_true')
    parser.add_argument('--figure-dir', type=Path, default=Path('figs/convolution_benchmarks'))
    args = parser.parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    directory = args.directory
    records = [json.loads(p.read_text()) for p in sorted(directory.glob('*.json'))]
    runs = [r for r in records if isinstance(r, dict) and 'path_relative_l2' in r]
    common_file = directory/'common_validation.json'
    common = {r['run']: r['validation_nll'] for r in json.loads(common_file.read_text())} if common_file.exists() else {}
    gaussian = [r for r in runs if r['backend'] == 'gaussian']
    score = lambda r: common.get(Path(r['output']).stem, r['best_validation_nll'])
    best = min(gaussian, key=score)
    selected = [r for r in runs if r['backend'] == 'spectral']
    selected += [r for r in gaussian if r['rank'] == 16 and r['degree'] == 3 and r['steps'] == 1500]
    if best not in selected:
        selected.append(best)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), constrained_layout=True)
    states = np.arange(1, len(best['path_truth'])-1)
    axes[0].plot(states, best['path_truth'][1:-1], color='black', lw=2, label='State probabilities')
    axes[0].plot(states, best['path_kde'][1:-1], lw=2, label=f"KDE ({best['kde_path_relative_l2']:.3f})")
    for record in selected:
        label = f"{record['backend']}, r={record['rank']}, n={record['degree']} ({record['path_relative_l2']:.3f})"
        if record is best:
            label += ' [best validation NLL]'
        axes[0].plot(states, record['path_estimate'][1:-1], label=label, alpha=.85)
    axes[0].set(xlabel='Path state', ylabel='Normalized density at state anchor',
                title='10D cryo-EM path diagnostic (relative L2 in legend)')
    axes[0].legend(fontsize=8)
    for record in runs:
        if common and Path(record['output']).stem not in common:
            continue
        marker = {'gaussian': 'o', 'spectral': 's', 'naive': '^'}[record['backend']]
        axes[1].scatter(score(record), record['path_relative_l2'], marker=marker, s=55)
    axes[1].axhline(best['kde_path_relative_l2'], color='gray', ls='--', label='KDE path error')
    axes[1].set(xlabel='Common held-out observation NLL', ylabel='Path relative L2',
                title='Observation likelihood versus clean path accuracy')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle(f"10D cryo-EM | n_train={best['n_train']:,} | steps={best['steps']:,} "
                 f"(selected {best['best_step']:,}) | batch={best['batch_size']}")
    fig.savefig(args.figure_dir/'errors.png', dpi=180)
    plt.close(fig)

    model_specs = [('Naive TT', 'naive_r16_n3'), ('Direct Gaussian', 'gaussian_r8_n5'),
                   ('General TT kernel', 'spectral_r8_n5')]
    model_records = {Path(r['output']).stem: r for r in runs}
    models = [(label, load_tt(directory/f'{name}.npz'), model_records[name]) for label, name in model_specs]
    cache = np.load('cryoem_data/cache/path_noise1_dim10.npz')
    observed, covariance = np.asarray(cache['points']), np.asarray(cache['cov_mean'])
    rng = np.random.default_rng(0)
    train = observed[rng.permutation(len(observed))[len(observed)//5:]]
    pairs = [(0, 1), (2, 3), (8, 9)]
    fig, axes = plt.subplots(len(pairs), 6, figsize=(23, 11.5), constrained_layout=True)
    column_titles = ['Observed data', 'Prescribed noise', 'Gaussian KDE', *[m[0] for m in models]]
    for row, pair in enumerate(pairs):
        lo = min(float(model[1].domain[pair[0], 0]) for model in models)
        hi = max(float(model[1].domain[pair[0], 1]) for model in models)
        ylo = min(float(model[1].domain[pair[1], 0]) for model in models)
        yhi = max(float(model[1].domain[pair[1], 1]) for model in models)
        x, y = np.linspace(lo, hi, 121), np.linspace(ylo, yhi, 121)
        xx, yy = np.meshgrid(x, y, indexing='xy')
        grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
        hist, _, _ = np.histogram2d(observed[:, pair[0]], observed[:, pair[1]], bins=121,
                                    range=[[lo, hi], [ylo, yhi]], density=True)
        std = np.sqrt(np.diag(covariance)[list(pair)])
        noise = np.exp(-.5*((xx/std[0])**2+(yy/std[1])**2))/(2*np.pi*std.prod())
        kde = kde_marginal_fn(jnp.asarray(train), pair)
        kde_values = evaluate_batched(kde, grid).reshape(xx.shape)
        model_values = []
        for _, model, _ in models:
            marginal = marginal_density_fn(model, 'tt', pair)
            model_values.append(evaluate_batched(marginal, grid).reshape(xx.shape))
            jax.clear_caches()
        vmax = max(hist.max(), kde_values.max(), *(z.max() for z in model_values))
        panels = [hist.T, noise, kde_values, *model_values]
        for column, (ax, values, title) in enumerate(zip(axes[row], panels, column_titles, strict=True)):
            image = ax.imshow(values, extent=(lo, hi, ylo, yhi), origin='lower', aspect='equal',
                              cmap='magma' if column == 1 else 'viridis',
                              vmin=0, vmax=None if column == 1 else vmax)
            suffix = ''
            if column >= 3:
                record = models[column-3][2]
                suffix = f"\npath L2={record['path_relative_l2']:.3f}; r={record['rank']}, n={record['degree']}"
            ax.set(title=title+suffix if row == 0 or column >= 3 else '',
                   xlabel=f'z{pair[0]+1}', ylabel=f'z{pair[1]+1}')
            if column == 1:
                fig.colorbar(image, ax=ax, shrink=.75)
    fig.suptitle(f'10D cryo-EM marginals | {len(observed):,} observed points | '
                 'all model panels use exact TT/KDE marginals')
    fig.savefig(args.figure_dir/'marginals.png', dpi=170)
    plt.close(fig)
    cache.close()



if __name__ == '__main__':
    main()
