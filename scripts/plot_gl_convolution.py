"""Create compact noisy-GL error and marginal figures without deleting checkpoints."""

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
from evaluate_gl_covariance import gl_marginal
from gl_gpu_noisy import NOISE_STDS, NOISE_WEIGHTS, _noniso_covariance

from ftde.ginzburg_landau import sample_ginzburg_landau


def load_tt(saved, prefix):
    domain = saved['domain']
    return ChebTT(cores=tuple(ChebyshevCoeffCore(jnp.asarray(saved[f'{prefix}_{k}']), tuple(domain[k]))
                              for k in range(len(domain))), domain=jnp.asarray(domain))


def evaluate_batched(function, points, batch=512):
    evaluate = jax.jit(function)
    return np.concatenate([np.asarray(evaluate(jnp.asarray(points[i:i+batch])))
                           for i in range(0, len(points), batch)])


def observations(record):
    d, size = record['dim'], record['n_train']
    clean = np.asarray(sample_ginzburg_landau(seed=0, n=size, d=d))
    rng = np.random.default_rng(0)
    if record['noise'] == 'mixture':
        component = rng.choice(2, size=size, p=np.asarray(NOISE_WEIGHTS))
        noise = rng.normal(size=(size, d))*np.asarray(NOISE_STDS)[component, None]
    elif record['noise'] == 'paired':
        noise = rng.multivariate_normal(np.zeros(d), _noniso_covariance(d, 1.), size=size)
    else:
        noise = rng.normal(size=(size, d))*.55
    return clean+noise


def noise_marginal(record, xx, yy):
    if record['noise'] == 'mixture':
        return sum(float(w)*np.exp(-.5*(xx**2+yy**2)/float(s)**2)/(2*np.pi*float(s)**2)
                   for w, s in zip(NOISE_WEIGHTS, NOISE_STDS, strict=True))
    covariance = (_noniso_covariance(record['dim'], 1.)[:2, :2] if record['noise'] == 'paired'
                  else np.eye(2)*.55**2)
    points = np.stack([xx, yy], axis=-1)
    return np.exp(-.5*np.einsum('...i,ij,...j->...', points, np.linalg.inv(covariance), points)
                  -.5*(2*np.log(2*np.pi)+np.linalg.slogdet(covariance)[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('artifacts/gl_convolution'))
    parser.add_argument('--keep-workfiles', action='store_true')
    parser.add_argument('--figure-dir', type=Path, default=Path('figs/gl_convolution'))
    args = parser.parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    directory = args.directory
    names = ['gl8_gaussian', 'gl8_mixture', 'gl8_paired', 'gl20_mixture']
    records = [json.loads((directory/f'{name}.json').read_text())
               for name in names if (directory/f'{name}.json').exists()]
    integrated = {r['noise']+str(r['dim']): json.loads(
        (directory/f"{Path(r['output']).stem}_integrated_l2.json").read_text()) for r in records}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    positions = np.arange(len(records))
    for index, (key, label) in enumerate([('kde', 'KDE'), ('naive', 'Naive TT'),
                                          ('deconvolved', 'Deconvolved TT')]):
        bars = axes[0].bar(positions+(index-1)*.25,
                           [r['metrics'][key]['relative_sample_l2'] for r in records],
                           width=.24, label=label)
        axes[0].bar_label(bars, fmt='%.3f', fontsize=8)
    axes[0].set(ylabel='Relative error on clean test samples', ylim=(0, 1.2),
                title='Held-out sample error')
    for index, (key, label) in enumerate([('naive', 'Naive TT'), ('deconvolved', 'Deconvolved TT')]):
        bars = axes[1].bar(positions+(index-.5)*.32,
                           [integrated[r['noise']+str(r['dim'])][key]['relative_l2'] for r in records],
                           width=.31, label=label)
        axes[1].bar_label(bars, fmt='%.3f', fontsize=8)
    axes[1].set(ylabel='Integrated relative L2', ylim=(0, 1.2), title='Full-density error')
    labels = [f"{r['dim']}D {r['noise']}" for r in records]
    for ax in axes:
        ax.set_xticks(positions, labels)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Noisy Ginzburg–Landau density recovery')
    fig.savefig(args.figure_dir/'errors.png', dpi=180)
    plt.close(fig)

    for record in records:
        saved = np.load(directory/f"{Path(record['output']).stem}.npz")
        naive, deconvolved = load_tt(saved, 'naive'), load_tt(saved, 'deconvolved')
        train = observations(record)
        domain = np.asarray(saved['domain'])
        x = np.linspace(domain[0, 0], domain[0, 1], 121)
        y = np.linspace(domain[1, 0], domain[1, 1], 121)
        xx, yy = np.meshgrid(x, y, indexing='xy')
        grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
        truth = gl_marginal(record['dim'], grid[:, 0], grid[:, 1], 0).reshape(xx.shape)
        hist, _, _ = np.histogram2d(train[:, 0], train[:, 1], bins=121,
                                    range=[[x[0], x[-1]], [y[0], y[-1]]], density=True)
        noise = noise_marginal(record, xx, yy)
        kde = kde_marginal_fn(jnp.asarray(train), (0, 1))
        kde_values = evaluate_batched(kde, grid).reshape(xx.shape)
        naive_values = evaluate_batched(marginal_density_fn(naive, 'tt', (0, 1)), grid).reshape(xx.shape)
        deconv_values = evaluate_batched(marginal_density_fn(deconvolved, 'tt', (0, 1)), grid).reshape(xx.shape)
        vmax = max(truth.max(), hist.max(), kde_values.max(), naive_values.max(), deconv_values.max())
        fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
        panels = [(truth, 'True GL marginal', 'viridis', vmax),
                  (hist.T, f'Observed noisy data ({len(train):,} points)', 'viridis', vmax),
                  (noise, f"Prescribed {record['noise']} noise", 'magma', None),
                  (kde_values, f"KDE; sample L2={record['metrics']['kde']['relative_sample_l2']:.3f}", 'viridis', vmax),
                  (naive_values, f"Naive TT; full L2={integrated[record['noise']+str(record['dim'])]['naive']['relative_l2']:.3f}", 'viridis', vmax),
                  (deconv_values, f"Deconvolved TT; full L2={integrated[record['noise']+str(record['dim'])]['deconvolved']['relative_l2']:.3f}", 'viridis', vmax)]
        for ax, (values, title, cmap, scale) in zip(axes.ravel(), panels, strict=True):
            image = ax.imshow(values, extent=(x[0], x[-1], y[0], y[-1]), origin='lower', aspect='equal',
                              cmap=cmap, vmin=0, vmax=scale)
            ax.set(title=title, xlabel='x1', ylabel='x2')
            if cmap == 'magma':
                fig.colorbar(image, ax=ax, shrink=.75)
        histories = record['histories']
        fig.suptitle(f"{record['dim']}D GL, {record['noise']} noise | n={record['n_train']:,} | "
                     f"rank={record['rank']}, {record['n_coeffs']} coeffs | steps={record['steps']:,} "
                     f"(naive selected {histories['naive']['best_step']:,}, deconv selected "
                     f"{histories['deconvolved']['best_step']:,})")
        fig.savefig(args.figure_dir/f"marginal_{record['dim']}d_{record['noise']}.png", dpi=170)
        plt.close(fig)
        saved.close()
        jax.clear_caches()



if __name__ == '__main__':
    main()
