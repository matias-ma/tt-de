"""Render selected cryo-EM fits in the existing marginal and path layouts."""

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from cryoem_density import plot_pair
from plot_convolution_benchmarks import load_tt


def render(rsi_path, naive_path, destination):
    rsi = json.loads(rsi_path.read_text())
    naive = json.loads(naive_path.read_text())
    for key in ('dim', 'rank', 'degree', 'seed', 'keep_outside', 'noise_eigenbasis'):
        if rsi[key] != naive[key]:
            raise ValueError(f'Incomparable fits: {key}')
    if rsi['noise_eigenbasis']:
        raise ValueError('This renderer expects the original latent coordinates')
    dim = rsi['dim']
    deconv_fit, naive_fit = load_tt(rsi_path.with_suffix('.npz')), load_tt(naive_path.with_suffix('.npz'))
    if not np.allclose(deconv_fit.domain, naive_fit.domain):
        raise ValueError('Fits must share their signal domain')
    with np.load(f'cryoem_data/cache/path_noise1_dim{dim}.npz') as cache:
        raw = cache['points']
    indices = np.random.default_rng(rsi['seed']).permutation(len(raw))
    train = raw[indices[len(raw)//5:]]
    domain = np.asarray(deconv_fit.domain)
    if not rsi['keep_outside']:
        train = train[np.all((train >= domain[:, 0]) & (train <= domain[:, 1]), axis=1)]
    destination.mkdir(parents=True, exist_ok=True)
    pairs = [(0, 1), (2, 3)] + ([(8, 9)] if dim >= 10 else [])
    with plt.rc_context({'font.size': 12}):
        for i, j in pairs:
            plot_pair(jnp.asarray(train), deconv_fit.domain, naive_fit, deconv_fit, 'tt', i, j, dim,
                      figure_dir=destination, deconv_label='TT with RSI deconvolution')

    states = np.arange(1, len(rsi['path_truth']) - 1)
    with plt.rc_context({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False}):
        fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
        ax.plot(states, rsi['path_truth'][1:-1], color='#202124', lw=2.5, label='True state probabilities')
        curves = [('Gaussian KDE', rsi['path_kde'], rsi['kde_path_relative_l2'], '#0072B2', '--'),
                  ('TT without noise model', naive['path_estimate'], naive['path_relative_l2'], '#E69F00', '-.'),
                  ('TT with RSI deconvolution', rsi['path_estimate'], rsi['path_relative_l2'], '#009E73', '-')]
        for label, values, error, color, style in curves:
            ax.plot(states, values[1:-1], color=color, ls=style, lw=2,
                    label=f'{label}  (relative error {error:.3f})')
        ax.set(xlabel='Conformational state', ylabel='Normalized density at state anchor',
               title=f'Cryo-EM along-path density  |  {dim} dimensions')
        ax.grid(alpha=.18)
        ax.legend(frameon=False, fontsize=10)
        fig.savefig(destination/f'cryoem_path_dim{dim}.png', dpi=180, bbox_inches='tight')
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        h = rsi['history']
        ax.plot([x['step'] for x in h], [x['validation_nll'] for x in h], color='#009E73', lw=2)
        ax.axvline(rsi['best_step'], color='#555555', ls='--', label=f"Selected step {rsi['best_step']:,}")
        ax.set(xlabel='Training step', ylabel='Held-out noisy negative log likelihood',
               title=f'RSI deconvolution training  |  {dim}D cryo-EM')
        ax.grid(alpha=.18)
        ax.legend(frameon=False)
        fig.savefig(destination/f'cryoem_training_dim{dim}.png', dpi=180, bbox_inches='tight')
        plt.close(fig)
    print(f'Saved cryo-EM {dim}D PNGs to {destination}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rsi', type=Path)
    parser.add_argument('naive', type=Path)
    parser.add_argument('--figure-dir', type=Path, default=Path('figs/cryoem/rsi'))
    args = parser.parse_args()
    render(args.rsi, args.naive, args.figure_dir)
