"""Render a saved GL convolution fit without repeating its training run."""

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from chebttge.chebtt import ChebTT
from chebttge.cores import ChebyshevCoeffCore

from gl_common import save_noisy_marginal_figure
from gl_gpu_noisy import NOISE_STDS, NOISE_WEIGHTS, _noniso_covariance
from ftde.ginzburg_landau import ginzburg_landau_log_normalizer, sample_ginzburg_landau
from ftde.kde import select_gaussian_kde_bandwidth
from ftde.noise_kernel import (
    build_gaussian_mixture_noise_kernel_tt,
    build_pairwise_correlated_noise_kernel_tt,
)


def render(path: Path, pairs, figure_dir: Path):
    record = json.loads(path.read_text())
    dim = record['dim']
    noise_scale = record.get('noise_scale', 1.)
    domain_file = np.load(path.with_suffix('.npz'))
    domain = jnp.asarray(domain_file['domain'])

    def fitted(label):
        cores = tuple(ChebyshevCoeffCore(jnp.asarray(domain_file[f'{label}_{k}']),
                                          tuple(np.asarray(domain[k]))) for k in range(dim))
        return ChebTT(cores=cores, domain=domain)

    rng = np.random.default_rng(0)
    clean_train = np.asarray(sample_ginzburg_landau(seed=0, n=record['n_train'], d=dim))
    if record['noise'] == 'mixture':
        component = rng.choice(2, size=record['n_train'], p=np.asarray(NOISE_WEIGHTS))
        noise = rng.normal(size=clean_train.shape) * np.asarray(NOISE_STDS)[component, None] * noise_scale
        kernel = build_gaussian_mixture_noise_kernel_tt(
            NOISE_STDS*noise_scale, NOISE_WEIGHTS, domain, dim,
            n_coeffs=record.get('kernel_n_coeffs', 257))
    elif record['noise'] == 'paired':
        covariance = _noniso_covariance(dim, noise_scale)
        noise = rng.multivariate_normal(np.zeros(dim), covariance, size=record['n_train'])
        kernel = build_pairwise_correlated_noise_kernel_tt(
            float(np.sqrt(covariance[0, 0])), .5, domain, dim,
            max_rank=record['kernel_rank'], n_starts=64, gd_steps=10)
    else:
        raise ValueError('saved-kernel marginal plots require mixture or paired noise')

    train = jnp.asarray(clean_train + noise)
    bandwidth = select_gaussian_kde_bandwidth(train)
    clean_test = sample_ginzburg_landau(seed=1, n=record['n_test'], d=dim)
    metrics = record['metrics']
    for pair in pairs:
        save_noisy_marginal_figure(
            dim, fitted('naive'), fitted('deconvolved'), kernel, clean_test, train,
            bandwidth, domain, *pair,
            log_z=ginzburg_landau_log_normalizer(dim),
            kde_error=metrics['kde']['relative_squared_sample_error'],
            naive_error=metrics['naive']['relative_squared_sample_error'],
            deconv_error=metrics['deconvolved']['relative_squared_sample_error'],
            naive_steps=record['naive_steps'], deconv_steps=record['deconv_steps'],
            n_density_points=record['n_train'], noise_type=record['noise'],
            naive_rank=record['rank'], deconv_rank=record['rank'],
            degree=record['n_coeffs'] - 1, sigma=noise_scale, fit_seconds=record['fit_seconds'],
            selected_step=record['histories']['deconvolved']['best_step'],
            figure_dir=figure_dir / record['noise'],
            deconv_label='TT with RSI deconvolution' if record.get('convolution_backend') == 'rsi' else 'Deconvolved TT',
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('records', nargs='+', type=Path)
    parser.add_argument('--pair', type=int, nargs=2, action='append', help='repeat for multiple marginals; default: 0 1, 2 3, 0 9')
    parser.add_argument('--figure-dir', type=Path, default=Path('figs/ginzburg_landau/rsi'))
    args = parser.parse_args()
    for path in args.records:
        dim = json.loads(path.read_text())['dim']
        pairs = args.pair or [(i, j) for i, j in [(0, 1), (2, 3), (0, 9)] if j < dim]
        render(path, pairs, args.figure_dir)


if __name__ == '__main__':
    main()
