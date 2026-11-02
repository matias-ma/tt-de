"""Numerical operator validation; no density estimator is fit to known truth."""

import argparse
import json
import time
from itertools import pairwise
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from chebttge.chebtt import ChebTT
from chebttge.cores import ChebyshevCoeffCore
from scipy.fft import dct

from ftde.gaussian_convolution import GaussianConvolution
from ftde.noisy_tt_density import log_density_noisy_tt
from ftde.tt_density import rebuild_tt, tt_params


def random_tt(d, rank, n, seed=0):
    rng = np.random.default_rng(seed)
    ranks = (1,) + (rank,) * (d - 1) + (1,)
    cores = tuple(ChebyshevCoeffCore(
        jnp.asarray(rng.normal(size=(a, b, n)) / np.sqrt(a*n)), (-1., 1.))
        for a, b in pairwise(ranks))
    return ChebTT(cores=cores, domain=jnp.array([[-1., 1.]]*d))


def gaussian_kernel(d, std, n):
    # Independent fixed DCT interpolation of the prescribed noise, not a
    # fit to any signal function. Kernel covers every y-x in [-2,2].
    width = 3 + 4*std
    nodes = width*np.cos(np.pi*np.arange(n)/(n-1))
    values = np.exp(-.5*(nodes/std)**2)/(std*np.sqrt(2*np.pi))
    c = dct(values, type=1)/(n-1)
    c[[0, -1]] *= .5
    cores = tuple(ChebyshevCoeffCore(jnp.asarray(c[None, None, :]), (-width, width))
                  for _ in range(d))
    return ChebTT(cores=cores, domain=jnp.array([[-width, width]]*d))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('.scratch/cryoem/operator_accuracy.json'))
    parser.add_argument("--high-degree", action="store_true", help="only the degree-256 stress case")
    args = parser.parse_args()
    rows = []
    cases = ([(10, 16, 257, .15, False)] if args.high_degree else
             [(3, 4, 17, .15, True), (10, 8, 5, .15, True),
              (10, 16, 33, .1, False), (20, 32, 65, .1, False),
              (100, 4, 5, .1, False)])
    for d, rank, n, std, old in cases:
        g = random_tt(d, rank, n)
        y = jnp.asarray(np.random.default_rng(8).uniform(-1, 1, (8, d)))
        op = GaussianConvolution.from_tt(g, [std]*d)
        reference = GaussianConvolution.from_tt(g, [std]*d,
                                                quadrature_order=np.array(op.orders)*2,
                                                positive=True)
        f = jax.jit(lambda p, op=op, g=g, y=y: op.log_density(rebuild_tt(g, p), y))
        params = tt_params(g)
        result = np.asarray(f(params))
        tic = time.monotonic()
        for _ in range(5):
            f(params).block_until_ready()
        seconds = (time.monotonic()-tic)/5
        truth = np.asarray(jax.jit(lambda p, reference=reference, g=g, y=y: reference.log_density(rebuild_tt(g, p), y))(params))
        grad = jax.jit(jax.grad(lambda p, f=f: -f(p).mean()))(params)
        row = {'dimension': d, 'rank': rank, 'n_coeffs': n, 'noise_std': std,
               'orders': op.orders, 'direct_seconds': seconds,
               'direct_max_abs_log_error': float(np.max(np.abs(result-truth))),
               'finite_gradient': all(np.all(np.isfinite(v)) for v in grad)}
        if old:
            row['spectral'] = []
            for kernel_n in (65, 129, 257):
                u = gaussian_kernel(d, std, kernel_n)
                spectral = jax.jit(lambda p, g=g, u=u, y=y: log_density_noisy_tt(rebuild_tt(g, p), u, y))
                estimate = np.asarray(spectral(params))
                tic = time.monotonic()
                for _ in range(5):
                    spectral(params).block_until_ready()
                row['spectral'].append({'kernel_n': kernel_n,
                                        'seconds': (time.monotonic()-tic)/5,
                                        'max_abs_log_error': float(np.max(np.abs(estimate-truth)))})
        rows.append(row)
        print(json.dumps(row), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2)+'\n')
        jax.clear_caches()


if __name__ == '__main__':
    main()
