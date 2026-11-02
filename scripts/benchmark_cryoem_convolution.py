"""Compare forward operators with identical data, initialization and NLL.

Labels are read only for final evaluation. Checkpoints are selected using
held-out noisy likelihood. Outputs include all settings and operator errors.
Run in chebdGE on a GPU allocation; see docs/convolution_research.md.
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from chebttge.computations import convolve_tt
from cryoem_common import robust_domain
from cryoem_path_density import path_anchors, relative_l2_error, true_path_probability

from ftde.gaussian_convolution import GaussianConvolution
from ftde.cheb_product import square_tt
from ftde.init import init_tt_from_kde
from ftde.kde import make_gaussian_kde
from ftde.noise_kernel import build_diagonal_gaussian_noise_kernel_tt
from ftde.noisy_regularization import DEFAULT_REG_TARGET_BUDGET
from ftde.noisy_tt_density import log_density_noisy_tt, make_noisy_train_step
from ftde.tt_density import (
    density_tt,
    log_density_tt,
    make_train_step,
    rebuild_tt,
    tt_params,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dim', type=int, default=10)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--degree', type=int, default=3, help='number of Chebyshev coefficients')
    parser.add_argument('--steps', type=int, default=1500)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--reg-lambda', type=float, default=0.)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--backend', choices=['gaussian', 'spectral', 'rsi', 'naive'], default='gaussian')
    parser.add_argument('--rsi-max-rank', type=int, default=64)
    parser.add_argument('--rsi-sketch-size', type=int, default=None,
                        help='number of random suffix projections; omitted uses the library default')
    parser.add_argument('--kernel-n-coeffs', type=int, default=257)
    parser.add_argument('--compare-spectral', action='store_true')
    parser.add_argument('--noise-eigenbasis', action='store_true',
                        help='rotate coordinates to represent the full prescribed covariance exactly')
    parser.add_argument('--keep-outside', action='store_true',
                        help='fit all noisy observations, including those outside the signal box')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    cache = np.load(f'cryoem_data/cache/path_noise1_dim{args.dim}.npz')
    raw = np.asarray(cache['points'])
    covariance = np.asarray(cache['cov_mean'])
    physical_raw = raw.copy()
    rotation = np.eye(args.dim)
    if args.noise_eigenbasis:
        if args.backend != 'gaussian' or not args.keep_outside:
            raise ValueError('--noise-eigenbasis requires gaussian and --keep-outside')
        values, vectors = np.linalg.eigh(covariance)
        rotation = vectors[:, ::-1]
        signs = np.sign(rotation[np.argmax(np.abs(rotation), axis=0), np.arange(args.dim)])
        rotation = rotation * signs
        raw = raw @ rotation
        covariance = np.diag(values[::-1])
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(raw))
    n_val = len(raw) // 5
    train_raw, valid_raw = raw[indices[n_val:]], raw[indices[:n_val]]
    domain = np.asarray(robust_domain(jnp.asarray(train_raw)))
    # Reproduce the existing box/trimming protocol. This comparison is
    # conditional on this data selection; it is not a full-tail fit.
    inside = lambda x: np.all((x >= domain[:, 0]) & (x <= domain[:, 1]), axis=1)
    if args.keep_outside and args.backend != "gaussian":
        raise ValueError("--keep-outside requires the Gaussian backend")
    train = jnp.asarray(train_raw if args.keep_outside else train_raw[inside(train_raw)])
    valid = jnp.asarray((valid_raw if args.keep_outside else valid_raw[inside(valid_raw)])[:4096])
    init = init_tt_from_kde(train, max_rank=args.rank, domain=jnp.asarray(domain),
                            n_coeffs=(args.degree,), n_starts=64, gd_steps=10,
                            random_seed=args.seed)
    print('init', init.ranks, 'train', train.shape, 'valid', valid.shape, flush=True)
    std = jnp.sqrt(jnp.diag(jnp.asarray(covariance)))
    op = GaussianConvolution.from_tt(init, std)
    refined = GaussianConvolution.from_tt(init, std, quadrature_order=np.array(op.orders)*2)
    print('quadrature orders', op.orders, flush=True)
    u = None
    if args.backend in ('spectral', 'rsi') or args.compare_spectral:
        u = build_diagonal_gaussian_noise_kernel_tt(std**2, init.domain, n_coeffs=(args.kernel_n_coeffs,))
    rsi_rows = None
    rsi_setup = None
    if args.backend == 'rsi':
        initial = convolve_tt(square_tt(init), u, method='rsi', max_rank=args.rsi_max_rank,
                              sketch_size=args.rsi_sketch_size, validation_samples=64)
        rsi_rows = tuple(stage['pivot_rows'] for stage in initial.diagnostics['stages'])
        rsi_setup = {'ranks': initial.ranks,
                     'relative_rms_error': initial.diagnostics['validation_relative_rms_error']}
        print('RSI setup', rsi_setup, flush=True)
    operator = op if args.backend == 'gaussian' else None
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(args.lr))
    if args.backend == 'naive':
        step = make_train_step(init, optimizer)
    else:
        step = make_noisy_train_step(init, u, optimizer, gaussian_convolution=operator,
                                    convolution_method='rsi' if args.backend == 'rsi' else 'naive',
                                    rsi_max_rank=args.rsi_max_rank, rsi_pivot_rows=rsi_rows,
                                    rsi_sketch_size=args.rsi_sketch_size,
                                    noise_std=std, reg_strength=args.reg_lambda,
                                    per_axis_target_budget=DEFAULT_REG_TARGET_BUDGET)
    # Validate each observation model: naive p(y), deconvolution (U*p)(y).
    if args.backend == "naive":
        evaluate = jax.jit(lambda p, x: log_density_tt(rebuild_tt(init, p), x))
    else:
        evaluate = jax.jit(lambda p, x: op.log_density(rebuild_tt(init, p), x))
    params = tt_params(init)
    state = optimizer.init(params)
    best_params = params
    best_nll = np.inf
    best_step = 0
    history = []
    key = jax.random.PRNGKey(args.seed)
    train_start = time.monotonic()
    for i in range(args.steps + 1):
        if i % 100 == 0 or i == args.steps:
            scores = np.concatenate([np.asarray(evaluate(params, valid[j:j+256]))
                                     for j in range(0, len(valid), 256)])
            nll = float(-scores.mean())
            if nll < best_nll:
                best_params, best_nll, best_step = params, nll, i
            history.append({'step': i, 'validation_nll': nll})
            print(i, 'validation_nll', nll, 'seconds', time.monotonic()-train_start, flush=True)
        if i == args.steps:
            break
        key, sub = jax.random.split(key)
        idx = jax.random.randint(sub, (args.batch_size,), 0, len(train))
        params, state, loss = step(params, state, train[idx])
        if not np.isfinite(float(loss)):
            raise FloatingPointError(f'Nonfinite loss at step {i}: {loss}')
    training_seconds = time.monotonic() - train_start
    fitted = rebuild_tt(init, best_params)
    probe = valid[:64]
    actual = np.asarray(op.log_density(fitted, probe))
    reference = np.asarray(refined.log_density(fitted, probe))
    errors = {'quadrature_max_abs_log_error': float(np.max(np.abs(actual-reference)))}
    if u is not None:
        spectral = np.asarray(log_density_noisy_tt(fitted, u, probe))
        errors['spectral_max_abs_log_error'] = float(np.max(np.abs(spectral-reference)))
    if rsi_rows is not None:
        rsi = np.asarray(log_density_noisy_tt(fitted, u, probe, convolution_method='rsi',
                                             rsi_max_rank=args.rsi_max_rank, rsi_sketch_size=args.rsi_sketch_size,
                                             rsi_pivot_rows=rsi_rows))
        errors['rsi_max_abs_log_error_vs_spectral'] = float(np.max(np.abs(rsi-spectral)))
        errors['rsi_max_abs_log_error_vs_gaussian'] = float(np.max(np.abs(rsi-reference)))
    # First and only access to labels: final validation, never training or selection.
    labels = np.asarray(cache['labels'])
    truth, n_states = true_path_probability(labels)
    anchors = path_anchors(raw, labels, n_states)
    estimated = np.asarray(density_tt(fitted, jnp.asarray(anchors)))
    physical_train = physical_raw[indices[n_val:]]
    if not args.keep_outside:
        physical_train = physical_train[inside(train_raw)]
    physical_anchors = path_anchors(physical_raw, labels, n_states)
    kde_values = np.asarray(make_gaussian_kde(jnp.asarray(physical_train))(jnp.asarray(physical_anchors)))
    result = {**vars(args), 'output': str(args.output), 'ranks': list(init.ranks),
              'n_train': len(train), 'n_validation': len(valid),
              'best_step': best_step, 'best_validation_nll': best_nll,
              'path_relative_l2': relative_l2_error(estimated, truth),
              'path_relative_l2_supported': relative_l2_error(np.where(inside(anchors), estimated, 0.), truth),
              'anchors_outside_signal_box': np.flatnonzero(~inside(anchors)).tolist(),
              'kde_path_relative_l2': relative_l2_error(kde_values, truth),
              'training_seconds': training_seconds, 'total_seconds': time.monotonic()-start,
              'quadrature_orders': op.orders, 'errors': errors, 'history': history, 'rsi_setup': rsi_setup,
              'path_estimate': (estimated/estimated[1:-1].sum()).tolist(),
              'path_truth': truth.tolist(),
              'path_kde': (kde_values/kde_values[1:-1].sum()).tolist()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    np.savez(args.output.with_suffix('.npz'), domain=domain, rotation=rotation,
             **{f'core_{k}': np.asarray(c) for k, c in enumerate(best_params)})
    print(json.dumps({k: v for k, v in result.items() if k not in
                      ('history', 'path_estimate', 'path_truth', 'path_kde')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
