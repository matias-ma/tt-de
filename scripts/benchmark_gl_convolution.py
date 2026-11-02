"""Noisy GL validation of RSI and direct convolution, with data-only NLL training.

The true GL density is evaluated only after checkpoint selection. Reports
both the repository's squared relative sample error and its square root;
neither is an unweighted Lebesgue-space L2 integral (test points follow p).

Training uses the same recipe validated by benchmark_gl_covariance.py: Gram/
Cholesky-preconditioned coefficients (so every Chebyshev degree gets a
comparable effective step size instead of the raw basis's poorly conditioned
one) and a cosine-decayed learning rate, plus ``TTConvolution.prepare()`` to
precompute integration moments once so more steps fit in the same budget.
Without these, the deconvolved fit was observed to pick an early checkpoint
and then degrade (see docs/gl_convolution_followup.md) -- an optimization/
regularization gap, not a forward-convolution accuracy problem (already
checked to floating-point precision in docs/convolution_research.md).
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from chebttge import convolve_tt
from gl_common import dim_pairs_for, robust_domain, save_noisy_marginal_figure
from gl_gpu_noisy import NOISE_STDS, NOISE_WEIGHTS, _noniso_covariance

from ftde.cheb_gram import chebyshev_gram_matrix
from ftde.cheb_product import square_tt
from ftde.gaussian_convolution import GaussianConvolution
from ftde.ginzburg_landau import (
    ginzburg_landau_density,
    ginzburg_landau_log_normalizer,
    sample_ginzburg_landau,
)
from ftde.init import init_tt_from_kde
from ftde.kde import make_gaussian_kde, select_gaussian_kde_bandwidth
from ftde.noise_kernel import (
    build_gaussian_mixture_noise_kernel_tt,
    build_pairwise_correlated_noise_kernel_tt,
)
from ftde.noisy_regularization import tikhonov_penalty_tt
from ftde.noisy_tt_density import log_density_noisy_tt
from ftde.sobolev import AmplitudeSobolevPenalty
from ftde.tt_convolution import TTConvolution
from ftde.tt_density import (
    density_tt,
    log_density_tt,
    rebuild_tt,
    squared_integral_tt,
    tt_params,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dim', type=int, default=8)
    parser.add_argument('--noise', choices=['mixture', 'paired', 'gaussian'], default='mixture')
    parser.add_argument('--noise-scale', type=float, default=1.,
                        help='multiply the prescribed noise standard deviations (default 1)')
    parser.add_argument('--n-train', type=int, default=100000)
    parser.add_argument('--n-test', type=int, default=4000)
    parser.add_argument('--rank', type=int, default=None)
    parser.add_argument('--n-coeffs', type=int, default=None)
    parser.add_argument('--steps', type=int, default=10000, help='default step count for both stages')
    parser.add_argument('--naive-steps', type=int, default=None,
                        help='override naive-fit steps (default: --steps); 0 skips training '
                             'and reports the untrained KDE-init crossapproximation itself')
    parser.add_argument('--deconv-steps', type=int, default=None,
                        help='override deconvolved-fit steps (default: --steps)')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--naive-lr', type=float, default=1e-3)
    parser.add_argument('--no-lr-schedule', action='store_true', help='disable cosine LR decay')
    parser.add_argument('--raw-coordinates', action='store_true',
                        help='disable Gram/Cholesky preconditioning of the optimized coefficients')
    parser.add_argument('--reg-lambda', type=float, default=1.)
    parser.add_argument('--regularizer', choices=['legacy', 'h1', 'h2'], default='legacy',
                        help='legacy core penalty or integrated amplitude derivatives; '
                             'set --reg-lambda explicitly when switching (e.g. 0.0001 for h2)')
    parser.add_argument('--kernel-rank', type=int, default=3,
                        help='Maximum rank of the pairwise correlated noise approximation')
    parser.add_argument('--kernel-n-coeffs', type=int, default=257,
                        help='Chebyshev coefficient count for mixture noise')
    parser.add_argument('--compare-spectral', action='store_true')
    parser.add_argument('--convolution-backend', choices=['direct', 'rsi'], default='direct')
    parser.add_argument('--rsi-max-rank', type=int, default=None,
                        help='convolution rank cap; omitted uses the library rank heuristic')
    parser.add_argument('--rsi-sketch-size', type=int, default=None,
                        help='number of random suffix projections; omitted uses the library default')
    parser.add_argument('--rsi-setup-only', action='store_true',
                        help='check initial RSI accuracy and exit before training')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--figure-dir', type=Path, default=Path('figs/ginzburg_landau'))
    parser.add_argument('--save-figures', action='store_true',
                        help='save the selected deconvolved fit to figs/ginzburg_landau')
    args = parser.parse_args()
    if not np.isfinite(args.noise_scale) or args.noise_scale <= 0:
        parser.error('--noise-scale must be positive and finite')
    if args.convolution_backend == 'rsi' and args.noise == 'gaussian':
        parser.error('RSI requires a prescribed TT kernel; choose mixture or paired noise')
    if args.rsi_setup_only and args.convolution_backend != 'rsi':
        parser.error('--rsi-setup-only requires --convolution-backend rsi')
    d = args.dim
    rank = args.rank or (8 if d <= 8 else 10 if d <= 14 else 14)
    n = args.n_coeffs or (17 if d <= 8 else 19 if d <= 14 else 21)
    naive_steps = args.naive_steps if args.naive_steps is not None else args.steps
    deconv_steps = args.deconv_steps if args.deconv_steps is not None else args.steps
    start = time.monotonic()

    def observations(seed, size):
        clean = np.asarray(sample_ginzburg_landau(seed=seed, n=size, d=d))
        rng = np.random.default_rng(seed)
        if args.noise == 'mixture':
            comp = rng.choice(2, size=size, p=np.asarray(NOISE_WEIGHTS))
            noise = rng.normal(size=(size, d))*np.asarray(NOISE_STDS)[comp, None]*args.noise_scale
        elif args.noise == 'paired':
            noise = rng.multivariate_normal(np.zeros(d), _noniso_covariance(d, args.noise_scale), size=size)
        else:
            noise = rng.normal(size=(size, d))*.55*args.noise_scale
        return jnp.asarray(clean+noise)

    train = observations(0, args.n_train)
    valid = observations(2, 2048)
    domain = robust_domain(train)
    # Keep all noisy observations for convolved NLL, even outside the signal box.
    mask = jnp.all((train >= domain[:, 0]) & (train <= domain[:, 1]), axis=1)
    naive_train = train[mask]
    bandwidth = select_gaussian_kde_bandwidth(train)
    fit_start = time.monotonic()
    init = init_tt_from_kde(train, max_rank=rank, domain=domain, n_coeffs=(n,),
                            bandwidth=bandwidth, n_starts=64, gd_steps=10)
    print('initialized', d, args.noise, init.ranks, 'domain', domain[0], flush=True)
    u = None
    if args.noise == 'mixture':
        u = build_gaussian_mixture_noise_kernel_tt(NOISE_STDS*args.noise_scale, NOISE_WEIGHTS, domain, d,
                                                   n_coeffs=args.kernel_n_coeffs)
        op = TTConvolution.from_tt(init, u)
        std = jnp.full(d, .3*args.noise_scale)
    elif args.noise == 'paired':
        covariance = _noniso_covariance(d, args.noise_scale)
        std = jnp.sqrt(jnp.diag(jnp.asarray(covariance)))
        u = build_pairwise_correlated_noise_kernel_tt(float(std[0]), .5, domain, d,
                                                     max_rank=args.kernel_rank, n_starts=64, gd_steps=10)
        op = TTConvolution.from_tt(init, u)
    else:
        std = jnp.full(d, .55*args.noise_scale)
        op = GaussianConvolution.from_tt(init, std)
    print('operator orders', op.orders, 'kernel ranks', None if u is None else u.ranks, flush=True)
    if u is not None:
        for data in (train, valid):
            if not bool(jnp.all((data >= op.safe_domain[:, 0]) & (data <= op.safe_domain[:, 1]))):
                raise ValueError('kernel box does not cover all noisy observations')

    unprepared = op
    observation_domain = None
    if isinstance(op, TTConvolution):
        # Precompute integration moments once (accuracy-neutral -- it is the
        # same closed-form moments computed on the fly, just cached) so more
        # optimizer steps fit in the same wall-clock budget; see
        # ftde/tt_convolution.py's TTConvolution.prepare docstring.
        all_obs = np.concatenate([np.asarray(train), np.asarray(valid)])
        observation_domain = np.stack([all_obs.min(axis=0)-.05, all_obs.max(axis=0)+.05], axis=1)
        print('preparing convolution moments...', flush=True)
        tic = time.monotonic()
        op = op.prepare(observation_domain, tolerance=1e-14)
        print(f'  prepared in {time.monotonic()-tic:.1f}s', flush=True)

    rsi_rows = None
    rsi_setup = None
    if args.convolution_backend == 'rsi':
        if u is None:
            raise ValueError('RSI requires a prescribed TT kernel; choose mixture or paired noise')
        if args.rsi_setup_only:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.output.with_suffix('.npz'), domain=np.asarray(domain),
                     kernel_domain=np.asarray(u.domain), observation_domain=observation_domain,
                     probe=np.asarray(valid[:256]),
                     **{f'signal_{k}': np.asarray(c.coeffs) for k, c in enumerate(init.cores)},
                     **{f'kernel_{k}': np.asarray(c.coeffs) for k, c in enumerate(u.cores)})
        tic = time.monotonic()
        initial_rsi = convolve_tt(square_tt(init), u, output_domain=observation_domain,
                                  max_rank=args.rsi_max_rank, sketch_size=args.rsi_sketch_size, validation_samples=64)
        rsi_rows = tuple(stage['pivot_rows'] for stage in initial_rsi.diagnostics['stages'])
        rsi_setup = {'seconds': time.monotonic()-tic,
                     'ranks': initial_rsi.ranks,
                     'validation_relative_rms_error': initial_rsi.diagnostics['validation_relative_rms_error'],
                     'validation_max_error': initial_rsi.diagnostics['validation_max_error']}
        print('RSI setup', rsi_setup, flush=True)
        rsi_setup['pivot_rows'] = rsi_rows
        rsi_setup['sketch_size'] = initial_rsi.diagnostics['sketch_size']
        if args.rsi_setup_only:
            exact = np.asarray(op.log_density(init, valid[:256]))
            approximate = np.log(np.maximum(np.asarray(initial_rsi(valid[:256])), 1e-300)) - np.log(float(squared_integral_tt(init)))
            rsi_setup['probe_max_abs_log_error'] = float(np.max(np.abs(exact-approximate)))
            rsi_setup['stages'] = initial_rsi.diagnostics['stages']
            args.output.write_text(json.dumps(rsi_setup, indent=2)+'\n')
            print('Initial observation log error', rsi_setup['probe_max_abs_log_error'], flush=True)
            return

    # Gram/Cholesky preconditioning: optimize c' = c @ chol(gram) instead of
    # the raw Chebyshev coefficients c, so every degree's coefficient has a
    # comparable effect on the L2 norm of g -- this is what let
    # benchmark_gl_covariance.py train reliably at higher degree/longer
    # schedules without the early-checkpoint degradation seen without it.
    factors = [np.linalg.cholesky(np.asarray(chebyshev_gram_matrix(n, tuple(box)))) for box in domain]
    inverse = [jnp.asarray(np.linalg.inv(f)) for f in factors]
    factors = [jnp.asarray(f) for f in factors]

    def to_precond(params):
        return tuple(c@f for c, f in zip(params, factors, strict=True))

    def rebuild(p):
        coeff = p if args.raw_coordinates else tuple(c@f for c, f in zip(p, inverse, strict=True))
        return rebuild_tt(init, coeff)

    def logp(p, x, label):
        g = rebuild(p)
        if label == 'naive':
            return log_density_tt(g, x)
        if args.convolution_backend == 'rsi':
            return log_density_noisy_tt(
                g, u, x, output_domain=observation_domain, convolution_method='rsi',
                rsi_max_rank=args.rsi_max_rank, rsi_sketch_size=args.rsi_sketch_size, rsi_pivot_rows=rsi_rows)
        return op.log_density(g, x)

    regularizer = (None if args.regularizer == 'legacy' else
                   AmplitudeSobolevPenalty.from_tt(init, order=int(args.regularizer[1]), length_scales=std))

    def objective(p, x, label):
        loss = -jnp.mean(logp(p, x, label))
        if label == 'deconvolved' and args.reg_lambda:
            g = rebuild(p)
            penalty = (regularizer(g) if regularizer is not None else
                       tikhonov_penalty_tt(square_tt(g), std, per_axis_target_budget=.5))
            loss = loss + args.reg_lambda*penalty
        return loss

    def make_optimizer(steps, lr):
        schedule = lr if args.no_lr_schedule or steps <= 0 else optax.cosine_decay_schedule(lr, steps, alpha=.1)
        return optax.chain(optax.clip_by_global_norm(1.), optax.adam(schedule))

    init_params = tt_params(init)
    histories, fits = {}, {}
    for label, steps, lr in (('naive', naive_steps, args.naive_lr), ('deconvolved', deconv_steps, args.lr)):
        params = init_params if args.raw_coordinates else to_precond(init_params)
        optimizer = make_optimizer(steps, lr)
        state = optimizer.init(params)

        def step_fn(p, s, x, label=label, optimizer=optimizer):
            loss, grad = jax.value_and_grad(lambda pp: objective(pp, x, label))(p)
            updates, s = optimizer.update(grad, s, p)
            return optax.apply_updates(p, updates), s, loss

        step_fn = jax.jit(step_fn)
        evaluate = jax.jit(lambda p, x, label=label: logp(p, x, label))
        if label == 'naive':
            training = naive_train
            inside = jnp.all((valid >= domain[:, 0]) & (valid <= domain[:, 1]), axis=1)
            validation = valid[inside][:1792]
        else:
            training, validation = train, valid
        key = jax.random.PRNGKey(0)
        best, best_nll, best_step, history = params, np.inf, 0, []
        nonfinite_step = None
        tic = time.monotonic()
        for i in range(steps+1):
            if i % 200 == 0 or i == steps:
                logp_vals = np.concatenate([np.asarray(evaluate(params, validation[j:j+256]))
                                            for j in range(0, len(validation), 256)])
                nll = -float(logp_vals.mean())
                if nll < best_nll:
                    best, best_nll, best_step = params, nll, i
                history.append({'step': i, 'validation_nll': nll})
                print(label, i, 'nll', nll, 'seconds', time.monotonic()-tic, flush=True)
            if i == steps:
                break
            key, sub = jax.random.split(key)
            batch = training[jax.random.randint(sub, (args.batch_size,), 0, len(training))]
            params, state, loss = step_fn(params, state, batch)
            if not np.isfinite(float(loss)):
                nonfinite_step = i
                print(f'{label}: nonfinite loss at {i}; selecting best finite validation checkpoint',
                      flush=True)
                break
        fits[label] = rebuild(best)
        histories[label] = {'best_step': best_step, 'best_nll': best_nll,
                            'nonfinite_step': nonfinite_step,
                            'training_seconds': time.monotonic()-tic, 'history': history}
    fit_seconds = time.monotonic()-fit_start
    probe = valid[:32]
    refined = (TTConvolution.from_tt(init, u, quadrature_order=np.array(unprepared.orders)*2) if u is not None else
               GaussianConvolution.from_tt(init, std, quadrature_order=np.array(op.orders)*2))
    logp_check = np.asarray(op.log_density(fits['deconvolved'], probe))
    error = float(np.max(np.abs(logp_check-np.asarray(refined.log_density(fits['deconvolved'], probe)))))
    checks = {'quadrature_max_abs_log_error': error}
    if rsi_rows is not None:
        exact_logp = np.asarray(op.log_density(fits['deconvolved'], valid[:256]))
        rsi_logp = np.asarray(log_density_noisy_tt(
            fits['deconvolved'], u, valid[:256], output_domain=observation_domain,
            convolution_method='rsi', rsi_max_rank=args.rsi_max_rank,
            rsi_sketch_size=args.rsi_sketch_size, rsi_pivot_rows=rsi_rows))
        checks['rsi_max_abs_log_error'] = float(np.max(np.abs(rsi_logp-exact_logp)))
        checks['rsi_relative_log_error'] = float(np.linalg.norm(rsi_logp-exact_logp)/np.linalg.norm(exact_logp))
        checks['exact_validation_nll_at_rsi_checkpoint'] = float(-np.mean(np.asarray(
            op.log_density(fits['deconvolved'], valid))))
    if args.noise == 'mixture':
        components = [GaussianConvolution.from_tt(init, jnp.full(d, float(s)*args.noise_scale)) for s in NOISE_STDS]
        logs = jnp.stack([c.log_density(fits['deconvolved'], probe) for c in components])
        exact_noise_logp = jax.scipy.special.logsumexp(logs+jnp.log(NOISE_WEIGHTS)[:, None], axis=0)
        checks['analytic_mixture_max_abs_log_error'] = float(np.max(np.abs(logp_check-np.asarray(exact_noise_logp))))
    if args.noise == 'paired':
        covariance = _noniso_covariance(d, args.noise_scale)
        noise_probe = np.random.default_rng(5).multivariate_normal(np.zeros(d), covariance, size=2000)
        actual_kernel = np.exp(-.5*np.einsum('bi,ij,bj->b', noise_probe, np.linalg.inv(covariance), noise_probe)
                               -.5*(d*np.log(2*np.pi)+np.linalg.slogdet(covariance)[1]))
        represented_kernel = np.asarray(u(jnp.asarray(noise_probe)))
        checks['kernel_relative_sample_l2'] = float(np.linalg.norm(represented_kernel-actual_kernel)/np.linalg.norm(actual_kernel))
        checks['kernel_negative_fraction'] = float(np.mean(represented_kernel < 0))
    if args.compare_spectral and u is not None:
        inside = jnp.all((valid >= domain[:, 0]) & (valid <= domain[:, 1]), axis=1)
        p = valid[inside][:16]
        checks['spectral_max_abs_log_error'] = float(jnp.max(jnp.abs(
            op.log_density(fits['deconvolved'], p)-log_density_noisy_tt(fits['deconvolved'], u, p))))
    # Only now evaluate the true density: never used in fitting or selection.
    clean = sample_ginzburg_landau(seed=1, n=args.n_test, d=d)
    truth = np.asarray(ginzburg_landau_density(clean, log_z=ginzburg_landau_log_normalizer(d)))
    estimates = {'kde': np.asarray(make_gaussian_kde(train, bandwidth)(clean))}
    support = np.all((np.asarray(clean) >= np.asarray(domain)[:, 0]) &
                     (np.asarray(clean) <= np.asarray(domain)[:, 1]), axis=1)
    for label, g in fits.items():
        estimates[label] = np.where(support, np.asarray(density_tt(g, clean)), 0.)
    metrics = {}
    for label, pred in estimates.items():
        squared = float(np.sum((pred-truth)**2)/np.sum(truth**2))
        metrics[label] = {'relative_squared_sample_error': squared, 'relative_sample_l2': np.sqrt(squared)}
    result = {**vars(args), 'output': str(args.output), 'figure_dir': str(args.figure_dir), 'rank': rank, 'n_coeffs': n,
              'naive_steps': naive_steps, 'deconv_steps': deconv_steps,
              'metrics': metrics, 'checks': checks, 'histories': histories,
              'rsi_setup': rsi_setup,
              'orders': unprepared.orders, 'fit_seconds': fit_seconds,
              'seconds': time.monotonic()-start}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    np.savez(args.output.with_suffix('.npz'), domain=np.asarray(domain),
             **{f'{label}_{k}': np.asarray(c.coeffs) for label, g in fits.items() for k, c in enumerate(g.cores)})
    if args.save_figures:
        if u is None:
            raise ValueError('marginal figures require a prescribed TT noise kernel')
        log_z = ginzburg_landau_log_normalizer(d)
        for i, j in dim_pairs_for(d):
            save_noisy_marginal_figure(
                d, fits['naive'], fits['deconvolved'], u, clean, train, bandwidth, domain, i, j,
                log_z=log_z,
                kde_error=metrics['kde']['relative_squared_sample_error'],
                naive_error=metrics['naive']['relative_squared_sample_error'],
                deconv_error=metrics['deconvolved']['relative_squared_sample_error'],
                naive_steps=naive_steps,
                deconv_steps=histories['deconvolved']['nonfinite_step'] or deconv_steps,
                selected_step=histories['deconvolved']['best_step'],
                n_density_points=len(train),
                noise_type=args.noise, naive_rank=rank, deconv_rank=rank,
                degree=n-1, sigma=args.noise_scale, fit_seconds=fit_seconds,
                figure_dir=args.figure_dir,
                deconv_label='TT with RSI deconvolution' if args.convolution_backend == 'rsi' else 'Deconvolved TT',
            )
    print(json.dumps({'metrics': metrics, 'checks': checks}, indent=2), flush=True)


if __name__ == '__main__':
    main()
