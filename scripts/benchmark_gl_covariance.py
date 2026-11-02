"""Data-only noisy GL experiment with a non-Gaussian covariance mixture."""

import argparse
import fcntl
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from gl_common import robust_domain
from gl_covariance_common import (
    build_noise_tt,
    draw_noise,
    load_record,
    load_tt,
    noise_density,
    noise_spec,
    save_record,
    save_tt,
)

from ftde.cheb_gram import chebyshev_gram_matrix, chebyshev_moment_vector
from ftde.ginzburg_landau import sample_ginzburg_landau
from ftde.init import init_tt_from_kde
from ftde.kde import select_gaussian_kde_bandwidth
from ftde.sobolev import amplitude_h1, derivative_grams
from ftde.tt_convolution import TTConvolution
from ftde.tt_density import log_density_tt, rebuild_tt, tt_params


def setup(directory, d, count, noise_scale):
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory/'data.npz'
    if cache.exists():
        saved = np.load(cache)
        saved_scale = float(saved['noise_scale']) if 'noise_scale' in saved else 1.
        if saved['train'].shape != (count, d) or saved_scale != noise_scale:
            raise ValueError('Use a separate directory for a different dataset size, dimension, or noise scale')
        return
    weights, covs = noise_spec(d, noise_scale)
    print('sampling training and validation data', flush=True)
    train = np.asarray(sample_ginzburg_landau(10, count, d))+draw_noise(110, count, weights, covs)[0]
    valid = np.asarray(sample_ginzburg_landau(20, 6000, d))+draw_noise(120, 6000, weights, covs)[0]
    domain = np.asarray(robust_domain(jnp.asarray(train)))
    all_obs = np.concatenate([train, valid])
    obs_domain = np.stack([all_obs.min(axis=0)-.05, all_obs.max(axis=0)+.05], axis=1)
    kernel_domain = np.stack([obs_domain[:, 0]-domain[:, 1]-.05,
                              obs_domain[:, 1]-domain[:, 0]+.05], axis=1)
    print('building noise TT', flush=True)
    u, tails = build_noise_tt(weights, covs, kernel_domain)
    mass = jnp.ones(1)
    for c in u.cores:
        mass = mass@jnp.einsum('abn,n->ab', c.coeffs, chebyshev_moment_vector(c.coeffs.shape[-1], c.domain))
    mass = float(mass[0])
    if not np.isfinite(mass) or mass <= 0:
        raise FloatingPointError('Noise TT has invalid mass; increase kernel resolution/rank')
    params = list(tt_params(u)); params[0] = params[0]/mass
    u = rebuild_tt(u, params)
    noise, _ = draw_noise(150, 4000, weights, covs)
    truth = noise_density(noise, weights, covs)
    approx = np.asarray(u(jnp.asarray(noise)))
    checks = {'kernel_relative_sample_l2': float(np.linalg.norm(approx-truth)/np.linalg.norm(truth)),
              'kernel_negative_fraction': float(np.mean(approx < 0)), 'kernel_mass_before_normalization': mass,
              'largest_discarded_svd_relative_norm': max(tails), 'kernel_ranks': list(u.ranks)}
    if checks['kernel_relative_sample_l2'] > 1e-5 or checks['kernel_negative_fraction'] > 0:
        raise FloatingPointError(f'Noise TT failed its accuracy check: {checks}')
    save_tt(directory/'noise.npz', u)
    temporary_cache = directory/'data.build.npz'
    np.savez(temporary_cache, train=train, valid=valid, domain=domain, observation_domain=obs_domain,
             weights=weights, covs=covs, noise_scale=noise_scale)
    temporary_cache.replace(cache)
    print('setup complete', checks, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('artifacts/gl_covariance_mixture'))
    parser.add_argument('--dim', type=int, default=10)
    parser.add_argument('--n-train', type=int, default=100000)
    parser.add_argument('--noise-scale', type=float, default=1.,
                        help='multiply every prescribed mixture component standard deviation')
    parser.add_argument('--setup-only', action='store_true')
    parser.add_argument('--name', default='r8_n21')
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--n-coeffs', type=int, default=21)
    parser.add_argument('--steps', type=int, default=8000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--smoothness', type=float, default=0.)
    parser.add_argument('--naive', action='store_true')
    parser.add_argument('--direct', action='store_true', help='Unprepared direct quadrature reference')
    parser.add_argument('--raw-coordinates', action='store_true', help='Disable polynomial Gram preconditioning')
    args = parser.parse_args()
    setup(args.directory, args.dim, args.n_train, args.noise_scale)
    if args.setup_only:
        return
    run = args.directory/args.name
    run.mkdir(exist_ok=True)
    # Permit independent experiment launchers without duplicating a run.
    run_lock = (run/'.lock').open('a')
    fcntl.flock(run_lock, fcntl.LOCK_EX)
    if (run/'result.npz').exists():
        saved = load_record(run/'result.npz')
        settings = {**vars(args), 'directory': str(args.directory)}
        if any(saved.get(k) != value for k, value in settings.items()):
            raise ValueError('Run name already has different settings; choose another --name')
        print('reusing completed run', args.name, flush=True)
        return
    data = np.load(args.directory/'data.npz')
    train, valid, domain = data['train'], data['valid'], data['domain']
    n = args.n_coeffs
    initial = args.directory/f'init_r{args.rank}_n{n}.npz'
    if initial.exists():
        init = load_tt(initial)
    else:
        # Initialization is a data-only computational shortcut; all 100k
        # observations enter the subsequent likelihood optimization.
        subset = jnp.asarray(train[np.random.default_rng(50).choice(len(train), min(10000, len(train)), replace=False)])
        bandwidth = select_gaussian_kde_bandwidth(subset)
        init = init_tt_from_kde(subset, max_rank=args.rank, domain=jnp.asarray(domain), n_coeffs=(n,),
                               bandwidth=bandwidth, n_starts=48, gd_steps=8)
        save_tt(initial, init)
    u = load_tt(args.directory/'noise.npz')
    direct = TTConvolution.from_tt(init, u)
    print('preparing moments', args.name, flush=True)
    tic = time.monotonic()
    op = direct if args.direct or args.naive else direct.prepare(data['observation_domain'], tolerance=1e-14)
    preparation = time.monotonic()-tic
    print('prepared', preparation, 'orders', None if op.moment_cores is None else [c.coeffs.shape[-1] for c in op.moment_cores], flush=True)
    factors = [np.linalg.cholesky(np.asarray(chebyshev_gram_matrix(n, tuple(box)))) for box in domain]
    inverse = [jnp.asarray(np.linalg.inv(f)) for f in factors]
    params = tt_params(init)
    if not args.raw_coordinates:
        params = tuple(c@jnp.asarray(f) for c, f in zip(params, factors, strict=True))

    def rebuild(p):
        coeff = p if args.raw_coordinates else tuple(c@f for c, f in zip(p, inverse, strict=True))
        return rebuild_tt(init, coeff)

    def logp(p, x):
        g = rebuild(p)
        return log_density_tt(g, x) if args.naive else op.log_density(g, x)

    derivatives = derivative_grams(init)
    noise_weights = jnp.asarray(np.einsum('c,cii->i', data['weights'], data['covs']))

    def objective(p, x):
        loss = -jnp.mean(logp(p, x))
        if args.smoothness:
            loss += args.smoothness*amplitude_h1(rebuild(p), direct.grams, derivatives, noise_weights)
        return loss

    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(
        optax.cosine_decay_schedule(args.lr, args.steps, alpha=.1)))
    state = optimizer.init(params)

    @jax.jit
    def step(p, state, x):
        loss, grad = jax.value_and_grad(objective)(p, x)
        updates, state = optimizer.update(grad, state, p)
        return optax.apply_updates(p, updates), state, loss

    evaluate = jax.jit(logp)
    checks = {}
    if not args.naive:
        probe = jnp.asarray(valid[:8])
        direct_eval = jax.jit(lambda p: direct.log_density(rebuild(p), probe))
        actual = np.asarray(direct_eval(params))
        represented = np.asarray(evaluate(params, probe))
        checks['initial_direct_max_log_difference'] = float(np.max(np.abs(actual-represented)))
        a = jax.grad(lambda p: jnp.mean(direct_eval(p)))(params)
        b = jax.grad(lambda p: jnp.mean(evaluate(p, probe)))(params)
        checks['initial_gradient_relative_difference'] = float(np.sqrt(sum(float(jnp.sum((aa-bb)**2)) for aa, bb in zip(a, b, strict=True))/sum(float(jnp.sum(aa**2)) for aa in a)))
        print('checks', checks, flush=True)
    if args.naive:
        train = train[np.all((train >= domain[:, 0]) & (train <= domain[:, 1]), axis=1)]
        valid = valid[np.all((valid >= domain[:, 0]) & (valid <= domain[:, 1]), axis=1)]
    train = jnp.asarray(train)
    history, best, best_nll, best_step = [], params, np.inf, 0
    rng = np.random.default_rng(60)
    start = time.monotonic()
    for i in range(args.steps+1):
        if i % 500 == 0 or i == args.steps:
            scores = np.concatenate([np.asarray(evaluate(params, jnp.asarray(valid[j:j+128])))
                                     for j in range(0, len(valid), 128)])
            nll = -float(scores.mean())
            history.append({'step': i, 'validation_nll': nll, 'seconds': time.monotonic()-start})
            save_tt(run/f'step_{i:06d}.npz', rebuild(params))
            if nll < best_nll:
                best, best_nll, best_step = params, nll, i
                save_tt(run/'best.npz', rebuild(best))
            print(args.name, i, nll, 'seconds', time.monotonic()-start, flush=True)
        if i == args.steps:
            break
        batch = train[rng.integers(0, len(train), args.batch_size)]
        params, state, loss = step(params, state, batch)
        if not np.isfinite(float(loss)):
            raise FloatingPointError(f'nonfinite training loss at step {i}')
    if not args.naive:
        probe = jnp.asarray(valid[:8])
        checks['final_direct_max_log_difference'] = float(np.max(np.abs(
            np.asarray(jax.jit(lambda p, x: direct.log_density(rebuild(p), x))(best, probe))-np.asarray(evaluate(best, probe)))))
    result = {**vars(args), 'directory': str(args.directory), 'best_nll': best_nll, 'best_step': best_step,
              'history': history, 'checks': checks, 'training_seconds': time.monotonic()-start,
              'preparation_seconds': preparation,
              'moment_coefficients': None if op.moment_cores is None else [c.coeffs.shape[-1] for c in op.moment_cores]}
    save_record(run/'result.npz', result)


if __name__ == '__main__':
    main()
