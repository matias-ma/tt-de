"""Reproducible gallery fits. Selection sees observations and prescribed noise only.

Run one case per process to release JAX compilation memory between datasets.
Plotting and GL truth evaluation live in refresh_density_figures.py.
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from benchmark_noisy_recovery import load_model
from gl_common import data_domain, robust_domain
from gl_gpu_noisy import NOISE_STDS, NOISE_WEIGHTS, _noniso_covariance

from ftde.cheb_gram import chebyshev_gram_matrix
from ftde.fisher_visibility import FisherVisibilityPenalty
from ftde.ginzburg_landau import sample_ginzburg_landau
from ftde.init import init_tt_from_kde
from ftde.kde import select_gaussian_kde_bandwidth
from ftde.marginal_reference import fit_noisy_marginal_reference
from ftde.noise_kernel import (
    build_diagonal_gaussian_noise_kernel_tt,
    build_gaussian_mixture_noise_kernel_tt,
    build_pairwise_correlated_noise_kernel_tt,
)
from ftde.noise_marginals import noise_marginals_from_tt
from ftde.noisy_model_selection import select_by_paired_noisy_nll
from ftde.sobolev import AmplitudeSobolevPenalty
from ftde.tt_convolution import TTConvolution
from ftde.tt_density import log_density_tt, rebuild_tt, squared_integral_tt, tt_params


def save_model(path, model):
    np.savez(path, domain=np.asarray(model.domain),
             **{f'signal_{k}': np.asarray(c.coeffs) for k, c in enumerate(model.cores)})


def read_model(path):
    with np.load(path) as saved:
        return load_model(saved, 'signal', saved['domain'])


def fit_candidate(init, train, valid, destination, *, kernel=None, penalty=None,
                  strength=0., steps=8000, lr=.0005, batch_size=256, checkpoint_rule='nll', seed=0,
                  preconditioning='l2', microbatches=1):
    if batch_size % microbatches:
        raise ValueError('batch_size must be divisible by microbatches')
    if checkpoint_rule not in ('nll', 'regularized'):
        raise ValueError('checkpoint_rule must be nll or regularized')
    if preconditioning not in ('l2', 'sobolev'):
        raise ValueError('preconditioning must be l2 or sobolev')
    if preconditioning == 'sobolev' and not isinstance(penalty, AmplitudeSobolevPenalty):
        raise ValueError('Sobolev preconditioning requires an amplitude Sobolev penalty')
    destination.mkdir(parents=True, exist_ok=True)
    if (destination/'result.json').exists():
        return read_model(destination/'fit.npz')
    start = time.monotonic()
    metrics = [np.asarray(chebyshev_gram_matrix(c.coeffs.shape[-1], tuple(box)))
               for c, box in zip(init.cores, init.domain, strict=True)]
    if preconditioning == 'sobolev':
        # Change optimizer coordinates, not the represented density or loss.
        # Physical L2 whitening alone leaves high-order derivative directions stiff.
        metrics = [gram+strength*w*np.asarray(derivative) for gram, w, derivative
                   in zip(metrics, penalty.weights, penalty.derivatives, strict=True)]
    factors = [np.linalg.cholesky(metric) for metric in metrics]
    inverse = tuple(jnp.asarray(np.linalg.inv(f)) for f in factors)
    params = tuple(c@f for c, f in zip(tt_params(init), factors, strict=True))

    def rebuild(p):
        return rebuild_tt(init, tuple(c@f for c, f in zip(p, inverse, strict=True)))

    direct = None
    if kernel is not None:
        obs = np.concatenate([train, valid])
        observation_domain = np.stack([obs.min(axis=0)-.02, obs.max(axis=0)+.02], axis=1)
        direct = TTConvolution.from_tt(init, kernel)
        op = direct.prepare(observation_domain, tolerance=1e-14)
        log_density = op.log_density
    else:
        box = np.asarray(init.domain)
        if not all(np.all((x >= box[:, 0]) & (x <= box[:, 1])) for x in (train, valid)):
            raise ValueError('An unconvolved fit cannot score observations outside its support')
        log_density = log_density_tt
    evaluate = jax.jit(lambda p, x: log_density(rebuild(p), x))
    roughness = jax.jit(lambda p: penalty(rebuild(p)) if penalty is not None else jnp.array(0.))
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(
        optax.cosine_decay_schedule(lr, max(steps, 1), alpha=.1)))

    @jax.jit
    def step(p, state, x):
        if microbatches == 1:
            loss, grad = jax.value_and_grad(lambda pp: -evaluate(pp, x).mean()+strength*roughness(pp))(p)
        else:
            # Same full-batch gradient, accumulated over equal chunks so the
            # convolution's backward pass holds one chunk at a time.
            def chunk(acc, xs):
                value, g = jax.value_and_grad(lambda pp: -evaluate(pp, xs).mean())(p)
                return (acc[0]+value, jax.tree.map(jnp.add, acc[1], g)), None
            zero = (jnp.zeros(()), jax.tree.map(jnp.zeros_like, p))
            (total, summed), _ = jax.lax.scan(chunk, zero, x.reshape(microbatches, -1, x.shape[-1]))
            reg, reg_grad = jax.value_and_grad(lambda pp: strength*roughness(pp))(p)
            loss = total/microbatches+reg
            grad = jax.tree.map(lambda a, b: a/microbatches+b, summed, reg_grad)
        updates, state = optimizer.update(grad, state, p)
        return optax.apply_updates(p, updates), state, loss

    state, key = optimizer.init(params), jax.random.PRNGKey(seed)
    training = jnp.asarray(train)
    # Validation also contracts two amplitude bonds through the noise TT.
    # Respect the memory budget used for training instead of silently reverting
    # to 256 rows for high-rank fits that require gradient accumulation.
    evaluation_batch = min(256, batch_size//microbatches)
    best, best_nll, best_step, best_logs, best_score = params, np.inf, 0, None, np.inf
    history = []
    for i in range(steps+1):
        if i % 400 == 0 or i == steps:
            logs = np.concatenate([np.asarray(evaluate(params, jnp.asarray(valid[j:j+evaluation_batch])))
                                   for j in range(0, len(valid), evaluation_batch)])
            if not np.isfinite(logs).all():
                raise FloatingPointError(f'{destination}: invalid likelihood at step {i}')
            nll = -float(logs.mean())
            energy = float(roughness(params))
            score = nll + (strength*energy if checkpoint_rule == 'regularized' else 0.)
            history.append({'step': i, 'nll': nll, 'penalty': energy})
            if score < best_score:
                best, best_nll, best_step, best_logs = params, nll, i, logs
                best_score = score
            print(destination.name, i, 'NLL', round(nll, 7), 'seconds', round(time.monotonic()-start), flush=True)
        if i == steps:
            break
        key, sub = jax.random.split(key)
        batch = training[jax.random.randint(sub, (batch_size,), 0, len(training))]
        params, state, loss = step(params, state, batch)
        if not np.isfinite(float(loss)):
            raise FloatingPointError(f'{destination}: invalid objective at step {i}')
    model = rebuild(best)
    discrepancy = 0.
    if direct is not None:
        refined = TTConvolution.from_tt(init, kernel, quadrature_order=2*np.asarray(direct.orders))
        discrepancy = float(np.max(np.abs(np.asarray(evaluate(best, jnp.asarray(valid[:8]))) -
                                         np.asarray(refined.log_density(model, jnp.asarray(valid[:8]))))))
        if discrepancy > 1e-7:
            raise FloatingPointError(f'Convolution quadrature discrepancy: {discrepancy}')
    save_model(destination/'fit.npz', model)
    np.save(destination/'validation.npy', best_logs)
    (destination/'result.json').write_text(json.dumps({'best_nll': best_nll, 'best_step': best_step,
        'strength': strength, 'steps': steps, 'lr': lr, 'batch_size': batch_size, 'history': history,
        'checkpoint_rule': checkpoint_rule, 'selection_score': best_score, 'seed': seed,
        'preconditioning': preconditioning, 'microbatches': microbatches,
        'device': jax.devices()[0].device_kind,
        'quadrature_log_error': discrepancy, 'seconds': time.monotonic()-start}, indent=2)+'\n')
    return model


def select_reference(train, valid, domain, kernel, output, clusters):
    if (output/'reference.npz').exists():
        return np.load(output/'reference.npz')['log_coeffs']
    records, coefficients, scores = [], [], []
    for degree in (2, 4, 6, 8):
        for shared in (True, False):
            coeff, logs, record = fit_noisy_marginal_reference(
                train, valid, domain, degree=degree, shared=shared, noise_tt=kernel)
            print('pilot', record, flush=True)
            records.append(record)
            coefficients.append(coeff)
            scores.append(logs)
    chosen, detail = select_by_paired_noisy_nll(np.stack(scores),
        [r['degree']*(1 if r['shared'] else train.shape[1]) for r in records], cluster_ids=clusters)
    np.savez(output/'reference.npz', domain=domain, log_coeffs=coefficients[chosen])
    (output/'reference_selection.json').write_text(json.dumps({'records': records, 'selected': chosen,
        'selection': detail}, indent=2)+'\n')
    return coefficients[chosen]


def observations(dim, noise, count, seed):
    clean = np.array(sample_ginzburg_landau(seed, count, dim))
    rng = np.random.default_rng(seed)
    if noise == 'paired':
        clean += rng.multivariate_normal(np.zeros(dim), _noniso_covariance(dim, 1.5), count)
    elif noise == 'mixture':
        comp = rng.choice(2, count, p=np.asarray(NOISE_WEIGHTS))
        clean += rng.normal(size=(count, dim))*np.asarray(NOISE_STDS)[comp, None]*1.5
    return clean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=['gl', 'cryoem'], required=True)
    parser.add_argument('--dim', type=int, required=True)
    parser.add_argument('--noise', choices=['free', 'paired', 'mixture'], default='paired')
    parser.add_argument('--path', action='store_true', help='4D projection of the cryo-EM 10D path embedding')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=8000)
    parser.add_argument('--strengths', type=float, nargs='+', default=[.1, .3, 1.])
    parser.add_argument('--n-train', type=int, default=100000, help='GL training observations')
    parser.add_argument('--train-seed', type=int, default=0, help='GL training seed (validation seed is 2)')
    args = parser.parse_args()
    output, d = args.output, args.dim
    output.mkdir(parents=True, exist_ok=True)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    # Default sample settings are omitted so earlier saved runs still match.
    for key, default in (('n_train', 100000), ('train_seed', 0)):
        if settings[key] == default:
            settings.pop(key)
    if args.train_seed == 2:
        raise ValueError('train seed 2 is the validation seed')
    if ((output/'settings.json').exists()
            and json.loads((output/'settings.json').read_text()) != settings):
        raise ValueError('Settings differ from saved run; choose a new output directory')
    (output/'settings.json').write_text(json.dumps(settings, indent=2)+'\n')
    if (output/'selection.json').exists() and (args.family != 'cryoem' or (output/'naive/result.json').exists()):
        print('Completed case already saved:', output, flush=True)
        return
    noisy = args.family == 'cryoem' or args.noise != 'free'
    print('Loading observations', args.family, d, args.noise, flush=True)
    if args.family == 'gl':
        train = observations(d, args.noise, args.n_train, args.train_seed)
        valid = observations(d, args.noise, 2048, 2)
        shift, scale = np.zeros(d), np.ones(d)
        clusters = np.arange(len(valid)) % 200
    else:
        filename = f'path_noise1_dim{d}.npz' if args.path else f'dim{d}.npz'
        cache = np.load(Path('cryoem_data/cache')/filename)
        points = cache['points']
        permutation = np.random.default_rng(0).permutation(len(points))
        train_idx, valid_idx = permutation[4096:], permutation[:4096]
        shift, scale = points[train_idx].mean(axis=0), points[train_idx].std(axis=0)
        train, valid = (points[train_idx]-shift)/scale, (points[valid_idx]-shift)/scale
        variances = np.diag(cache['cov_mean'])/scale**2
        clusters = None
        np.savez(output/'split.npz', train_idx=train_idx, valid_idx=valid_idx)
    np.savez(output/'data.npz', train=train, valid=valid, shift=shift, scale=scale)
    domain = np.asarray(robust_domain(jnp.asarray(train)) if noisy else data_domain(jnp.asarray(train)))
    if not noisy and not np.all((valid >= domain[:, 0]) & (valid <= domain[:, 1])):
        raise ValueError('Noise-free validation lies outside training-derived support')
    print('Selecting KDE bandwidth', len(train), 'training observations', flush=True)
    bandwidth = select_gaussian_kde_bandwidth(jnp.asarray(train))
    np.save(output/'bandwidth.npy', bandwidth)
    rank = 8 if d <= 8 else 10 if noisy or d <= 14 else 14
    if (output/'init.npz').exists():
        init = read_model(output/'init.npz')
    else:
        print('Initializing Chebyshev TT', flush=True)
        init = init_tt_from_kde(jnp.asarray(train), max_rank=rank, domain=jnp.asarray(domain),
            n_coeffs=(25 if noisy else 21,), bandwidth=bandwidth, n_starts=64, gd_steps=10)
        save_model(output/'init.npz', init)
    print('Initialized', args.family, d, args.noise, init.ranks, flush=True)
    if not noisy:
        fit = fit_candidate(init, train, valid, output/'noise_free', steps=args.steps, lr=.001, batch_size=2048)
        save_model(output/'selected.npz', fit)
        (output/'selection.json').write_text(json.dumps({'selected': 'noise_free', 'rule': 'minimum validation NLL'})+'\n')
        return
    if (output/'kernel.npz').exists():
        kernel = read_model(output/'kernel.npz')
    else:
        if args.family == 'cryoem':
            # The prescribed diagonal Gaussian approximation used by the existing
            # cryo-EM experiments; the training and pilot APIs receive only its TT.
            kernel = build_diagonal_gaussian_noise_kernel_tt(jnp.asarray(variances), jnp.asarray(domain),
                n_coeffs=(65, 129, 257, 513))
        elif args.noise == 'paired':
            std = float(np.sqrt(_noniso_covariance(d, 1.5)[0, 0]))
            kernel = build_pairwise_correlated_noise_kernel_tt(std, .5, jnp.asarray(domain), d,
                max_rank=12, n_starts=64, gd_steps=10)
        else:
            kernel = build_gaussian_mixture_noise_kernel_tt(NOISE_STDS*1.5, NOISE_WEIGHTS,
                jnp.asarray(domain), d, n_coeffs=257)
        save_model(output/'kernel.npz', kernel)
    noise_scales = np.array([m.moments()[1] for m in noise_marginals_from_tt(kernel)])
    smoothness = AmplitudeSobolevPenalty.from_tt(init, order=2, length_scales=noise_scales)
    warm = fit_candidate(init, train, valid, output/'warmup', kernel=kernel, penalty=smoothness,
                         strength=3e-5, steps=args.steps)
    reference = select_reference(train, valid, domain, kernel, output, clusters)
    for count in (33, 49, 65):
        warm = rebuild_tt(warm, tuple(jnp.pad(c.coeffs, ((0, 0), (0, 0), (0, count-c.coeffs.shape[-1]))) for c in warm.cores))
        try:
            fisher = FisherVisibilityPenalty.from_tt(warm, reference, noise_tt=kernel, visibility_floor=.01)
            break
        except ValueError as exc:
            if 'reference amplitude needs more Chebyshev coefficients' not in str(exc) or count == 65:
                raise
            print('Increasing basis to resolve the noisy-data pilot', flush=True)
    np.savez(output/'fisher_spectra.npz', **{f'axis_{k}': v for k, v in enumerate(fisher.eigenvalues)})
    fits, logs, roughness = [warm], [np.load(output/'warmup/validation.npy')], [float(fisher(warm))]
    names = ['warmup']
    for strength in args.strengths:
        name = f'fisher_{strength:g}'
        fit = fit_candidate(warm, train, valid, output/name, kernel=kernel, penalty=fisher,
                            strength=strength, steps=args.steps)
        fits.append(fit)
        logs.append(np.load(output/name/'validation.npy'))
        roughness.append(float(fisher(fit)))
        names.append(name)
    chosen, detail = select_by_paired_noisy_nll(np.stack(logs), roughness, cluster_ids=clusters)
    save_model(output/'selected.npz', fits[chosen])
    (output/'selection.json').write_text(json.dumps({'selected': names[chosen], 'names': names,
        'roughness': roughness, 'selection': detail, 'rule': 'paired noisy one-SE, then minimum common Fisher penalty'}, indent=2)+'\n')
    print('SELECTED', names[chosen], flush=True)
    if args.family == 'cryoem':
        # A separate observed-density baseline, including every training point.
        baseline_domain = data_domain(jnp.asarray(train))
        baseline = init_tt_from_kde(jnp.asarray(train), max_rank=rank, domain=baseline_domain,
            n_coeffs=(25,), bandwidth=bandwidth, n_starts=64, gd_steps=10)
        mass = float(squared_integral_tt(baseline))
        initialization = {'method': 'KDE', 'initial_squared_integral': str(mass),
                          'initial_ranks': list(baseline.ranks)}
        if not np.isfinite(mass) or mass <= 0:
            # Uniform pivot searches can miss a concentrated KDE on a very
            # large high-dimensional box. Broaden only the initialization;
            # the NLL still fits every original observation without a penalty.
            save_model(output/'naive_failed_init.npz', baseline)
            broad_bandwidth = np.maximum(bandwidth, .25*np.ptp(np.asarray(baseline_domain), axis=1))
            print('Naive KDE initialization has invalid squared integral', mass,
                  '; retrying with a broader KDE initialization', flush=True)
            baseline = init_tt_from_kde(jnp.asarray(train), max_rank=rank, domain=baseline_domain,
                n_coeffs=(25,), bandwidth=jnp.asarray(broad_bandwidth), n_starts=64, gd_steps=10)
            initialization.update(method='broadened KDE', bandwidth=broad_bandwidth.tolist())
        initialization['final_squared_integral'] = str(float(squared_integral_tt(baseline)))
        (output/'naive_initialization.json').write_text(json.dumps(initialization, indent=2)+'\n')
        save_model(output/'naive_init.npz', baseline)
        fit_candidate(baseline, train, valid, output/'naive', steps=args.steps, lr=.001, batch_size=1024)


if __name__ == '__main__':
    main()
