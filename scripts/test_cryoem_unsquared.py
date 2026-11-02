"""Fit a density TT directly and audit final clipping on a full 4D grid.

Experimental candidates stay outside the public gallery. Clipping is applied
to the joint density before marginalization, never just to displayed marginals.
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import optax
from chebttge.chebtt import ChebTT
from chebttge.computations import integrate
from chebttge.cores import ChebyshevCoeffCore
from refresh_density_fits import read_model, save_model
from scipy.special import roots_legendre

from ftde.cheb_gram import chebyshev_gram_matrix
from ftde.cheb_product import square_tt
from ftde.linear_tt_convolution import LinearTTConvolution
from ftde.marginals import squared_pair_grid
from ftde.noise_marginals import noise_marginals_from_tt
from ftde.sobolev import AmplitudeSobolevPenalty
from ftde.tt_density import rebuild_tt


def compress_physical(model, rank):
    """Right-canonicalize in physical L2, then truncate true Schmidt spectra."""
    arrays, inverses = [], []
    for core in model.cores:
        chol = np.linalg.cholesky(np.asarray(chebyshev_gram_matrix(core.coeffs.shape[-1], core.domain)))
        arrays.append(np.asarray(core.coeffs)@chol)
        inverses.append(np.linalg.inv(chol))
    for k in range(len(arrays)-1, 0, -1):
        a, b, n = arrays[k].shape
        q, r = np.linalg.qr(arrays[k].reshape(a, b*n).T)
        arrays[k] = q.T.reshape(-1, b, n)
        arrays[k-1] = np.einsum('abn,cb->acn', arrays[k-1], r)
    original_squared_norm = float(np.sum(arrays[0]**2))
    discarded = 0.
    for k in range(len(arrays)-1):
        a, b, n = arrays[k].shape
        u, s, vh = np.linalg.svd(arrays[k].transpose(0, 2, 1).reshape(a*n, b), full_matrices=False)
        keep = min(rank, len(s))
        discarded += float(np.sum(s[keep:]**2))
        arrays[k] = u[:, :keep].reshape(a, n, keep).transpose(0, 2, 1)
        arrays[k+1] = np.einsum('ab,bcn->acn', s[:keep, None]*vh[:keep], arrays[k+1])
    cores = tuple(ChebyshevCoeffCore(jnp.asarray(c@inv), old.domain)
                  for c, inv, old in zip(arrays, inverses, model.cores, strict=True))
    compressed = ChebTT(cores=cores, domain=model.domain, ranks=tuple(c.shape[1] for c in arrays[:-1]))
    return compressed, {'relative_l2_bound_before_normalization': float(np.sqrt(discarded/original_squared_norm)),
                        'integral_before_normalization': float(integrate(compressed))}


def normalize(model):
    return rebuild_tt(model, (model.cores[0].coeffs/integrate(model),
                              *[c.coeffs for c in model.cores[1:]]))


def clipping_audit(model, kernel, data, output, order, count=256):
    """Tensor quadrature in 4D; independently rescore the clipped joint model."""
    t, w = roots_legendre(order)
    box = np.asarray(model.domain)
    nodes = box[:, :1]+np.ptp(box, axis=1)[:, None]*(t+1)/2
    weights = np.ptp(box, axis=1)[:, None]*w/2
    start = time.monotonic()
    state = jnp.ones(1)
    for k, core in enumerate(model.cores):
        state = jnp.einsum('...a,tab->...tb', state, core(jnp.asarray(nodes[k])))
    values = np.asarray(state[..., 0])
    ww = weights[0, :, None, None, None]*weights[1, None, :, None, None]*weights[2, None, None, :, None]*weights[3, None, None, None, :]
    signed_mass = float(np.sum(values*ww))
    negative_mass = float(-np.sum(np.minimum(values, 0)*ww))
    positive = np.maximum(values, 0)
    positive_mass = float(np.sum(positive*ww))
    @jax.jit
    def convolved(weighted, y):
        us = [u((y[:, k, None]-jnp.asarray(nodes[k])[None]).ravel()).reshape(len(y), order, *u.shape)
              for k, u in enumerate(kernel.cores)]
        state = jnp.einsum('abcd,tar->trbcd', weighted, us[0][:, :, 0, :])
        state = jnp.einsum('trbcd,tbrs->tscd', state, us[1])
        state = jnp.einsum('tscd,tcsu->tud', state, us[2])
        return jnp.einsum('tud,tdu->t', state, us[3][:, :, :, 0])

    # A fixed subset provides a paired diagnostic; it is not used to tune fits.
    validation = data['valid'][:count]
    def score(weighted):
        weighted = jnp.asarray(weighted)
        return np.concatenate([np.asarray(convolved(weighted, jnp.asarray(validation[k:k+8])))
                               for k in range(0, len(validation), 8)])
    q = score(positive*ww/positive_mass)
    raw_grid_q = score(values*ww)
    exact_logs = np.load(output/'validation.npy')[:len(validation)]
    raw_path = output/'raw_validation.npy'
    exact_values = np.load(raw_path)[:len(validation)] if raw_path.exists() else np.exp(exact_logs)
    positive_raw = np.all((exact_values > 0) & (raw_grid_q > 0))
    clipped_valid = np.all(np.isfinite(q) & (q > 0))
    with np.errstate(invalid='ignore', divide='ignore'):
        clipped_logs = np.log(q)
    np.save(output/f'clipped_validation_order{order}_n{len(validation)}.npy', clipped_logs)
    panels = {}
    for i, j in [(0, 1), (2, 3)]:
        others = tuple(k for k in range(4) if k not in (i, j))
        signed = np.sum(values*ww, axis=others)/(weights[i, :, None]*weights[j, None, :])
        clipped = np.sum(positive*ww, axis=others)/(weights[i, :, None]*weights[j, None, :])/positive_mass
        panels[str(i)+str(j)] = {'xs': nodes[i], 'ys': nodes[j], 'signed': signed.T, 'clipped': clipped.T}
        np.savez(output/f'marginal_{i}{j}_order{order}.npz', **panels[str(i)+str(j)])
    record = {'order': order, 'validation_count': len(validation), 'signed_integral': signed_mass, 'negative_mass': negative_mass,
              'positive_mass': positive_mass, 'minimum_density_on_grid': float(values.min()),
              'negative_grid_fraction': float(np.mean(values < 0)),
              'clipped_validation_nll': float(-clipped_logs.mean()) if clipped_valid else None,
              'clipped_validation_nonpositive': int(np.sum(q <= 0)),
              'raw_validation_nll': float(-exact_logs.mean()) if np.all(exact_values > 0) else None,
              'raw_validation_nonpositive': int(np.sum(exact_values <= 0)),
              'raw_quadrature_max_absolute_error': float(np.max(np.abs(raw_grid_q-exact_values))),
              'raw_quadrature_max_log_error': float(np.max(np.abs(np.log(raw_grid_q)-exact_logs))) if positive_raw else None,
              'seconds': time.monotonic()-start}
    (output/f'clipping_order{order}_n{len(validation)}.json').write_text(json.dumps(record, indent=2)+'\n')
    print('CLIPPING', record, flush=True)
    return panels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('artifacts/cryoem_refinement/gallery_dim4'))
    parser.add_argument('--initial-amplitude', type=Path, default=Path('artifacts/figure_refresh/cryoem_dim4/init.npz'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rank', type=int, default=32)
    parser.add_argument('--steps', type=int, default=2400)
    parser.add_argument('--lr', type=float, default=.0002)
    parser.add_argument('--strength', type=float, default=1e-4)
    parser.add_argument('--negative-penalty', type=float, default=0.)
    parser.add_argument('--checkpoint-rule', choices=['raw-valid', 'final'], default='raw-valid')
    parser.add_argument('--orders', nargs='+', type=int, default=[49, 65])
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    if (out/'experiment.json').exists():
        previous = json.loads((out/'experiment.json').read_text())
        previous.setdefault('checkpoint_rule', 'raw-valid')
        previous.setdefault('lr', .0002)
        if previous != config:
            raise ValueError('Different settings; choose a new directory')
    (out/'experiment.json').write_text(json.dumps(config, indent=2)+'\n')
    data = np.load(args.base/'data.npz')
    if data['train'].shape[1] != 4:
        raise ValueError('Full clipping audit currently requires dimension four')
    kernel = read_model(args.base/'kernel.npz')
    amplitude = read_model(args.initial_amplitude)
    compressed, compression = compress_physical(square_tt(amplitude), args.rank)
    initial = normalize(compressed)
    (out/'initialization.json').write_text(json.dumps(compression, indent=2)+'\n')
    save_model(out/'init.npz', initial)
    obs = np.concatenate([data['train'], data['valid']])
    obs_box = np.stack([obs.min(0)-.02, obs.max(0)+.02], axis=1)
    operator = LinearTTConvolution.from_tt(initial, kernel, obs_box)
    scales = [m.moments()[1] for m in noise_marginals_from_tt(kernel)]
    roughness = AmplitudeSobolevPenalty.from_tt(initial, order=2, length_scales=scales)
    factors = [np.linalg.cholesky(np.asarray(gram)+args.strength*weight*np.asarray(derivative))
               for gram, weight, derivative in zip(roughness.grams, roughness.weights, roughness.derivatives, strict=True)]
    inverse = [jnp.asarray(np.linalg.inv(c)) for c in factors]
    params = tuple(c.coeffs@f for c, f in zip(initial.cores, factors, strict=True))
    def rebuild(p):
        return normalize(rebuild_tt(initial, tuple(c@f for c, f in zip(p, inverse, strict=True))))
    train = jnp.asarray(data['train'])
    bounds = jnp.asarray(initial.domain)
    volume = jnp.prod(bounds[:, 1]-bounds[:, 0])
    evaluate = jax.jit(lambda p, y: operator.density(rebuild(p), y))
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(optax.cosine_decay_schedule(args.lr, args.steps, alpha=.1)))
    def loss(p, y, probes):
        f = rebuild(p)
        q = operator.density(f, y)
        floor = 1e-12
        # Linear continuation is only a training safeguard for nonpositive
        # observed predictions. Reported validation likelihood is never floored.
        nll = jnp.mean(-jnp.log(jnp.maximum(q, floor))+jnp.maximum(1-q/floor, 0))
        penalty = roughness(f)
        negativity = volume*jnp.mean(jnp.maximum(-f(probes), 0)) if args.negative_penalty else 0.
        return nll+args.strength*penalty+args.negative_penalty*negativity
    @jax.jit
    def step(p, state, y, probes):
        objective, gradient = jax.value_and_grad(loss)(p, y, probes)
        update, state = optimizer.update(gradient, state, p)
        return optax.apply_updates(p, update), state, objective
    if (out/'fit.npz').exists():
        fitted = read_model(out/'fit.npz')
    else:
        state, key = optimizer.init(params), jax.random.PRNGKey(0)
        best, best_score, best_logs, best_step = params, np.inf, None, 0
        history = []
        start = time.monotonic()
        for k in range(args.steps+1):
            if k % 200 == 0 or k == args.steps:
                q = np.concatenate([np.asarray(evaluate(params, jnp.asarray(data['valid'][j:j+512]))) for j in range(0, len(data['valid']), 512)])
                valid = np.all(np.isfinite(q) & (q > 0))
                nll = float(-np.log(q).mean()) if valid else None
                energy = float(roughness(rebuild(params)))
                history.append({'step': k, 'nll': nll, 'nonpositive_observed': int(np.sum(q <= 0)), 'roughness': energy})
                if valid and nll+args.strength*energy < best_score:
                    best, best_score, best_logs, best_step = params, nll+args.strength*energy, np.log(q), k
                print('DIRECT', k, nll, 'roughness', energy, 'seconds', time.monotonic()-start, flush=True)
            if k == args.steps:
                break
            key, first, second = jax.random.split(key, 3)
            batch = train[jax.random.randint(first, (512,), 0, len(train))]
            probes = jax.random.uniform(second, (2048, 4), minval=bounds[:, 0], maxval=bounds[:, 1])
            params, state, objective = step(params, state, batch, probes)
            if not np.isfinite(float(objective)):
                print('Nonfinite objective; retain best valid observed-density checkpoint', flush=True)
                break
        if best_logs is None:
            raise FloatingPointError('No checkpoint has positive finite noisy validation density')
        if args.checkpoint_rule == 'final':
            best, best_step = params, k
            q = np.concatenate([np.asarray(evaluate(params, jnp.asarray(data['valid'][j:j+512])))
                                for j in range(0, len(data['valid']), 512)])
            if not np.isfinite(q).all():
                raise FloatingPointError('Final signed iterate has nonfinite observed predictions')
            with np.errstate(invalid='ignore', divide='ignore'):
                best_logs = np.log(q)
        else:
            q = np.exp(best_logs)
        fitted = rebuild(best)
        save_model(out/'fit.npz', fitted)
        np.save(out/'validation.npy', best_logs)
        np.save(out/'raw_validation.npy', q)
        (out/'result.json').write_text(json.dumps({'best_nll': float(-best_logs.mean()) if np.all(q > 0) else None, 'best_step': best_step,
            'history': history, 'seconds': time.monotonic()-start, 'parameterization': 'unsquared signed density TT',
            'checkpoint_rule': args.checkpoint_rule, 'raw_validation_nonpositive': int(np.sum(q <= 0)),
            'preconditioning': 'sobolev', 'lr': args.lr, 'strength': args.strength, 'negative_penalty': args.negative_penalty}, indent=2)+'\n')
    original = read_model(args.base/'selected.npz')
    panels = None
    for order in args.orders:
        panels = clipping_audit(fitted, kernel, data, out, order)
    for label, panel in panels.items():
        i, j = map(int, label)
        baseline = squared_pair_grid(original, i, j, panel['xs'], panel['ys'])
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout='constrained')
        for ax, grid, title in zip(axes, [baseline, panel['signed'], panel['clipped']],
                ['Previous squared TT', 'Direct TT (signed)', 'Direct TT clipped jointly and normalized'], strict=True):
            peak = max(baseline.max(), panel['clipped'].max())
            signed = title == 'Direct TT (signed)'
            if signed:
                peak = max(peak, np.max(np.abs(grid)))
            im = ax.pcolormesh(panel['xs'], panel['ys'], grid, shading='nearest',
                              cmap='coolwarm' if signed else 'magma', vmin=-peak if signed else 0, vmax=peak)
            ax.set(title=title, xlabel=f'x{i} (standardized)', ylabel=f'x{j} (standardized)')
            fig.colorbar(im, ax=ax)
        fig.suptitle('Unsquared density experiment; signed panel requires the negative-mass audit')
        fig.savefig(out/f'comparison_{label}.png', dpi=150)
        plt.close(fig)


if __name__ == '__main__':
    main()
