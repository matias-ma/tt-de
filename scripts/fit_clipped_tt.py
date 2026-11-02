"""Optimize a direct Chebyshev density TT with clipping inside the likelihood.

Uses observed samples and prescribed noise only. No fitted signal mixture or
target function enters initialization, training, or checkpoint selection.
"""
import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from prescribed_noise import prescribed_noise
from refresh_density_fits import read_model, save_model
from test_cryoem_unsquared import compress_physical, normalize

from ftde.cheb_product import square_tt
from ftde.clipped_tt_density import ClippedTTQuadrature4D
from ftde.sobolev import AmplitudeSobolevPenalty
from ftde.tt_density import rebuild_tt, squared_integral_tt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--initial', type=Path)
    parser.add_argument('--initial-is-density', action='store_true')
    parser.add_argument('--rank', type=int, default=32)
    parser.add_argument('--order', type=int, default=65)
    parser.add_argument('--steps', type=int, default=2400)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=.0005)
    parser.add_argument('--strength', type=float, default=1e-5)
    parser.add_argument('--penalty-normalization', choices=['raw', 'positive'], default='positive')
    parser.add_argument('--negative-penalty', type=float, default=0.)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    if (out/'experiment.json').exists() and json.loads((out/'experiment.json').read_text()) != config:
        raise ValueError('Settings differ; choose a fresh output directory')
    (out/'experiment.json').write_text(json.dumps(config, indent=2)+'\n')
    if (out/'result.json').exists():
        print('Already completed', out, flush=True)
        return
    data = np.load(args.base/'data.npz')
    nc, nw = prescribed_noise(args.base)
    start = time.monotonic()
    if (out/'init.npz').exists():
        initial = read_model(out/'init.npz')
    else:
        source = read_model(args.initial or args.base/'init.npz')
        # Only a change of representation at initialization. All subsequent
        # trainable coefficients parameterize f itself, never its square root.
        initial, report = compress_physical(source if args.initial_is_density else square_tt(source), args.rank)
        initial = normalize(initial)
        save_model(out/'init.npz', initial)
        (out/'initialization.json').write_text(json.dumps(report, indent=2)+'\n')
    op = ClippedTTQuadrature4D.from_tt(initial, nc, nw, order=args.order)
    scales = np.sqrt(np.einsum('c,cii->i', nw, nc))
    smooth = AmplitudeSobolevPenalty.from_tt(initial, order=2, length_scales=scales)
    factors = [np.linalg.cholesky(np.asarray(g)+max(args.strength, 1e-5)*w*np.asarray(d))
               for g, w, d in zip(smooth.grams, smooth.weights, smooth.derivatives, strict=True)]
    inverses = tuple(jnp.asarray(np.linalg.inv(f)) for f in factors)
    params = tuple(c.coeffs@f for c, f in zip(initial.cores, factors, strict=True))
    def rebuild(p):
        return rebuild_tt(initial, tuple(c@f for c, f in zip(p, inverses, strict=True)))
    def penalty(f, weighted, mass):
        energy = smooth(f)
        if args.penalty_normalization == 'positive':
            positive_norm2 = jnp.sum(weighted**2/(op.pair_weights[0][:, None]*op.pair_weights[1][None]))*mass**2
            energy *= squared_integral_tt(f)/positive_norm2
        return energy
    def objective(p, y):
        f = rebuild(p)
        weighted, mass, negative = op.weighted_grid(f)
        q = op.convolve_weighted(weighted, op.noise_factors(y))
        nll = -jnp.mean(jnp.log(jnp.maximum(q, 1e-300)))
        return nll+args.strength*penalty(f, weighted, mass)+args.negative_penalty*negative/mass
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(
        optax.cosine_decay_schedule(args.lr, max(args.steps, 1), alpha=.1)))
    @jax.jit
    def step(p, state, y):
        loss, grad = jax.value_and_grad(objective)(p, y)
        update, state = optimizer.update(grad, state, p)
        return optax.apply_updates(p, update), state, loss
    grid = jax.jit(lambda p: op.weighted_grid(rebuild(p)))
    convolve = jax.jit(lambda w, y: op.convolve_weighted(w, op.noise_factors(y)))
    def roughness_fn(p):
        f = rebuild(p)
        weighted, mass, _ = op.weighted_grid(f)
        return penalty(f, weighted, mass)
    roughness = jax.jit(roughness_fn)
    train, valid = jnp.asarray(data['train']), data['valid']
    state, key = optimizer.init(params), jax.random.PRNGKey(args.seed)
    best_nll, best_step, history = np.inf, 0, []
    for k in range(args.steps+1):
        if k % 100 == 0 or k == args.steps:
            weighted, mass, negative = grid(params)
            q = np.concatenate([np.asarray(convolve(weighted, jnp.asarray(valid[j:j+256])))
                                for j in range(0, len(valid), 256)])
            valid_q = np.all(np.isfinite(q) & (q > 0))
            nll = float(-np.log(q).mean()) if valid_q else np.inf
            entry = {'step': k, 'nll': nll, 'negative_to_positive_mass': float(negative/mass),
                     'roughness': float(roughness(params)), 'seconds': time.monotonic()-start}
            history.append(entry)
            if nll < best_nll:
                best_nll, best_step = nll, k
                save_model(out/'fit.npz', rebuild(params))
                np.save(out/'validation.npy', np.log(q))
            print('CLIPPED', entry, flush=True)
        if k == args.steps:
            break
        key, draw = jax.random.split(key)
        batch = train[jax.random.randint(draw, (args.batch_size,), 0, len(train))]
        params, state, loss = step(params, state, batch)
        if not np.isfinite(float(loss)):
            raise FloatingPointError('Nonfinite clipped training objective')
    save_model(out/'final.npz', rebuild(params))
    result = {'best_nll': best_nll, 'best_step': best_step, 'history': history,
              'seconds': time.monotonic()-start, 'parameterization': 'max(f_TT, 0) / integral max(f_TT, 0)',
              'selection': 'minimum clipped noisy validation NLL', 'quadrature_order': args.order,
              'signal_mixture_used': False}
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
