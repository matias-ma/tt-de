"""Measure final clipping and select candidates on noisy validation only."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from fit_direct_optimization import prepare_problem, record
from refresh_density_fits import read_model

from ftde.clipped_tt_density import ClippedTTQuadrature4D
from ftde.linear_tt_convolution import LinearTTConvolution
from ftde.noisy_model_selection import select_by_paired_noisy_nll


def grid_weights(model, kind, covariances, weights, order):
    op = ClippedTTQuadrature4D.from_tt(model, covariances, weights, order=order)
    raw = jax.jit(op.raw_grid)(model)
    qw = op.pair_weights[0][:, None]*op.pair_weights[1][None]
    if kind == 'squared':
        values, negative = raw**2, jnp.array(0.)
    else:
        values, negative = jnp.maximum(raw, 0), -jnp.sum(jnp.minimum(raw, 0)*qw)
    mass = jnp.sum(values*qw)
    return op, values*qw/mass, float(mass), float(negative)


def log_predictions(op, weighted, observations, batch=128):
    @jax.jit
    def evaluate(grid, y):
        return op.convolve_weighted(grid, op.noise_factors(y))
    values = []
    for start in range(0, len(observations), batch):
        y = observations[start:start+batch]
        count = len(y)
        y = np.pad(y, ((0, batch-count), (0, 0)))
        values.append(np.asarray(evaluate(weighted, jnp.asarray(y)))[:count])
    values = np.concatenate(values)
    if not np.all(np.isfinite(values) & (values > 0)):
        raise FloatingPointError('Invalid final normalized density prediction')
    return np.log(values)


def score(base, root, order=65):
    data, domain, covs, weights = prepare_problem(base, root)
    candidates = sorted(p for p in root.iterdir() if p.is_dir() and (p/'result.json').exists())
    specs = [(p.name, p/'fit.npz', 'signed' if p.name.startswith('signed') else 'squared') for p in candidates]
    specs += [('previous_squared', base/'selected.npz', 'squared')]
    result, logs, complexity = {}, [], []
    for name, path, kind in specs:
        dest = root/name
        dest.mkdir(exist_ok=True)
        score_file = dest/f'validation_q{order}.json'
        if score_file.exists():
            r = json.loads(score_file.read_text())
            assert r['fit_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
            result[name] = r
            logs.append(np.load(dest/f'clipped_validation_q{order}.npy'))
            complexity.append(r['coefficient_count'])
            continue
        start = time.monotonic()
        model = read_model(path)
        op, weighted, mass, negative = grid_weights(model, kind, covs, weights, order)
        predictions = log_predictions(op, weighted, data['valid'])
        r = {'name': name, 'kind': kind, 'order': order, 'positive_mass': mass,
             'negative_mass': negative, 'validation_nll': -float(predictions.mean()),
             'fit_path': str(path), 'fit_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
             'coefficient_count': sum(c.coeffs.size for c in model.cores),
             'seconds': time.monotonic()-start}
        if kind == 'signed':
            kernel = read_model(base/'kernel.npz')
            obs_domain = np.stack([np.asarray(kernel.domain)[:, 0]+domain[:, 1],
                                   np.asarray(kernel.domain)[:, 1]+domain[:, 0]], axis=1)
            forward = LinearTTConvolution.from_tt(model, kernel, obs_domain)
            evaluate = jax.jit(forward.density)
            raw = np.concatenate([np.asarray(evaluate(model, jnp.asarray(data['valid'][j:j+128])))
                                  for j in range(0, len(data['valid']), 128)])
            r['raw_convolved_nonpositive'] = int(np.sum(raw <= 0))
            r['raw_nll_if_positive'] = -float(np.log(raw).mean()) if np.all(raw > 0) else None
        np.save(dest/f'clipped_validation_q{order}.npy', predictions)
        np.savez(dest/f'marginals_q{order}.npz', nodes=np.asarray(op.nodes),
                 pair01=np.asarray((weighted.sum(1)/op.pair_weights[0]).reshape(order, order).T),
                 pair23=np.asarray((weighted.sum(0)/op.pair_weights[1]).reshape(order, order).T))
        record(score_file, r)
        result[name] = r
        logs.append(predictions)
        complexity.append(r['coefficient_count'])
        print('VALIDATION', r, flush=True)
        jax.clear_caches()
    names = [s[0] for s in specs]
    chosen, detail = select_by_paired_noisy_nll(np.stack(logs), complexity)
    # Include the historical fit as a reference, but identify the best NEW fit
    # separately: the old one had a fitted initializer and a different budget.
    new_indices = [i for i, name in enumerate(names) if name != 'previous_squared']
    new_chosen, new_detail = select_by_paired_noisy_nll(np.stack(logs)[new_indices], np.array(complexity)[new_indices])
    penalized = [i for i, name in enumerate(names) if name.startswith('signed') and '_neg' in name]
    extra = {}
    if penalized:
        choice, detail_penalty = select_by_paired_noisy_nll(np.stack(logs)[penalized],
            [result[names[i]]['negative_mass'] for i in penalized])
        extra = {'penalized_selected': names[penalized[choice]], 'penalized_selection': detail_penalty,
                 'penalized_rule': 'paired one-SE likelihood, then smallest negative mass removed by final clipping'}
    record(root/f'selection_q{order}.json', {'selected': names[chosen], 'new_selected': names[new_indices[new_chosen]],
        'names': names, 'selection': detail, 'new_selection': new_detail,
        'rule': 'paired one-SE noisy validation likelihood, then coefficient count; truth unused', 'results': result, **extra})
    print('SELECTED', names[chosen], 'NEW', names[new_indices[new_chosen]], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('artifacts/swirl/swirl_mixture_dim4'))
    parser.add_argument('--root', type=Path, default=Path('artifacts/direct_optimization/swirl4'))
    parser.add_argument('--order', type=int, default=65)
    args = parser.parse_args()
    score(args.base, args.root, args.order)
