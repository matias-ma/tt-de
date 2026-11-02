"""Observation-only noise and quadrature checks for the direct optimizer."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from fit_direct_optimization import record
from refresh_density_fits import read_model
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from score_direct_optimization import grid_weights, log_predictions

from ftde.signed_tt_least_squares import SignedTTLeastSquares4D

root = Path('artifacts/direct_optimization')
checks = {}
for dim in (4, 10):
    noise = np.load(f'artifacts/swirl_improvement/swirl_mixture_dim{dim}/noise.npz')
    covs, weights = noise['covariances'], noise['weights']
    rng = np.random.default_rng(68123)
    labels = rng.choice(len(weights), 1024, p=weights)
    points = np.einsum('nij,nj->ni', np.linalg.cholesky(covs)[labels], rng.normal(size=(1024, dim)))
    analytic = np.exp(logsumexp(np.stack([np.log(w)+multivariate_normal.logpdf(points, cov=c)
                     for w, c in zip(weights, covs, strict=True)]), axis=0))
    kernel = read_model(f'artifacts/swirl/swirl_mixture_dim{dim}/kernel.npz')
    represented = np.asarray(jax.jit(lambda x, kernel=kernel: kernel(x))(jnp.asarray(points)))
    difference = float(np.max(np.abs(represented/analytic-1)))
    if not np.all(represented > 0) or difference > 1e-6:
        raise FloatingPointError('Prescribed noise specification and kernel disagree')
    checks[f'noise{dim}'] = {'max_relative_error': difference, 'samples': len(points)}
    print('NOISE', dim, difference, flush=True)

folder = root/'swirl4'
coarse = np.load(folder/'geometry_n49_q81.npz')
domain = np.asarray(json.loads((folder/'problem.json').read_text())['domain'])
noise = np.load('artifacts/swirl_improvement/swirl_mixture_dim4/noise.npz')
fine = SignedTTLeastSquares4D.from_noise(domain, 49, noise['covariances'], noise['weights'], order=97)
for side, arrays in [('left_grams', fine.left_grams), ('right_grams', fine.right_grams)]:
    difference = float(np.linalg.norm(np.asarray(arrays)-coarse[side])/np.linalg.norm(coarse[side]))
    if difference > 1e-7:
        raise FloatingPointError(f'Integrated convolution Gram quadrature disagreement: {side}, {difference}')
    checks[side] = {'relative_frobenius_error': difference, 'orders': [81, 97]}
    print('GRAM', side, difference, flush=True)

# Audit the displayed signed representatives BEFORE any signal truth enters.
selection = json.loads((folder/'selection_q97.json').read_text())
families = [[n for n in selection['names'] if n.startswith('signed') and ('_neg' in n) == penalized]
            for penalized in (False, True)]
names = {min(group, key=lambda n: selection['results'][n]['validation_nll']) for group in families}
names.add(selection['penalized_selected'])
if selection['new_selected'].startswith('signed'):
    names.add(selection['new_selected'])
valid = np.load('artifacts/swirl/swirl_mixture_dim4/data.npz')['valid'][:256]
for name in sorted(names):
    model = read_model(selection['results'][name]['fit_path'])
    op, weighted, mass, _ = grid_weights(model, 'signed', noise['covariances'], noise['weights'], 129)
    fine_logs = log_predictions(op, weighted, valid)
    old_logs = np.load(folder/name/'clipped_validation_q97.npy')[:len(valid)]
    delta = fine_logs-old_logs
    check = {'orders': [97, 129], 'maximum_log_error': float(np.max(np.abs(delta))),
             'mean_log_error': float(delta.mean()),
             'positive_mass_change': mass-selection['results'][name]['positive_mass']}
    checks[f'clipping_{name}'] = check
    print('CLIPPING', name, check, flush=True)
    if abs(check['mean_log_error']) > 1e-3 or abs(check['positive_mass_change']) > 1e-3:
        raise FloatingPointError('Refine clipping quadrature before freezing model selection')
    del op, weighted
    jax.clear_caches()
record(root/'numerical_checks.json', checks)
