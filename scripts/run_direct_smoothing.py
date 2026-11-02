"""Select spatial TT smoothing from observations, then evaluate frozen fits."""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from evaluate_direct_noise_free import nll_summary, values_at
from fit_direct_optimization import record
from refresh_density_fits import read_model, save_model
from score_direct_optimization import grid_weights

from ftde.tt_smoothing import heat_smooth_tt

ROOT = Path('artifacts/direct_optimization/smoothing_free4')
SOURCE_ROOT = Path('artifacts/direct_optimization/swirl_free4')
FRACTIONS = (0., .0025, .005, .01, .02, .04, .08, .16)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score():
    ROOT.mkdir(exist_ok=True)
    selection = json.loads((SOURCE_ROOT/'selection_free.json').read_text())
    source = Path(selection['models'][selection['displayed']['signed']]['fit_path'])
    data_path = Path('artifacts/swirl/swirl_free_dim4/data.npz')
    valid = np.load(data_path)['valid']
    original = read_model(source)
    widths = np.ptp(np.asarray(original.domain), axis=1)
    protocol = {'source': str(source), 'source_sha256': digest(source), 'data_sha256': digest(data_path),
                'fractions_of_box_width': FRACTIONS, 'selection': 'minimum held-out least-squares risk',
                'operation': 'mass-preserving Galerkin heat smoothing in the existing Chebyshev TT; truncate residual negatives only afterward',
                'changes_original_fits': False, 'truth_used_for_selection': False,
                'new_test_seed': 61492, 'order': 129}
    protocol = json.loads(json.dumps(protocol))
    if (ROOT/'protocol.json').exists() and json.loads((ROOT/'protocol.json').read_text()) != protocol:
        raise ValueError('Smoothing protocol changed')
    record(ROOT/'protocol.json', protocol)
    results = {}
    for fraction in FRACTIONS:
        name = f'heat_{fraction:g}'
        folder = ROOT/name
        folder.mkdir(exist_ok=True)
        if (folder/'validation.json').exists():
            results[name] = json.loads((folder/'validation.json').read_text())
            assert digest(folder/'fit.npz') == results[name]['fit_sha256']
            continue
        smooth = heat_smooth_tt(original, fraction*widths)
        save_model(folder/'fit.npz', smooth)
        op, weighted, mass, negative = grid_weights(smooth, 'signed', np.eye(4)[None], [1.], 129)
        if abs(mass-negative-1) > 1e-7:
            raise FloatingPointError('Spatial filter did not conserve signed mass')
        q2 = float(jnp.sum(weighted**2/(op.pair_weights[0][:, None]*op.pair_weights[1][None])))
        predicted = values_at(smooth, 'signed', valid, mass)
        np.save(folder/'validation.npy', predicted)
        results[name] = {'fraction': fraction, 'fit_sha256': digest(folder/'fit.npz'),
                         'positive_mass': mass, 'negative_mass': negative, 'norm_q_squared': q2,
                         'validation_risk': q2-2*float(predicted.mean()), **nll_summary(predicted)}
        record(folder/'validation.json', results[name])
        np.savez(folder/'marginals.npz', nodes=np.asarray(op.nodes),
                 pair01=np.asarray((weighted.sum(1)/op.pair_weights[0]).reshape(129, 129).T),
                 pair23=np.asarray((weighted.sum(0)/op.pair_weights[1]).reshape(129, 129).T))
        print('SCORE', name, results[name], flush=True)
        del smooth, op, weighted
        jax.clear_caches()
    chosen = min(results, key=lambda n: results[n]['validation_risk'])
    checks = {}
    for name in sorted({chosen, 'heat_0'}):
        model = read_model(ROOT/name/'fit.npz')
        op, weighted, mass, negative = grid_weights(model, 'signed', np.eye(4)[None], [1.], 145)
        q2 = float(jnp.sum(weighted**2/(op.pair_weights[0][:, None]*op.pair_weights[1][None])))
        risk = q2-2*float(values_at(model, 'signed', valid, mass).mean())
        checks[name] = {'order': 145, 'mass_change': mass-results[name]['positive_mass'],
                        'risk_change': risk-results[name]['validation_risk']}
        if abs(checks[name]['mass_change']) > 5e-4 or abs(checks[name]['risk_change']) > 1e-4:
            raise FloatingPointError('Refine smoothing validation quadrature before selecting')
        del model, op, weighted
        jax.clear_caches()
    record(ROOT/'selection.json', {'selected': chosen, 'models': results, 'quadrature_checks': checks,
                                   'truth_used_for_selection': False})
    files = [ROOT/'protocol.json', ROOT/'selection.json']
    files += list(ROOT.glob('*/fit.npz'))+list(ROOT.glob('*/validation.json'))
    frozen = {'files': {str(p): digest(p) for p in files}, 'selected': chosen, 'truth_used_for_selection': False}
    if (ROOT/'frozen.json').exists() and json.loads((ROOT/'frozen.json').read_text()) != frozen:
        raise ValueError('Frozen smoothing fits changed')
    record(ROOT/'frozen.json', frozen)
    print('FROZEN', chosen, flush=True)


def evaluate():
    frozen = json.loads((ROOT/'frozen.json').read_text())
    for path, expected in frozen['files'].items():
        assert digest(path) == expected
    selection = json.loads((ROOT/'selection.json').read_text())
    from ftde.swirl import (
        TUBE_STD,
        curve,
        path_quadrature,
        sample_swirl,
        swirl_log_density,
        swirl_squared_norm,
    )
    t, weights = path_quadrature(1600)
    means, p2 = curve(t, 4), swirl_squared_norm(4, order=1600)
    test = sample_swirl(61492, 8192, 4)
    profile_t = np.linspace(0., 1., 2001)
    points = curve(profile_t, 4)
    truth = np.exp(swirl_log_density(points, order=1600))
    np.savez(ROOT/'truth.npz', t=profile_t, density=truth)
    results = {}
    # Evaluate the entire already-frozen grid as a diagnostic, never reselect
    # a smoothing amount using these clean-density errors.
    for name in selection['models']:
        folder = ROOT/name
        r = selection['models'][name]
        model = read_model(folder/'fit.npz')
        op, weighted, mass, negative = grid_weights(model, 'signed', np.eye(4)[None]*TUBE_STD**2, [1.], 145)
        @jax.jit
        def cross_at(grid, y, op=op):
            return op.convolve_weighted(grid, op.noise_factors(y))
        cross = 0.
        for start in range(0, len(t), 128):
            count = len(means[start:start+128])
            padded = np.pad(means[start:start+128], ((0, 128-count), (0, 0)))
            cross += float(weights[start:start+count]@np.asarray(cross_at(weighted, jnp.asarray(padded)))[:count])
        q2 = float(jnp.sum(weighted**2/(op.pair_weights[0][:, None]*op.pair_weights[1][None])))
        error = float(np.sqrt(max(0., (q2-2*cross+p2)/p2)))
        predicted = values_at(model, 'signed', test, mass)
        np.save(folder/'test.npy', predicted)
        np.savez(folder/'profile.npz', raw=np.asarray(model(jnp.asarray(points))),
                 density=values_at(model, 'signed', points, mass))
        results[name] = {**r, 'order': 145, 'positive_mass': mass, 'negative_mass': negative, 'norm_q_squared': q2,
                         'relative_l2': error, 'test': nll_summary(predicted)}
        print('EVALUATED', name, results[name], flush=True)
        del model, op, weighted
        jax.clear_caches()
    record(ROOT/'evaluation.json', {'selected': selection['selected'], 'models': results, 'test_seed': 61492})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('score', 'evaluate', 'all'), default='all', nargs='?')
    stage = parser.parse_args().stage
    if stage in ('score', 'all'):
        score()
    if stage in ('evaluate', 'all'):
        evaluate()
