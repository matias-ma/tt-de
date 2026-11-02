"""Score on held-out observations, freeze, then evaluate noise-free TT fits."""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from evaluate_direct_optimization import freeze
from fit_direct_optimization import prepare_problem, record
from refresh_density_fits import read_model
from score_direct_optimization import grid_weights

BASE = Path('artifacts/swirl/swirl_free_dim4')
ROOT = Path('artifacts/direct_optimization/swirl_free4')


def values_at(model, kind, points, mass):
    box = np.asarray(model.domain)
    @jax.jit
    def evaluate(x):
        inside = jnp.all((x >= box[:, 0]) & (x <= box[:, 1]), axis=1)
        raw = model(jnp.clip(x, box[:, 0], box[:, 1]))
        return jnp.where(inside, (raw**2 if kind == 'squared' else jnp.maximum(raw, 0))/mass, 0.)
    return np.concatenate([np.asarray(evaluate(jnp.asarray(points[s:s+512]))) for s in range(0, len(points), 512)])


def nll_summary(values):
    bad = int(np.sum(values <= 0))
    return {'nll': -float(np.log(values).mean()) if bad == 0 else None, 'nonpositive_predictions': bad}


def score():
    data, _, _, _ = prepare_problem(BASE, ROOT)
    required = ['squared_n25_r16_l0.0001_b1024_s0', 'squared_n49_r32_l0_b1024_s0',
                'squared_n49_r32_l1e-05_b1024_s0', 'squared_n49_r32_l0.0001_b1024_s0',
                'squared_n49_r32_l0.0001_b1024_s1', 'squared_n65_r32_l0.0001_b1024_s0',
                'squared_n49_r32_l0.001_b1024_s0', 'squared_n49_r32_l0.001_b1024_s0_lbfgs',
                'squared_n49_r64_l0.0001_b1024_s0', 'squared_n49_r32_l0.0001_b1024_s0_lbfgs',
                'signed_n49_r32_l1e-05_s0_neg1']
    required += [f'signed_n{n}_r{r}_l{strength:g}_s0' for n, r in [(25, 16), (49, 32), (49, 64)]
                 for strength in (1e-6, 1e-5, 1e-4)]
    if any(not (ROOT/name/'result.json').exists() for name in required):
        raise RuntimeError('Complete the observation-only noise-free controls before freezing')
    runs = sorted(p for p in ROOT.glob('*') if (p/'result.json').exists())
    specs = [(p.name, p/'fit.npz', 'signed' if p.name.startswith('signed') else 'squared') for p in runs]
    specs += [('previous_squared', BASE/'selected.npz', 'squared'),
              ('uniform_start', ROOT/'squared_n49_r32_l0.0001_b1024_s0/init.npz', 'squared')]
    results = {}
    for name, path, kind in specs:
        destination = ROOT/name
        destination.mkdir(exist_ok=True)
        saved = destination/'validation_free.json'
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if saved.exists():
            results[name] = json.loads(saved.read_text())
            assert results[name]['fit_sha256'] == sha
            continue
        model = read_model(path)
        # Integrate density squared: for squared amplitudes this has degree 4(n-1).
        order = max(129, 2*max(c.coeffs.shape[-1] for c in model.cores)-1)
        op, weighted, mass, negative = grid_weights(model, kind, np.eye(4)[None], [1.], order)
        qw = op.pair_weights[0][:, None]*op.pair_weights[1][None]
        q2 = float(jnp.sum(weighted**2/qw))
        predictions = values_at(model, kind, data['valid'], mass)
        raw_valid = np.concatenate([np.asarray(model(jnp.asarray(data['valid'][s:s+512])))
                                    for s in range(0, len(data['valid']), 512)])
        negative_fraction = float(np.mean(raw_valid < 0))
        np.save(destination/'validation_density.npy', predictions)
        result = {'fit_path': str(path), 'fit_sha256': sha, 'kind': kind, 'order': order,
                  'positive_mass': mass, 'negative_mass': negative, 'norm_q_squared': q2,
                  'raw_tt_negative_fraction_validation': negative_fraction,
                  'raw_tt_minority_sign_fraction_validation': min(negative_fraction, 1-negative_fraction),
                  'validation_risk': q2-2*float(predictions.mean()), **nll_summary(predictions)}
        record(saved, result)
        results[name] = result
        np.savez(destination/'marginals_free.npz', nodes=np.asarray(op.nodes),
                 pair01=np.asarray((weighted.sum(1)/op.pair_weights[0]).reshape(order, order).T),
                 pair23=np.asarray((weighted.sum(0)/op.pair_weights[1]).reshape(order, order).T))
        print('VALIDATION', name, result, flush=True)
        del op, weighted, qw, model
        jax.clear_caches()
    new = [name for name in results if name not in ('previous_squared', 'uniform_start')]
    selected = min(new, key=lambda name: results[name]['validation_risk'])
    families = {'adam': [n for n in new if n.startswith('squared') and not n.endswith('_lbfgs')],
                'lbfgs': [n for n in new if n.endswith('_lbfgs')],
                'signed': [n for n in new if n.startswith('signed') and '_neg' not in n],
                'penalized': [n for n in new if n.startswith('signed') and '_neg' in n]}
    displayed = {key: min(names, key=lambda n: results[n]['validation_risk']) for key, names in families.items()}
    selection = {'new_selected': selected, 'displayed': displayed, 'models': results,
                 'rule': 'minimum held-out L2 risk of the final normalized density; no truth used; zero likelihoods reported without flooring'}
    record(ROOT/'selection_free.json', selection)
    freeze(ROOT, specs, selection)
    print('FROZEN', selected, flush=True)


def evaluate():
    selection = json.loads((ROOT/'selection_free.json').read_text())
    specs = [(name, Path(r['fit_path']), r['kind']) for name, r in selection['models'].items()]
    freeze(ROOT, specs, selection)
    # All model choices and hashes are frozen before the first truth evaluation.
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
    test = sample_swirl(87329, 8192, 4)
    profile_t = np.linspace(0., 1., 2001)
    profile_x = curve(profile_t, 4)
    profile_truth = np.exp(swirl_log_density(profile_x, order=1600))
    np.savez(ROOT/'truth_evaluation.npz', t=profile_t, density=profile_truth)
    results = {}
    for name, path, kind in specs:
        destination = ROOT/name
        r = selection['models'][name]
        saved = destination/'evaluation_free.json'
        if saved.exists():
            results[name] = json.loads(saved.read_text())
            continue
        model = read_model(path)
        order = r['order']
        op, weighted, mass, negative = grid_weights(model, kind, np.eye(4)[None]*TUBE_STD**2, [1.], order)
        @jax.jit
        def cross_at(grid, y, op=op):
            return op.convolve_weighted(grid, op.noise_factors(y))
        cross = 0.
        for start in range(0, len(t), 128):
            points = means[start:start+128]
            count = len(points)
            points = np.pad(points, ((0, 128-count), (0, 0)))
            cross += float(weights[start:start+count]@np.asarray(cross_at(weighted, jnp.asarray(points)))[:count])
        squared = r['norm_q_squared']-2*cross+p2
        if squared < -1e-8*p2:
            raise FloatingPointError('Invalid clean-density squared error')
        predictions = values_at(model, kind, test, mass)
        raw_profile = np.asarray(model(jnp.asarray(profile_x)))
        profile = values_at(model, kind, profile_x, mass)
        np.save(destination/'test_density.npy', predictions)
        np.savez(destination/'profile_free.npz', density=profile, raw_tt=raw_profile)
        result = {**r, 'relative_l2': float(np.sqrt(max(0., squared/p2))), 'cross_integral': cross,
                  'norm_p_squared': p2, 'test': nll_summary(predictions),
                  'minimum_density_ratio_on_true_curve': float(np.min(profile/profile_truth)),
                  'fraction_true_curve_below_half_density': float(np.mean(profile < .5*profile_truth)),
                  'raw_tt_sign_changes_on_true_curve': int(np.sum(raw_profile[:-1]*raw_profile[1:] < 0)),
                  'raw_tt_negative_fraction_on_true_curve': float(np.mean(raw_profile < 0))}
        record(saved, result)
        results[name] = result
        print('EVALUATION', name, result, flush=True)
        del op, weighted, model
        jax.clear_caches()
    record(ROOT/'evaluation_free.json', {'selection': selection['new_selected'], 'displayed': selection['displayed'],
                                       'models': results})
    # Clipping is nonpolynomial. Audit its normalization and L2 independently
    # at higher quadrature, without using the result for reselection.
    checks = {}
    for name in sorted({selection['displayed']['signed'], selection['displayed']['penalized'], selection['new_selected']}):
        r = results[name]
        model = read_model(r['fit_path'])
        order = max(129, r['order']+16)
        op, weighted, mass, negative = grid_weights(model, r['kind'], np.eye(4)[None]*TUBE_STD**2, [1.], order)
        @jax.jit
        def cross_at_high(grid, y, op=op):
            return op.convolve_weighted(grid, op.noise_factors(y))
        cross = 0.
        for start in range(0, len(t), 128):
            points = means[start:start+128]
            count = len(points)
            points = np.pad(points, ((0, 128-count), (0, 0)))
            cross += float(weights[start:start+count]@np.asarray(cross_at_high(weighted, jnp.asarray(points)))[:count])
        q2 = float(jnp.sum(weighted**2/(op.pair_weights[0][:, None]*op.pair_weights[1][None])))
        error = float(np.sqrt(max(0., (q2-2*cross+p2)/p2)))
        checks[name] = {'low_order': r['order'], 'high_order': order, 'positive_mass_change': mass-r['positive_mass'],
                        'negative_mass_change': negative-r['negative_mass'], 'high_order_relative_l2': error,
                        'relative_l2_change': error-r['relative_l2']}
        print('AUDIT', name, checks[name], flush=True)
        del op, weighted, model
        jax.clear_caches()
    record(ROOT/'clipping_quadrature.json', checks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['score', 'evaluate'], required=True)
    args = parser.parse_args()
    (score if args.stage == 'score' else evaluate)()
