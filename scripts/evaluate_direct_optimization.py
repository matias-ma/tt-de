"""Independent evaluation AFTER noisy-data selection; no fitting in this script."""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from fit_direct_optimization import prepare_problem, record
from refresh_density_fits import read_model
from score_direct_optimization import grid_weights, log_predictions

from ftde.clipped_tt_density import ClippedTTQuadrature4D
from ftde.marginals import squared_pair_grid
from ftde.tt_convolution import TTConvolution
from ftde.tt_density import log_density_tt, tt_params
from ftde.tt_sampling import make_squared_tt_moments, make_squared_tt_sampler


def freeze(root, specs, selection):
    tracked = {path for _, path, _ in specs}
    for _, path, _ in specs:
        tracked.update(p for name in ('init.npz', 'last.npz', 'result.json', 'config.json')
                       if (p := path.parent/name).exists())
    value = {'selection': selection, 'files': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                             for path in sorted(tracked)},
             'truth_used_for_selection': False, 'independent_test_seed': 87329}
    path = root/'frozen_evaluation.json'
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Frozen evaluation choices changed')
    else:
        record(path, value)


def evaluate_four(base, root, order, audit_order):
    data, _domain, covs, weights = prepare_problem(base, root)
    selected = json.loads((root/'selection_q97.json').read_text())
    specs = [(name, Path(r['fit_path']), r['kind']) for name, r in selected['results'].items()]
    specs.append(('uniform_start', root/'squared_n25_r16_l0.0001_b256_s0/init.npz', 'squared'))
    freeze(root, specs, selected)
    # Independent tests focus on validation-selected method representatives.
    # All trials remain frozen and retain their observation-only scores.
    families = [
        [n for n in selected['names'] if n.startswith('signed') and '_neg' not in n],
        [n for n in selected['names'] if n.startswith('signed') and '_neg' in n],
        [n for n in selected['names'] if n.startswith('squared') and not n.endswith('_lbfgs')],
        [n for n in selected['names'] if n.endswith('_lbfgs')],
    ]
    representatives = {min(names, key=lambda n: selected['results'][n]['validation_nll']) for names in families}
    representatives.update([selected['new_selected'], selected['penalized_selected'], 'previous_squared', 'uniform_start'])
    specs = [spec for spec in specs if spec[0] in representatives]
    # Known signal and fresh noisy data enter ONLY after the choices freeze.
    from swirl_fits import observations

    from ftde.swirl import TUBE_STD, curve, path_quadrature, swirl_squared_norm
    test = observations(4, 'mixture', 8192, 87329)
    t, path_weights = path_quadrature(1600)
    means = curve(t, 4)
    p2 = swirl_squared_norm(4, order=1600)
    results = {}
    for name, path, kind in specs:
        out = root/name
        out.mkdir(exist_ok=True)
        destination = out/f'evaluation_q{order}.json'
        if destination.exists():
            results[name] = json.loads(destination.read_text())
            continue
        model = read_model(path)
        op, weighted, mass, negative = grid_weights(model, kind, covs, weights, order)
        logs = log_predictions(op, weighted, test)
        np.save(out/f'test_q{order}.npy', logs)
        truth_op = ClippedTTQuadrature4D.from_tt(model, np.eye(4)[None]*TUBE_STD**2, [1.], order=order)
        @jax.jit
        def true_cross(grid, y, truth_op=truth_op):
            return truth_op.convolve_weighted(grid, truth_op.noise_factors(y))
        cross = 0.
        for start in range(0, len(t), 128):
            y = means[start:start+128]
            count = len(y)
            y = np.pad(y, ((0, 128-count), (0, 0)))
            values = np.asarray(true_cross(weighted, jnp.asarray(y)))[:count]
            cross += float(path_weights[start:start+count]@values)
        qw = op.pair_weights[0][:, None]*op.pair_weights[1][None]
        q2 = float(jnp.sum(weighted**2/qw))
        if q2-2*cross+p2 < -1e-8*p2:
            raise FloatingPointError('Inconsistent L2 quadrature integrals')
        relative_l2 = float(np.sqrt(max(0., (q2-2*cross+p2)/p2)))
        r = {'relative_l2': relative_l2, 'test_nll': -float(logs.mean()), 'positive_mass': mass,
             'negative_mass': negative, 'norm_p_squared': p2, 'norm_q_squared': q2,
             'cross_integral': cross, 'order': order, 'kind': kind, 'fit_path': str(path),
             'fit_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        ridge_values = np.asarray(model(jnp.asarray(curve(np.linspace(0., 1., 2001), 4))))
        r['raw_tt_sign_changes_on_true_curve'] = int(np.sum(ridge_values[:-1]*ridge_values[1:] < 0))
        r['raw_tt_negative_fraction_on_true_curve'] = float(np.mean(ridge_values < 0))
        np.savez(out/f'marginals_q{order}.npz', nodes=np.asarray(op.nodes),
                 pair01=np.asarray((weighted.sum(1)/op.pair_weights[0]).reshape(order, order).T),
                 pair23=np.asarray((weighted.sum(0)/op.pair_weights[1]).reshape(order, order).T))
        record(destination, r)
        results[name] = r
        print('EVALUATION', name, r, flush=True)
        jax.clear_caches()
    reference = np.load(root/'previous_squared'/f'test_q{order}.npy')
    for name, r in results.items():
        gain = np.load(root/name/f'test_q{order}.npy')-reference
        r['test_gain_vs_previous'] = float(gain.mean())
        r['test_gain_paired_se'] = float(gain.std(ddof=1)/np.sqrt(len(gain)))
    record(root/f'evaluation_q{order}.json', {'selection': selected['new_selected'], 'models': results})
    if audit_order:
        # Reintegration of frozen selected candidates; no optimization or reselection.
        signed = [name for name in selected['names'] if name.startswith('signed')]
        best_signed = min(signed, key=lambda name: selected['results'][name]['validation_nll'])
        audit_names = sorted({best_signed, selected['penalized_selected'], selected['new_selected'], 'previous_squared'})
        checks = {}
        for name in audit_names:
            r = selected['results'][name]
            model = read_model(r['fit_path'])
            op, weighted, mass, negative = grid_weights(model, r['kind'], covs, weights, audit_order)
            high = log_predictions(op, weighted, data['valid'][:256])
            low_op, low_weighted, _, _ = grid_weights(model, r['kind'], covs, weights, order)
            low = log_predictions(low_op, low_weighted, data['valid'][:256])
            truth_op = ClippedTTQuadrature4D.from_tt(model, np.eye(4)[None]*TUBE_STD**2, [1.], order=audit_order)
            @jax.jit
            def high_cross(grid, y, truth_op=truth_op):
                return truth_op.convolve_weighted(grid, truth_op.noise_factors(y))
            cross = 0.
            for start in range(0, len(t), 128):
                y = means[start:start+128]
                count = len(y)
                y = np.pad(y, ((0, 128-count), (0, 0)))
                cross += float(path_weights[start:start+count]@np.asarray(high_cross(weighted, jnp.asarray(y)))[:count])
            q2 = float(jnp.sum(weighted**2/(op.pair_weights[0][:, None]*op.pair_weights[1][None])))
            high_l2 = float(np.sqrt(max(0., (q2-2*cross+p2)/p2)))
            checks[name] = {'low_order': order, 'high_order': audit_order,
                           'maximum_log_difference': float(np.max(np.abs(high-low))),
                           'mean_log_difference': float(np.mean(high-low)),
                           'positive_mass_change': mass-results[name]['positive_mass'],
                           'negative_mass_change': negative-results[name]['negative_mass'],
                           'high_order_relative_l2': high_l2,
                           'relative_l2_change': high_l2-results[name]['relative_l2']}
            print('QUADRATURE', name, checks[name], flush=True)
            jax.clear_caches()
        record(root/'clipping_quadrature.json', checks)


def evaluate_ten(base, root, samples=16384):
    _data, domain, _covs, _weights = prepare_problem(base, root)
    runs = sorted(p for p in root.glob('squared_*') if (p/'result.json').exists())
    specs = [(p.name, p/'fit.npz', 'squared') for p in runs]
    specs += [('previous_squared', base/'selected.npz', 'squared'), ('uniform_start', runs[0]/'init.npz', 'squared')]
    # Selection is the minimum common regularized validation score, using
    # only the recorded observation-based checkpoints.
    new_selected = min(runs, key=lambda p: json.loads((p/'result.json').read_text()).get('selection_score',
                            json.loads((p/'result.json').read_text()).get('best_score', np.inf))).name
    freeze(root, specs, {'new_selected': new_selected})
    from swirl_fits import observations

    from ftde.swirl import (
        TUBE_STD,
        curve,
        path_quadrature,
        sample_swirl,
        swirl_squared_norm,
    )
    noise_free = _covs is None
    test, truth_draws = observations(10, 'free' if noise_free else 'mixture', 8192, 87329), sample_swirl(87330, samples, 10)
    t, w = path_quadrature(1600)
    centers = jnp.asarray(curve(t, 10))
    @jax.jit
    def truth_logs(x):
        d2 = jnp.maximum(jnp.sum(x*x, axis=1)[:, None]+jnp.sum(centers*centers, axis=1)[None]-2*x@centers.T, 0)
        return jax.scipy.special.logsumexp(jnp.log(jnp.asarray(w))[None]-d2/(2*TUBE_STD**2), axis=1)-5*jnp.log(2*jnp.pi*TUBE_STD**2)
    def batches(fn, x, size=256):
        return np.concatenate([np.asarray(fn(jnp.asarray(x[k:k+size]))) for k in range(0, len(x), size)])
    lp_truth = batches(truth_logs, truth_draws)
    profile_t = np.linspace(0., 1., 2001)
    profile_points = curve(profile_t, 10)
    profile_truth = np.exp(batches(truth_logs, profile_points))
    np.savez(root/'truth_profile.npz', t=profile_t, density=profile_truth)
    p2 = swirl_squared_norm(10, order=1600)
    results = {}
    for name, path, _ in specs:
        out = root/name
        out.mkdir(exist_ok=True)
        if (out/'evaluation.json').exists():
            results[name] = json.loads((out/'evaluation.json').read_text())
            continue
        model = read_model(path)
        parameters = tt_params(model)
        sampler = make_squared_tt_sampler(model)
        uniforms = np.random.default_rng(87331).uniform(size=(samples, 10))
        fitted_draws = batches(lambda u, sampler=sampler, parameters=parameters: sampler(parameters, u), uniforms)
        @jax.jit
        def fitted_logs(x, model=model):
            inside = jnp.all((x >= domain[:, 0]) & (x <= domain[:, 1]), axis=1)
            return jnp.where(inside, log_density_tt(model, jnp.clip(x, domain[:, 0], domain[:, 1])), -jnp.inf)
        lp = np.r_[lp_truth, batches(truth_logs, fitted_draws)]
        lq = np.r_[batches(fitted_logs, truth_draws), batches(fitted_logs, fitted_draws)]
        lr = np.logaddexp(lp, lq)-np.log(2.)
        peak = np.maximum(lp, lq)
        terms = np.exp(2*peak-lr)*(np.exp(lp-peak)-np.exp(lq-peak))**2/p2
        squared = float(.5*(terms[:samples].mean()+terms[samples:].mean()))
        se = .5*np.sqrt(terms[:samples].var(ddof=1)/samples+terms[samples:].var(ddof=1)/samples)
        means, variances = map(np.asarray, make_squared_tt_moments(model)(parameters))
        mean_z = float(np.max(np.abs(fitted_draws.mean(0)-means)/np.sqrt(variances/samples)))
        if noise_free:
            logs = batches(fitted_logs, test, 128)
            discrepancy = 0.
        else:
            kernel = read_model(base/'kernel.npz')
            op = TTConvolution.from_tt(model, kernel).prepare(tolerance=1e-14)
            noisy = jax.jit(lambda x, op=op, model=model: op.log_density(model, x))
            logs = batches(noisy, test, 128)
            fine = TTConvolution.from_tt(model, kernel, quadrature_order=2*np.asarray(op.orders))
            discrepancy = float(np.max(np.abs(logs[:8]-np.asarray(fine.log_density(model, jnp.asarray(test[:8]))))))
        if not np.all(np.isfinite(logs) | (noise_free & np.isneginf(logs))) or mean_z > 6:
            raise FloatingPointError('Independent prediction or sampler check failed')
        if discrepancy > 1e-7:
            raise FloatingPointError('Independent convolution check failed')
        np.save(out/'test.npy', logs)
        norm_terms = np.exp(2*lp-lr)/p2
        norm_se = .5*np.sqrt(norm_terms[:samples].var(ddof=1)/samples+norm_terms[samples:].var(ddof=1)/samples)
        if abs(norm_terms.mean()-1) > 6*norm_se:
            raise FloatingPointError('Known squared-norm importance check failed')
        r = {'relative_l2': np.sqrt(squared), 'relative_l2_mc_se': se/(2*np.sqrt(squared)),
             'test_nll': -float(logs.mean()) if np.isfinite(logs).all() else None,
             'test_zero_predictions': int(np.sum(np.isneginf(logs))), 'max_sample_mean_z': mean_z,
             'p2_importance_ratio': float(norm_terms.mean()), 'quadrature_log_discrepancy': discrepancy,
             'p2_ratio_mc_se': float(norm_se),
             'fit_path': str(path), 'fit_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        raw_profile = np.asarray(model(jnp.asarray(profile_points)))
        profile = np.exp(batches(fitted_logs, profile_points))
        np.savez(out/'profile.npz', density=profile, raw_tt=raw_profile)
        r['raw_tt_sign_changes_on_true_curve'] = int(np.sum(raw_profile[:-1]*raw_profile[1:] < 0))
        r['minimum_density_ratio_on_true_curve'] = float(np.min(profile/profile_truth))
        r['fraction_true_curve_below_half_density'] = float(np.mean(profile < .5*profile_truth))
        grid = np.linspace(-2.5, 2.5, 161)
        marginals = {}
        for i, j in [(0, 1), (2, 3)]:
            values = squared_pair_grid(model, i, j, np.clip(grid, *domain[i]), np.clip(grid, *domain[j]))
            inside = ((grid >= domain[j, 0]) & (grid <= domain[j, 1]))[:, None]
            inside = inside & ((grid >= domain[i, 0]) & (grid <= domain[i, 1]))[None]
            marginals[f'pair{i}{j}'] = np.where(inside, values, 0.)
        np.savez(out/'marginals.npz', nodes=np.tile(grid, (10, 1)),
                 **marginals)
        record(out/'evaluation.json', r)
        results[name] = r
        print('EVALUATION10', name, r, flush=True)
        jax.clear_caches()
    reference = np.load(root/'previous_squared/test.npy')
    for name, r in results.items():
        gain = np.load(root/name/'test.npy')-reference
        r['test_gain_vs_previous'] = float(gain.mean()) if np.isfinite(gain).all() else None
        r['test_gain_paired_se'] = float(gain.std(ddof=1)/np.sqrt(len(gain))) if np.isfinite(gain).all() else None
    record(root/'evaluation.json', {'selection': new_selected, 'models': results})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dim', type=int, choices=[4, 10], default=4)
    p.add_argument('--noise-free', action='store_true', help='10D noise-free control (4D has a dedicated evaluator)')
    p.add_argument('--order', type=int, default=97)
    p.add_argument('--audit-order', type=int, default=129)
    args = p.parse_args()
    if args.noise_free and args.dim != 10:
        raise ValueError('Use evaluate_direct_noise_free.py for the 4D controls')
    base = Path(f'artifacts/swirl/swirl_{"free" if args.noise_free else "mixture"}_dim{args.dim}')
    root = Path(f'artifacts/direct_optimization/swirl{"_free" if args.noise_free else ""}{args.dim}')
    if args.dim == 4:
        evaluate_four(base, root, args.order, args.audit_order)
    else:
        evaluate_ten(base, root)
