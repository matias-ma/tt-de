"""Freeze observation-only choices, then evaluate the noisy-density controls."""
import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np
from noise_diagnosis import ROOT, SIGMAS, GaussianMomentProblem
from objective_study import CASES, noise_parameters

from ftde.direct_training import (
    DirectTrainingProblem,
    NoisyDirectTrainingProblem,
    load_tt,
    write_record,
)
from ftde.marginals import squared_pair_grid
from ftde.tt_convolution import TTConvolution


def read(path):
    return json.loads(Path(path).read_text())


def paired_se(values, case):
    """Account for repeated GL samples from each of 200 MCMC chains."""
    values = np.asarray(values)
    if not case.startswith('gl'):
        return float(values.std(ddof=1)/np.sqrt(len(values)))
    sums = np.bincount(np.arange(len(values)) % 200, weights=values-values.mean(), minlength=200)
    return float(np.sqrt(200/199*np.sum(sums**2))/len(values))


def gl_observed_logs(points, weights, covariance, log_z, *, order=96, radius=3.):
    """Truth-only GL transfer quadrature under diagonal Gaussian-mixture noise."""
    from scipy.special import logsumexp, roots_legendre

    from ftde.ginzburg_landau import GL_BETA, GL_LAM
    dim = points.shape[1]
    nodes, quadrature = roots_legendre(order)
    nodes, quadrature = radius*nodes, radius*quadrature
    edge_scale = GL_BETA*GL_LAM*(dim+1)**2/2
    edges = np.exp(-edge_scale*(nodes[:, None]-nodes[None, :])**2)
    boundary = np.exp(-edge_scale*nodes**2)
    site = -GL_BETA*(1-nodes**2)**2/(4*GL_LAM)+np.log(quadrature)
    answers = []
    for weight, cov in zip(weights, covariance, strict=True):
        if not np.array_equal(cov, np.diag(np.diag(cov))):
            raise ValueError('Truth transfer here requires diagonal noise components')
        state = np.broadcast_to(boundary, (len(points), order)).copy()
        scale = np.zeros(len(points))
        for k in range(dim):
            if k:
                state = state@edges
            unary = site[None]-.5*(points[:, k, None]-nodes)**2/cov[k, k]-.5*np.log(2*np.pi*cov[k, k])
            peak = unary.max(axis=1)
            state *= np.exp(unary-peak[:, None])
            norm = state.max(axis=1)
            state /= norm[:, None]
            scale += peak+np.log(norm)
        answers.append(np.log(weight)+scale+np.log(state@boundary)-log_z)
    return logsumexp(np.stack(answers), axis=0)


def freeze():
    if (ROOT/'frozen.json').exists():
        for path, digest in read(ROOT/'frozen.json')['files'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
        return
    expected = {'swirl4_mixture': ['seed0', 'seed1', 'seed2', 'longer_adam', 'lbfgs'],
                'gl6_mixture': ['seed1', 'longer_adam', 'lbfgs'],
                'gl6_free': ['seed0'], 'gl6_wide_free': ['seed0', 'modes49'],
                **{f'gaussian_{sigma:g}': ['seed0'] for sigma in SIGMAS}}
    cases, files = {}, {}
    for case, names in expected.items():
        if case.startswith('gaussian'):
            data_path = ROOT/case/'observations.npz'
        else:
            data_path = Path(CASES['gl6_free' if case == 'gl6_wide_free' else case])/'data.npz'
        files[str(data_path)] = hashlib.sha256(data_path.read_bytes()).hexdigest()
        runs = {name: ROOT/case/name for name in names}
        if case.endswith('mixture'):
            capacity = 'r32_n49' if case.startswith('swirl') else 'r10_n25'
            runs['old_adam'] = Path('artifacts/objective_v3')/case/f'nll_{capacity}_l0.001_w100'
        models, all_logs = {}, {}
        for name, path in runs.items():
            result = read(path/'result.json')
            config = read(path/'config.json')
            logpath = path/('validation.npz' if name == 'old_adam' else 'validation.npy')
            logs = np.load(logpath)
            if name == 'old_adam':
                logs = logs['logs']
            all_logs[name] = np.asarray(logs)
            if not np.isfinite(logs).all():
                raise FloatingPointError('Nonfinite validation score')
            np.testing.assert_allclose(-logs.mean(), result['best_nll'], atol=1e-12)
            models[name] = {'path': str(path/'fit.npz'), 'config': config, 'result': result,
                            'validation_nll': -float(logs.mean())}
            for p in [path/'fit.npz', path/'result.json', path/'config.json', logpath]:
                files[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        selected = min(models, key=lambda n: models[n]['validation_nll'])
        if 'old_adam' in models and models['old_adam']['validation_nll'] <= models[selected]['validation_nll']+1e-12:
            selected = 'old_adam'
        for name, info in models.items():
            difference = all_logs[selected]-all_logs[name]
            info['excess_validation_nll'] = float(difference.mean())
            info['paired_validation_se'] = paired_se(difference, case)
        # Inspect where deterministic optimization actually ended, even if
        # validation rejected that endpoint. This diagnostic cannot select
        # a model or influence any additional training.
        if 'lbfgs' in runs:
            path = runs['lbfgs']/'last.npz'
            record = models['lbfgs']['result']
            models['lbfgs_last'] = {'path': str(path), 'config': models['lbfgs']['config'],
                'result': record, 'validation_nll': record['history'][-1]['validation_nll'],
                'eligible_for_selection': False, 'diagnostic': 'final iterate, not validation-selected checkpoint'}
            files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        cases[case] = {'models': models, 'selected': selected}
    write_record(ROOT/'frozen.json', {'cases': cases, 'files': files,
        'selection': 'minimum observed validation NLL within each fixed-capacity case',
        'truth_evaluation_not_yet_run': True})
    print('FROZEN', {c: v['selected'] for c, v in cases.items()}, flush=True)


def evaluate(case):
    frozen = read(ROOT/'frozen.json')
    for path, digest in frozen['files'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    case_info = frozen['cases'][case]
    swirl = case.startswith(('swirl', 'gaussian'))
    dim = 4 if swirl else 6
    # Truth appears only below the freeze verification, exclusively for evaluation.
    if swirl:
        from swirl_figures import integrated_l2

        from ftde.swirl import (
            sample_swirl,
            swirl_convolved_log_density,
            swirl_log_density,
            swirl_pair_marginal,
        )
        clean = sample_swirl(732981, 4096, dim)
    else:
        from evaluate_gl_l2 import integrated_terms, relative_l2_from_terms
        from gl_common import _true_marginal_grid

        from ftde.ginzburg_landau import (
            GL_BETA,
            ginzburg_landau_log_normalizer,
            sample_ginzburg_landau,
        )
        clean = np.asarray(sample_ginzburg_landau(732981, 4096, dim))
        log_z = ginzburg_landau_log_normalizer(dim)
        p2 = float(np.exp(ginzburg_landau_log_normalizer(dim, beta=2*GL_BETA)-2*log_z))
    if case.startswith('gaussian'):
        sigma = read(ROOT/case/'data_protocol.json')['sigma']
        weights, covariance = np.ones(1), np.eye(dim)[None]*sigma**2
        train = np.load(ROOT/case/'observations.npz')['train']
        test = clean+sigma*np.random.default_rng(740981).normal(size=clean.shape)
        noisy = sigma > 0
    else:
        source = 'gl6_free' if case == 'gl6_wide_free' else case
        base = Path(CASES[source])
        train = np.load(base/'data.npz')['train']
        weights, covariance = noise_parameters(source, dim)
        noisy = not case.endswith('free')
        test = clean.copy()
        if noisy:
            rng = np.random.default_rng(732982)
            labels = rng.choice(len(weights), len(test), p=weights)
            test += np.einsum('nij,nj->ni', np.linalg.cholesky(covariance)[labels], rng.normal(size=test.shape))
    grid = np.linspace(-2.6 if swirl else -2.5, 2.6 if swirl else 2.5, 181)
    if swirl:
        oracle = (swirl_convolved_log_density(test, covariance, weights, chunk=128) if noisy
                  else swirl_log_density(test, chunk=128))
        truth = {f'pair{i}{j}': swirl_pair_marginal(dim, i, j, grid, grid) for i, j in [(0, 1), (2, 3)]}
    else:
        from ftde.ginzburg_landau import ginzburg_landau_log_prob
        if noisy:
            oracle = gl_observed_logs(test, weights, covariance, log_z)
            fine = gl_observed_logs(test, weights, covariance, log_z, order=192)
            wider = gl_observed_logs(test[:32], weights, covariance, log_z, order=256, radius=4.)
            oracle_errors = {'double_order': float(np.max(np.abs(oracle-fine))),
                             'wider_domain': float(np.max(np.abs(wider-fine[:32])))}
            if max(oracle_errors.values()) > 1e-8:
                raise FloatingPointError('Observed GL truth failed quadrature refinement')
            write_record(ROOT/case/'oracle_quadrature.json', oracle_errors)
            oracle = fine
        else:
            oracle = np.asarray(ginzburg_landau_log_prob(test))-log_z
        truth = {f'pair{i}{j}': _true_marginal_grid(dim, i, j, np.tile([grid[0], grid[-1]], (dim, 1)), log_z, n_grid=len(grid))
                 for i, j in [(0, 1), (2, 3)]}
    np.savez_compressed(ROOT/case/'truth.npz', grid=grid, **truth)
    results, test_logs = {}, {}
    for name, info in case_info['models'].items():
        output = ROOT/case/f'evaluation_{name}.json'
        logs_path = ROOT/case/f'test_logs_{name}.npy'
        if output.exists():
            results[name], test_logs[name] = read(output), np.load(logs_path)
            continue
        model = load_tt(info['path'])
        if swirl:
            metric = integrated_l2(ROOT/case, model, dim, name)
            if metric['path_quadrature_relative_change'] > 1e-6 or metric['relative_squared_l2'] < -1e-9:
                raise FloatingPointError('Swirl integration failed')
        else:
            order = max(98, 2*max(c.coeffs.shape[-1] for c in model.cores))
            a = tuple(map(float, jax.jit(lambda model=model, order=order: integrated_terms(model, order, log_z))()))
            b = tuple(map(float, jax.jit(lambda model=model, order=order: integrated_terms(model, order*3//2, log_z))()))
            error = max(abs(a[0]-b[0]), abs(a[1]-b[1]))/p2
            if error > 1e-6:
                raise FloatingPointError('GL integration failed')
            metric = {'relative_l2': relative_l2_from_terms(*b, p2), 'quadrature_relative_change': error}
        if not noisy:
            problem = DirectTrainingProblem(model, strength=.001, regularizer='hessian')
        elif case.startswith('gaussian'):
            problem = GaussianMomentProblem(model, np.full(dim, sigma), strength=.001, regularizer='hessian')
        else:
            observations = np.concatenate([test, train[:20000]])
            obs_box = np.stack([observations.min(0)-.02, observations.max(0)+.02], axis=1)
            forward = TTConvolution.from_tt(model, load_tt(base/'kernel.npz')).prepare(obs_box, tolerance=1e-14)
            problem = NoisyDirectTrainingProblem(model, forward, strength=.001, regularizer='hessian')
        def score(points, problem=problem):
            features = problem.features(points)
            return np.concatenate([np.asarray(problem.logs(problem.params, tuple(x[s:s+128] for x in features)))
                                   for s in range(0, len(points), 128)])
        logs = score(test)
        energies = np.asarray(problem.energies(problem.params))
        metric.update(test_nll=-float(logs.mean()), training_subset_nll=-float(score(train[:20000]).mean()),
                      training_subset_count=20000, roughness_hessian=float(energies[3]/energies[0]))
        if oracle is not None:
            excess = oracle-logs
            metric.update(estimated_observed_kl=float(excess.mean()), observed_kl_se=paired_se(excess, case),
                          true_observed_test_nll=-float(oracle.mean()))
        if case.startswith('gaussian') and noisy:
            # Refine the actual evaluation integrals, including observed tails.
            order = 2*np.asarray(problem.quadrature.orders)
            refined = GaussianMomentProblem(model, np.full(dim, sigma), quadrature_order=order, strength=.001, regularizer='hessian')
            ids = np.unique(np.r_[np.arange(24), np.argsort(np.linalg.norm(test, axis=1))[-8:]])
            fine = refined.logs(refined.params, refined.features(test[ids]))
            metric['gaussian_quadrature_refinement'] = float(np.max(np.abs(np.asarray(fine)-logs[ids])))
            if metric['gaussian_quadrature_refinement'] > 1e-8:
                raise FloatingPointError('Gaussian moment quadrature failed refinement')
        marginal = {f'pair{i}{j}': squared_pair_grid(model, i, j, grid, grid) for i, j in [(0, 1), (2, 3)]}
        for pair, values in marginal.items():
            metric[f'{pair}_relative_l2'] = float(np.linalg.norm(values-truth[pair])/np.linalg.norm(truth[pair]))
        np.savez_compressed(ROOT/case/f'marginals_{name}.npz', grid=grid, **marginal)
        np.save(logs_path, logs)
        write_record(output, metric)
        results[name], test_logs[name] = metric, logs
        print('EVALUATED', case, name, metric, flush=True)
        jax.clear_caches()
    reference = 'old_adam' if 'old_adam' in results else case_info['selected']
    differences = {}
    for name, logs in test_logs.items():
        delta = test_logs[reference]-logs
        differences[name] = {'test_nll_minus_reference': float(delta.mean()), 'paired_se': paired_se(delta, case)}
    write_record(ROOT/case/'evaluation.json', {'models': results, 'selected': case_info['selected'],
        'test_comparisons_reference': reference, 'test_comparisons': differences, 'test_count': len(test),
        'truth_used_only_after_freeze': True,
        'standard_errors': 'clustered by 200 MCMC chains' if case.startswith('gl') else 'IID paired observations'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('case')
    args = p.parse_args()
    if args.case == 'freeze':
        freeze()
    else:
        evaluate(args.case)
