"""Post-freeze truth diagnostics; never fit or reselect any model here."""
import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from ftde.direct_training import DirectTrainingProblem, load_tt, write_record
from ftde.marginals import squared_pair_grid
from ftde.simple_initialization import init_tt_constant

ROOT = Path('artifacts/noise_free_v2')


def run(case):
    folder = ROOT/case
    frozen = json.loads((folder/'frozen.json').read_text())
    for path, digest in (frozen['files'] | frozen['result_files']).items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    names = list(dict.fromkeys(['baseline', frozen['selected'], *[n for n in
        ['hessian_0.01', 'h1_0.01', 'sweep_flat', 'lbfgs_hessian_continue', 'adam_logpen',
         'conditional_0.1', 'conditional_1', 'conditional_10', 'quadratic_0.0001', 'quadratic_0.001'] if n in frozen['names']]]))
    models = {name: load_tt(folder/name/'fit.npz') for name in names}
    models['uniform_start'] = init_tt_constant(np.asarray(models['baseline'].domain), rank=32, modes=49)
    if case == 'swirl4':
        path = Path('artifacts/direct_optimization/swirl_free4/squared_n49_r32_l0.0001_b1024_s0_lbfgs/fit.npz')
        models['previous_direct'] = load_tt(path)
    elif case == 'swirl10':
        models['previous_direct'] = load_tt('artifacts/direct_optimization/swirl_free10/squared_n49_r32_l0.0001_b256_s0_lbfgs/fit.npz')
    # Only now import and evaluate the known synthetic target.
    if case.startswith('swirl'):
        from swirl_figures import integrated_l2

        from ftde.swirl import (
            curve,
            sample_swirl,
            swirl_log_density,
            swirl_pair_marginal,
        )
        dim = int(case[5:])
        t = np.linspace(0, 1, 2001)
        profile_x = curve(t, dim)
        profile_truth = np.exp(swirl_log_density(profile_x, order=1600))
        test = sample_swirl(731983, 16384, dim)
        grid = np.linspace(-2.6, 2.6, 181)
        truth = {f'pair{i}{j}': swirl_pair_marginal(dim, i, j, grid, grid) for i, j in [(0, 1), (2, 3)]}
        np.savez_compressed(folder/'truth.npz', t=t, profile=profile_truth, grid=grid, **truth)
    else:
        from evaluate_gl_l2 import integrated_terms, relative_l2_from_terms

        from ftde.ginzburg_landau import GL_BETA, ginzburg_landau_log_normalizer
        dim = 6
        log_z = ginzburg_landau_log_normalizer(dim)
        p2 = float(np.exp(ginzburg_landau_log_normalizer(dim, beta=2*GL_BETA)-2*log_z))
        grid = np.linspace(-2., 2., 181)
        from gl_common import _true_marginal_grid
        truth = {f'pair{i}{j}': _true_marginal_grid(dim, i, j, np.tile([-2., 2.], (dim, 1)), log_z, n_grid=181) for i, j in [(0, 1), (2, 3)]}
        np.savez_compressed(folder/'truth.npz', grid=grid, **truth)
    results = {}
    for name, model in models.items():
        out = folder/name
        out.mkdir(exist_ok=True)
        if (out/'evaluation.json').exists():
            results[name] = json.loads((out/'evaluation.json').read_text())
            continue
        problem = DirectTrainingProblem(model)
        if case.startswith('swirl'):
            result = integrated_l2(folder, model, dim, name)
            if result['relative_squared_l2'] < -1e-9 or result['path_quadrature_relative_change'] > 1e-6:
                raise FloatingPointError('Swirl integration failed convergence check')
            test_logs = np.concatenate([np.asarray(problem.logs(problem.params, problem.features(test[k:k+1024])))
                                        for k in range(0, len(test), 1024)])
            profile = np.exp(np.asarray(problem.logs(problem.params, problem.features(profile_x))))
            result.update(test_nll=-float(test_logs.mean()), test_count=len(test),
                          curve_minimum_ratio=float(np.min(profile/profile_truth)),
                          curve_fraction_below_half=float(np.mean(profile < .5*profile_truth)))
            np.savez_compressed(out/'profile.npz', density=profile)
        else:
            q2, pq = map(float, jax.jit(lambda model=model: integrated_terms(model, 98, log_z))())
            q2f, pqf = map(float, jax.jit(lambda model=model: integrated_terms(model, 147, log_z))())
            result = {'relative_l2': relative_l2_from_terms(q2f, pqf, p2),
                      'quadrature_relative_change': max(abs(q2-q2f), abs(pq-pqf))/p2}
        marginal = {f'pair{i}{j}': squared_pair_grid(model, i, j, grid, grid) for i, j in [(0, 1), (2, 3)]}
        np.savez_compressed(out/'marginals.npz', grid=grid, **marginal)
        result['model_coefficients_sha256'] = hashlib.sha256(b''.join(np.asarray(c.coeffs).tobytes() for c in model.cores)).hexdigest()
        write_record(out/'evaluation.json', result)
        results[name] = result
        print('EVALUATION', case, name, result, flush=True)
        jax.clear_caches()
    write_record(folder/'evaluation.json', {'selected': frozen['selected'], 'models': results,
                 'selection_frozen_before_truth': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases', nargs='+')
    for case in parser.parse_args().cases:
        run(case)
