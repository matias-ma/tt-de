"""Truth diagnostics after both observation-based selections are frozen."""
import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np
from objective_study import setup

from ftde.direct_training import (
    DirectTrainingProblem,
    NoisyDirectTrainingProblem,
    load_tt,
    write_record,
)
from ftde.kernel_score import (
    FourierKernelScore,
    draw_frequencies,
    empirical_characteristic,
    gaussian_mixture_characteristic,
)
from ftde.marginals import squared_pair_grid
from ftde.tt_convolution import TTConvolution


def model_path(folder, name):
    run, checkpoint = name.rsplit('/', 1)
    return folder/run/('fit.npz' if checkpoint == 'nll' else 'kernel_fit.npz')


def run(case):
    folder, base, train, _, domain, weights, covariances = setup(case)
    frozen = json.loads((folder/'frozen.json').read_text())
    for path, digest in frozen['files'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    capacity = json.loads((folder/'capacity_selection.json').read_text())
    names = [frozen['selected_nll'], frozen['selected_kernel'], capacity['selected']+'/nll']
    for name in frozen['names']:
        if name.startswith('mmd_') and name.endswith('/kernel'):
            names.append(name)
        if name.startswith('hybrid_') and name.endswith('/nll'):
            names.append(name)
    if case.endswith('free'):
        names += [name+'/nll' for name in capacity['names']]
    # Include the NLL-only control at the transferred swirl capacity even
    # when neither validation criterion selects it. This separates the
    # objective comparison from changing rank and polynomial resolution.
    extension_path = folder.parent/'capacity_extension.json'
    if extension_path.exists():
        extension = json.loads(extension_path.read_text())
        if case in extension['cases']:
            control = f'nll_r{extension["rank"]}_n{extension["modes"]}_l{extension["strength"]:g}_w100/nll'
            assert control in frozen['names']
            names.append(control)
    names = list(dict.fromkeys(names))
    # The known density and samplers are only imported after the frozen hashes
    # are verified. They never feed the training or selection scripts.
    dim = len(domain)
    swirl = case.startswith('swirl')
    if swirl:
        from swirl_figures import integrated_l2

        from ftde.swirl import sample_swirl, swirl_pair_marginal
        clean = sample_swirl(732981, 4096, dim)
    else:
        from evaluate_gl_l2 import integrated_terms, relative_l2_from_terms

        from ftde.ginzburg_landau import (
            GL_BETA,
            ginzburg_landau_log_normalizer,
            sample_ginzburg_landau,
        )
        clean = np.asarray(sample_ginzburg_landau(732981, 4096, dim))
        log_z = ginzburg_landau_log_normalizer(dim)
        p2 = float(np.exp(ginzburg_landau_log_normalizer(dim, beta=2*GL_BETA)-2*log_z))
    test = np.array(clean)
    if not case.endswith('free'):
        rng = np.random.default_rng(732982)
        labels = rng.choice(len(weights), len(test), p=weights)
        test += np.einsum('nij,nj->ni', np.linalg.cholesky(covariances)[labels], rng.normal(size=test.shape))
    frequencies = draw_frequencies(train, 1024, 2703)
    target = empirical_characteristic(test, frequencies)
    noise = gaussian_mixture_characteristic(frequencies, weights, covariances)
    grid = np.linspace(-2.6 if swirl else -2.5, 2.6 if swirl else 2.5, 181)
    if not (folder/'truth.npz').exists():
        if swirl:
            truth = {f'pair{i}{j}': swirl_pair_marginal(dim, i, j, grid, grid) for i, j in [(0, 1), (2, 3)]}
        else:
            from gl_common import _true_marginal_grid
            truth = {f'pair{i}{j}': _true_marginal_grid(dim, i, j, np.tile([grid[0], grid[-1]], (dim, 1)), log_z, n_grid=len(grid))
                     for i, j in [(0, 1), (2, 3)]}
        np.savez_compressed(folder/'truth.npz', grid=grid, **truth)
    results = {}
    for name in names:
        path = model_path(folder, name)
        out = folder/name.rsplit('/', 1)[0]
        suffix = name.rsplit('/', 1)[1]
        evaluation_file = out/f'evaluation_{suffix}.json'
        if evaluation_file.exists():
            results[name] = json.loads(evaluation_file.read_text())
            continue
        model = load_tt(path)
        if swirl:
            result = integrated_l2(folder, model, dim, name)
            if result['path_quadrature_relative_change'] > 1e-6 or result['relative_squared_l2'] < -1e-9:
                raise FloatingPointError('Swirl error quadrature failed')
        else:
            n = max(c.coeffs.shape[-1] for c in model.cores)
            order = max(98, 2*n)
            q2, pq = map(float, jax.jit(lambda model=model, order=order: integrated_terms(model, order, log_z))())
            q2fine, pqfine = map(float, jax.jit(lambda model=model, order=order: integrated_terms(model, order*3//2, log_z))())
            result = {'relative_l2': relative_l2_from_terms(q2fine, pqfine, p2),
                      'quadrature_relative_change': max(abs(q2-q2fine), abs(pq-pqfine))/p2}
            if result['quadrature_relative_change'] > 1e-6:
                raise FloatingPointError('GL error quadrature failed')
        if case.endswith('free'):
            problem = DirectTrainingProblem(model)
        else:
            forward = TTConvolution.from_tt(model, load_tt(base/'kernel.npz')).prepare(
                np.stack([test.min(axis=0)-.02, test.max(axis=0)+.02], axis=1), tolerance=1e-14)
            problem = NoisyDirectTrainingProblem(model, forward)
        logs = np.concatenate([np.asarray(problem.logs(problem.params, problem.features(test[k:k+128]))) for k in range(0, len(test), 128)])
        scorer = FourierKernelScore(problem, frequencies, noise_characteristic=noise)
        prediction = scorer.predict(problem.params, batch_size=32)
        # Reintegrate independent frequencies at twice the quadrature order.
        refined = FourierKernelScore(problem, frequencies[:32], noise_characteristic=noise[:32], order_multiplier=2)
        error = float(np.max(np.abs(prediction[:32]-refined.predict(problem.params, batch_size=32))))
        if error > 1e-9:
            raise FloatingPointError('Fourier integration failed refinement')
        result.update(test_nll=-float(logs.mean()), test_kernel_discrepancy=float(np.mean(np.abs(prediction-target)**2)),
                      test_count=len(test), fourier_refinement_max_error=error,
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        marginal = {f'pair{i}{j}': squared_pair_grid(model, i, j, grid, grid) for i, j in [(0, 1), (2, 3)]}
        np.savez_compressed(out/f'marginals_{suffix}.npz', grid=grid, **marginal)
        write_record(evaluation_file, result)
        results[name] = result
        print('EVALUATED', case, name, result, flush=True)
        del problem, scorer, refined, model
        jax.clear_caches()
    write_record(folder/'evaluation.json', {'models': results, 'selected_nll': frozen['selected_nll'],
                 'selected_kernel': frozen['selected_kernel'], 'selection_frozen_before_truth': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases', nargs='+')
    for case in parser.parse_args().cases:
        run(case)
