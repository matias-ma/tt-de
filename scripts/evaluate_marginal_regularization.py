"""Post-freeze truth evaluation and explicit 1D oscillation diagnostics."""
import hashlib
from pathlib import Path

import jax
import numpy as np
from run_marginal_regularization import CASES_RUN, ROOT, read
from scipy.special import roots_legendre

from ftde.direct_training import DirectTrainingProblem, load_tt, write_record
from ftde.marginals import squared_univariate_grid
from ftde.noisy_model_selection import select_by_paired_noisy_nll


def validation_logs(folder):
    folder = Path(folder)
    if (folder/'validation.npy').exists():
        return np.load(folder/'validation.npy')
    return np.load(folder/'validation.npz')['logs']


def truth_marginals(case, grid):
    if case.startswith('swirl'):
        from ftde.swirl import TUBE_STD, curve, path_quadrature
        t, w = path_quadrature()
        centers = curve(t, 4)
        return np.array([np.exp(-.5*((grid[:, None]-centers[None, :, k])/TUBE_STD)**2)@w
                         /(np.sqrt(2*np.pi)*TUBE_STD) for k in range(4)])
    from ftde.ginzburg_landau import GL_BETA, GL_LAM, ginzburg_landau_log_normalizer
    dim = 6
    x, w = roots_legendre(240)
    x, w = x*3, w*3
    scale = GL_BETA*GL_LAM*(dim+1)**2/2
    edge = lambda a, b: np.exp(-scale*(a-b)**2)
    site = lambda a: np.exp(-GL_BETA*(1-a*a)**2/(4*GL_LAM))
    transition = edge(x[:, None], x[None, :])*(w*site(x))[None]
    def message(n):
        if not n:
            return edge(0., grid)
        state = edge(0., x)*w*site(x)
        for _ in range(n-1):
            state = state@transition
        return state@edge(x[:, None], grid[None])
    z = np.exp(ginzburg_landau_log_normalizer(dim))
    return np.array([message(k)*site(grid)*message(dim-1-k)/z for k in range(dim)])


def freeze_smooth_choices():
    frozen = read(ROOT/'frozen.json')
    for path, digest in frozen['files'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f'Frozen file changed: {path}')
    # Complete BOTH cases' observation-only secondary choices before any truth.
    smooth_path = ROOT/'smooth_selection.json'
    if not smooth_path.exists():
        choices = {}
        for case in CASES_RUN:
            models = frozen['cases'][case]['models']
            names, logs, roughness = list(models), [], []
            for name in names:
                folder = Path(models[name]['path']).parent
                logs.append(validation_logs(folder))
                problem = DirectTrainingProblem(load_tt(models[name]['path']), strength=0., marginal_strength=1.)
                roughness.append(float(jax.jit(problem.marginal_penalty)(problem.params)))
            clusters = np.arange(len(logs[0])) % 200 if case.startswith('gl') else None
            selected, detail = select_by_paired_noisy_nll(np.stack(logs), roughness, cluster_ids=clusters)
            choices[case] = {'selected': names[selected], 'names': names, 'detail': detail}
        write_record(smooth_path, {'cases': choices,
            'protocol_sha256': hashlib.sha256((ROOT/'secondary_selection_protocol.json').read_bytes()).hexdigest(),
            'frozen_sha256': hashlib.sha256((ROOT/'frozen.json').read_bytes()).hexdigest(),
            'truth_evaluation_not_yet_run': True})
        print('SMOOTH CHOICES', {c: x['selected'] for c, x in choices.items()}, flush=True)
    smooth_record = read(smooth_path)
    for key, path in [('protocol_sha256', ROOT/'secondary_selection_protocol.json'),
                      ('frozen_sha256', ROOT/'frozen.json')]:
        if key in smooth_record and smooth_record[key] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise RuntimeError('Secondary selection inputs changed')
    return frozen


def evaluate():
    import evaluate_noise_diagnosis as previous
    previous.ROOT = ROOT
    frozen = freeze_smooth_choices()
    for case in CASES_RUN:
        previous.evaluate(case)
        folder = ROOT/case
        evaluation = read(folder/'evaluation.json')
        grid = np.linspace(-2.6, 2.6, 801)
        truth = truth_marginals(case, grid)
        arrays = {'grid': grid, 'truth': truth}
        original_logs = np.load(folder/'test_logs_original.npy')
        reference_info = frozen['cases'][case]['models']['original']
        reference_validation = validation_logs(Path(reference_info['path']).parent)
        for name, info in frozen['cases'][case]['models'].items():
            model = load_tt(info['path'])
            curves = np.array([squared_univariate_grid(model, k, grid) for k in range(len(model.cores))])
            arrays[name] = curves
            problem = DirectTrainingProblem(model, strength=0., marginal_strength=1.)
            marginal_roughness = float(problem.marginal_penalty(problem.params))
            del problem
            error = np.sqrt(np.sum((curves-truth)**2, axis=1)/np.sum(truth**2, axis=1))
            metric = evaluation['models'][name]
            validation = validation_logs(Path(info['path']).parent)
            validation_delta = reference_validation-validation
            metric.update(validation_nll_minus_original=float(validation_delta.mean()),
                          paired_validation_se=previous.paired_se(validation_delta, case),
                          marginal_curvature=marginal_roughness,
                          univariate_relative_l2=error.tolist(), mean_univariate_relative_l2=float(error.mean()))
            delta = original_logs-np.load(folder/f'test_logs_{name}.npy')
            evaluation['test_comparisons'][name] = {'test_nll_minus_reference': float(delta.mean()),
                                                       'paired_se': previous.paired_se(delta, case)}
        evaluation['test_comparisons_reference'] = 'original'
        np.savez_compressed(folder/'univariate.npz', **arrays)
        write_record(folder/'evaluation.json', evaluation)
        print('MARGINALS', case, {n: (m['relative_l2'], m['mean_univariate_relative_l2'], m['marginal_curvature'])
              for n, m in evaluation['models'].items()}, flush=True)
    write_record(ROOT/'evaluation_complete.json', {'frozen_sha256': hashlib.sha256((ROOT/'frozen.json').read_bytes()).hexdigest()})


if __name__ == '__main__':
    evaluate()
