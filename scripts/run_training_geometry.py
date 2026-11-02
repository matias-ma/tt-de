"""Matched observation-only tests of whitening and polynomial continuation."""
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
from objective_study import CASES
from run_marginal_regularization import CASES_RUN, baseline, read

from ftde.direct_training import NoisyDirectTrainingProblem, load_tt, write_record
from ftde.simple_initialization import init_tt_constant
from ftde.tt_convolution import TTConvolution

ROOT = Path('artifacts/training_geometry_v8')
RECIPES = {'baseline': (1e-5, False), 'precondition': (.001, False),
           'curriculum': (1e-5, True), 'combined': (.001, True)}
SCHEDULES = {'swirl4_mixture': [(0, 13), (1000, 25), (2000, 49)],
             'gl6_mixture': [(0, 9), (1000, 17), (2000, 25)]}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(case, recipe, seed):
    seed = int(seed)
    if (ROOT/'frozen.json').exists():
        raise RuntimeError('Study frozen; use a new root')
    out = ROOT/case/f'{recipe}_seed{seed}'
    if (out/'result.json').exists():
        return
    source = Path(CASES[case])
    with np.load(source/'data.npz') as data:
        train, valid = data['train'], data['valid']
    template = load_tt(baseline(case)/'fit.npz')
    initial = init_tt_constant(np.asarray(template.domain),
        rank=max(c.coeffs.shape[1] for c in template.cores), modes=template.cores[0].coeffs.shape[-1], seed=seed)
    box = np.stack([np.minimum(train.min(0), valid.min(0))-.02,
                    np.maximum(train.max(0), valid.max(0))+.02], axis=1)
    forward = TTConvolution.from_tt(initial, load_tt(source/'kernel.npz')).prepare(box, tolerance=1e-14)
    beta, curriculum = RECIPES[recipe]
    schedule = SCHEDULES[case] if curriculum else None
    problem = NoisyDirectTrainingProblem(initial, forward, regularizer='hessian', strength=1e-5,
                                         precondition_strength=beta)
    out.mkdir(parents=True, exist_ok=True)
    write_record(out/'config.json', {'case': case, 'recipe': recipe, 'seed': seed, 'strength': 1e-5,
        'marginal_strength': 0., 'precondition_strength': beta, 'mode_schedule': schedule,
        'selection_start': 2000, 'steps': 4000, 'learning_rate': .001, 'batch_size': 256, 'batch_seed': 0,
        'uses_truth': False, 'data_sha256': digest(source/'data.npz'),
        'initialization': 'constant plus small random coefficients; seed is independent of batch seed'})
    problem.fit_adam(train, valid, out, steps=4000, learning_rate=.001, batch_size=256,
        checkpoint_every=200, validation_batch_size=128, mode_schedule=schedule, selection_start=2000)


def launch(case, recipe, seed):
    print('RUN', case, recipe, seed, flush=True)
    with (ROOT/f'{case}_{recipe}_seed{seed}.log').open('a') as log:
        subprocess.run([sys.executable, '-u', __file__, case, recipe, str(seed)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def driver():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT/'frozen.json').exists():
        raise RuntimeError('Study already frozen')
    previous = read('artifacts/marginal_regularization_v5/frozen.json')
    for path, sha in previous['files'].items():
        if digest(path) != sha:
            raise RuntimeError(f'Previous frozen file changed: {path}')
    write_record(ROOT/'plan.json', {'recipes': RECIPES, 'schedules': SCHEDULES, 'cases': CASES_RUN,
        'objective': 'noisy NLL plus amplitude Hessian at strength 1e-5, no marginal penalty',
        'matched': ['data', 'capacity', 'batch sequence', '4000 updates', 'global cosine learning rate',
                    'checkpoint cadence', 'selection restricted to steps >= 2000'],
        'reuse': 'seed0 full-mode baseline from v5; require its selected step >= 2000 and matching config',
        'followup': 'repeat lowest-validation-NLL new seed0 recipe and baseline at seed1 in each case',
        'primary': 'minimum observed validation NLL across matched fits',
        'truth_evaluation_not_yet_run': True})
    write_record(ROOT/'secondary_selection_protocol.json', {
        'rule': 'lowest marginal curvature within one paired SE of minimum validation NLL',
        'GL_clusters': 200, 'truth_evaluation_not_yet_run': True})
    paths_by_case = {}
    for case in CASES_RUN:
        control = Path('artifacts/marginal_regularization_v5')/case/'fresh_amplitude'
        config, result = read(control/'config.json'), read(control/'result.json')
        for key, expected in {'strength': 1e-5, 'marginal_strength': 0., 'steps': 4000,
                              'learning_rate': .001, 'batch_size': 256, 'batch_seed': 0}.items():
            if config[key] != expected:
                raise RuntimeError(f'Unmatched old control: {key}')
        if result['best_step'] < 2000 or config['data_sha256'] != digest(Path(CASES[case])/'data.npz'):
            raise RuntimeError('Old control cannot be reused with this selection window/data')
        for recipe in RECIPES:
            if recipe != 'baseline':
                launch(case, recipe, 0)
        chosen = min(['precondition', 'curriculum', 'combined'],
                     key=lambda n: read(ROOT/case/f'{n}_seed0'/'result.json')['best_nll'])
        write_record(ROOT/case/'repeat_choice.json', {'recipe': chosen,
            'rule': 'minimum observed validation NLL among new seed0 recipes', 'uses_truth': False})
        for recipe in ['baseline', chosen]:
            launch(case, recipe, 1)
        paths_by_case[case] = {'original': control,
            **{p.name: p for p in (ROOT/case).iterdir() if p.is_dir()}}
    files, cases = {}, {}
    for case, paths in paths_by_case.items():
        models = {}
        for name, folder in paths.items():
            models[name] = {'path': str(folder/'fit.npz'),
                'validation_nll': float(-np.load(folder/'validation.npy').mean()),
                'config': read(folder/'config.json'), 'result': read(folder/'result.json')}
            for p in [folder/'fit.npz', folder/'config.json', folder/'result.json', folder/'validation.npy']:
                files[str(p)] = digest(p)
        for p in [Path(CASES[case])/'data.npz', Path(CASES[case])/'kernel.npz',
                  ROOT/case/'repeat_choice.json', ROOT/'plan.json', ROOT/'secondary_selection_protocol.json']:
            files[str(p)] = digest(p)
        cases[case] = {'models': models, 'selected': min(models, key=lambda n: models[n]['validation_nll'])}
    for p in [Path(__file__), Path('src/ftde/direct_training.py'), Path('src/ftde/mode_schedule.py')]:
        files[str(p)] = digest(p)
    write_record(ROOT/'frozen.json', {'cases': cases, 'files': files, 'truth_evaluation_not_yet_run': True})
    print('FROZEN', {c: v['selected'] for c, v in cases.items()}, flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        run(*sys.argv[1:])
    else:
        driver()
