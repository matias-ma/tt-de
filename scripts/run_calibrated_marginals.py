"""Lighter marginal smoothing calibrated from a frozen fitted density, not truth."""
import hashlib
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
from objective_study import CASES
from run_marginal_regularization import CASES_RUN, read
from run_marginal_regularization import ROOT as PREVIOUS

from ftde.direct_training import (
    DirectTrainingProblem,
    NoisyDirectTrainingProblem,
    load_tt,
    write_record,
)
from ftde.tt_convolution import TTConvolution

ROOT = Path('artifacts/calibrated_marginals_v7')
BUDGETS = {'control': 0., 'marginal_light': .001, 'marginal_moderate': .003}


def run(case, name):
    if (ROOT/'frozen.json').exists():
        raise RuntimeError('Study frozen')
    out = ROOT/case/name
    if (out/'result.json').exists():
        return
    protocol = read(ROOT/case/'protocol.json')
    source = Path(CASES[case])
    data = np.load(source/'data.npz')
    train, valid = data['train'], data['valid']
    initial = load_tt(protocol['source_model'])
    marginal_strength = BUDGETS[name]/protocol['source_marginal_curvature']
    obs_box = np.stack([np.minimum(train.min(0), valid.min(0))-.02,
                        np.maximum(train.max(0), valid.max(0))+.02], axis=1)
    forward = TTConvolution.from_tt(initial, load_tt(source/'kernel.npz')).prepare(obs_box, tolerance=1e-14)
    problem = NoisyDirectTrainingProblem(initial, forward, regularizer='hessian',
                                         strength=protocol['strength'], marginal_strength=marginal_strength)
    out.mkdir(parents=True, exist_ok=True)
    write_record(out/'config.json', {**protocol, 'marginal_strength': marginal_strength,
        'initial_marginal_penalty_budget': BUDGETS[name], 'steps': 2000, 'learning_rate': .0001,
        'batch_size': 256, 'batch_seed': 0, 'uses_truth': False})
    problem.fit_adam(train, valid, out, steps=2000, learning_rate=.0001, batch_size=256,
                     checkpoint_every=200, validation_batch_size=128)


def driver():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT/'frozen.json').exists():
        return
    previous = read(PREVIOUS/'frozen.json')
    for path, digest in previous['files'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    write_record(ROOT/'plan.json', {'budgets_nats_at_source': BUDGETS,
        'strength_rule': 'budget divided by exact marginal curvature of preceding observation-selected fit',
        'source_rule': 'preceding regularization study minimum noisy validation NLL',
        'objective': 'same amplitude-Hessian strength as source, plus marginal curvature',
        'all_runs': '2000 Adam updates, learning rate .0001, identical batches and checkpoints',
        'primary': 'minimum observed validation NLL', 'truth_evaluation_not_yet_run': True})
    write_record(ROOT/'secondary_selection_protocol.json', {'rule': 'lowest marginal curvature within one paired SE of minimum validation NLL',
        'GL_clusters': 200, 'truth_evaluation_not_yet_run': True})
    for case in CASES_RUN:
        info = previous['cases'][case]
        source = info['models'][info['selected']]
        folder = ROOT/case
        folder.mkdir(exist_ok=True)
        problem = DirectTrainingProblem(load_tt(source['path']), strength=0., marginal_strength=1.)
        curvature = float(jax.jit(problem.marginal_penalty)(problem.params))
        if curvature <= 0:
            raise ValueError('Calibration requires positive source curvature')
        write_record(folder/'protocol.json', {'source_name': info['selected'], 'source_model': source['path'],
            'source_marginal_curvature': curvature, 'strength': source['config']['strength']})
        for name in BUDGETS:
            print('RUN', case, name, flush=True)
            with (ROOT/f'{case}_{name}.log').open('a') as log:
                subprocess.run([sys.executable, '-u', __file__, case, name], stdout=log, stderr=subprocess.STDOUT, check=True)
    files, cases = {}, {}
    for case in CASES_RUN:
        info = previous['cases'][case]
        source = Path(info['models'][info['selected']]['path']).parent
        paths = {'original': source, **{n: ROOT/case/n for n in BUDGETS}}
        models = {}
        for name, folder in paths.items():
            models[name] = {'path': str(folder/'fit.npz'), 'validation_nll': float(-np.load(folder/'validation.npy').mean()),
                            'config': read(folder/'config.json'), 'result': read(folder/'result.json')}
            for p in [folder/'fit.npz', folder/'config.json', folder/'result.json', folder/'validation.npy']:
                files[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(CASES[case])/'data.npz', Path(CASES[case])/'kernel.npz', ROOT/case/'protocol.json', ROOT/'plan.json']:
            files[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        cases[case] = {'models': models, 'selected': min(models, key=lambda n: models[n]['validation_nll'])}
    write_record(ROOT/'frozen.json', {'cases': cases, 'files': files, 'truth_evaluation_not_yet_run': True})
    print('FROZEN', {c: x['selected'] for c, x in cases.items()}, flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        run(*sys.argv[1:])
    else:
        driver()
