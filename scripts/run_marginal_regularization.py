"""Observation-only noisy regularization study; freeze before evaluating truth."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from objective_study import CASES

from ftde.direct_training import NoisyDirectTrainingProblem, load_tt, write_record
from ftde.simple_initialization import init_tt_constant
from ftde.tt_convolution import TTConvolution

ROOT = Path('artifacts/marginal_regularization_v5')
RECIPES = {'control': (.001, 0.), 'weak': (.0001, 0.), 'weaker': (.00001, 0.),
           'marginal_low': (.0001, .001), 'marginal_medium': (.0001, .01),
           'marginal_high': (.0001, .1)}
CASES_RUN = ('swirl4_mixture', 'gl6_mixture')


def read(path):
    return json.loads(Path(path).read_text())


def baseline(case):
    capacity = 'r32_n49' if case.startswith('swirl') else 'r10_n25'
    return Path('artifacts/objective_v3')/case/f'nll_{capacity}_l0.001_w100'


def run(case, name, recipe, fresh=False):
    if (ROOT/'frozen.json').exists():
        raise RuntimeError('Study frozen; use a new root for further fitting')
    output = ROOT/case/name
    if (output/'result.json').exists():
        return
    if sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()) > 256*2**20:
        raise RuntimeError('Artifact budget exceeded')
    source = Path(CASES[case])
    data = np.load(source/'data.npz')
    train, valid = data['train'], data['valid']
    old = load_tt(baseline(case)/'fit.npz')
    initial = init_tt_constant(np.asarray(old.domain), rank=max(c.coeffs.shape[1] for c in old.cores),
                               modes=old.cores[0].coeffs.shape[-1], seed=0) if fresh else old
    obs_box = np.stack([np.minimum(train.min(0), valid.min(0))-.02,
                        np.maximum(train.max(0), valid.max(0))+.02], axis=1)
    forward = TTConvolution.from_tt(initial, load_tt(source/'kernel.npz')).prepare(obs_box, tolerance=1e-14)
    strength, marginal_strength = RECIPES[recipe]
    problem = NoisyDirectTrainingProblem(initial, forward, regularizer='hessian',
        strength=strength, marginal_strength=marginal_strength)
    output.mkdir(parents=True, exist_ok=True)
    config = {'case': case, 'recipe': recipe, 'strength': strength, 'marginal_strength': marginal_strength,
        'steps': 4000 if fresh else 2000, 'learning_rate': .001 if fresh else .0003, 'batch_size': 256,
        'source': str(source), 'data_sha256': hashlib.sha256((source/'data.npz').read_bytes()).hexdigest(),
        'initialization': 'constant plus small random coefficients' if fresh else str(baseline(case)/'fit.npz'),
        'uses_truth': False, 'uses_signal_pilot': False, 'batch_seed': 0}
    write_record(output/'config.json', config)
    problem.fit_adam(train, valid, output, steps=config['steps'], batch_size=256,
        learning_rate=config['learning_rate'], checkpoint_every=200, validation_batch_size=128)


def launch(case, name, recipe, fresh=False):
    log = ROOT/f'{case}_{name}.log'
    with log.open('a') as stream:
        command = [sys.executable, '-u', __file__, '--case', case, '--name', name, '--recipe', recipe]
        if fresh:
            command.append('--fresh')
        print('RUN', case, name, recipe, flush=True)
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)


def driver():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT/'frozen.json').exists():
        raise RuntimeError('Study already frozen')
    write_record(ROOT/'plan.json', {'recipes': RECIPES, 'cases': CASES_RUN,
        'continuation_steps': 2000, 'fresh_steps': 4000,
        'followup': 'fresh seed0 runs of best marginal and best non-marginal continuation recipes',
        'selection': 'minimum observed validation NLL, including original control',
        'truth': 'unavailable until every setting and checkpoint is frozen', 'storage_budget_MiB': 256})
    for case in CASES_RUN:
        for recipe in RECIPES:
            launch(case, recipe, recipe)
        groups = {'fresh_amplitude': ['control', 'weak', 'weaker'],
                  'fresh_marginal': ['marginal_low', 'marginal_medium', 'marginal_high']}
        chosen = {name: min(names, key=lambda n: read(ROOT/case/n/'result.json')['best_nll'])
                  for name, names in groups.items()}
        write_record(ROOT/case/'fresh_choices.json', chosen)
        for name, recipe in chosen.items():
            launch(case, name, recipe, fresh=True)
    files, cases = {}, {}
    for case in CASES_RUN:
        models = {}
        paths = {'original': baseline(case), **{p.name: p for p in (ROOT/case).iterdir() if p.is_dir()}}
        for name, folder in paths.items():
            logpath = folder/('validation.npz' if name == 'original' else 'validation.npy')
            logs = np.load(logpath)
            if name == 'original':
                logs = logs['logs']
            models[name] = {'path': str(folder/'fit.npz'), 'validation_nll': float(-logs.mean()),
                                'result': read(folder/'result.json'), 'config': read(folder/'config.json')}
            for p in (folder/'fit.npz', folder/'result.json', folder/'config.json', logpath):
                files[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(CASES[case])/'data.npz', Path(CASES[case])/'kernel.npz', ROOT/case/'fresh_choices.json']:
            files[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        cases[case] = {'models': models, 'selected': min(models, key=lambda n: models[n]['validation_nll'])}
    write_record(ROOT/'frozen.json', {'cases': cases, 'files': files, 'truth_evaluation_not_yet_run': True})
    print('FROZEN', {c: v['selected'] for c, v in cases.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=CASES_RUN)
    parser.add_argument('--name')
    parser.add_argument('--recipe', choices=RECIPES)
    parser.add_argument('--fresh', action='store_true')
    args = parser.parse_args()
    if args.case:
        run(args.case, args.name, args.recipe, args.fresh)
    else:
        driver()
