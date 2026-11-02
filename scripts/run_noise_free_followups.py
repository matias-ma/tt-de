"""Deterministic direct coefficient optimization, with elementary starts only."""
import argparse
from pathlib import Path

import jax
import numpy as np

from ftde.core_sweeps import fit_core_sweeps
from ftde.direct_lbfgs import fit_lbfgs
from ftde.direct_training import DirectTrainingProblem, load_tt, write_record
from ftde.simple_initialization import init_tt_constant

ROOT = Path('artifacts/noise_free_v2')


def run(case, names):
    base = Path('artifacts/swirl')/f'swirl_free_dim{case[5:]}' if case.startswith('swirl') else Path('artifacts/figure_refresh/gl_free_dim6')
    with np.load(base/'data.npz') as data:
        train, valid = data['train'], data['valid']
    with np.load(base/'init.npz') as data:
        domain = data['domain']
    for name in names:
        output = ROOT/case/name
        if (output/'result.json').exists():
            continue
        initial = init_tt_constant(domain, rank=32, modes=49, seed=0, perturbation=.01)
        reg = 'h1' if 'h1' in name else ('hessian' if 'hessian' in name else 'axis_h2')
        strength = .1 if reg == 'h1' else (.001 if reg == 'hessian' else .0001)
        if 'strong' in name:
            strength = .01
        elif 'medium' in name:
            strength = .003
        parent = None
        if 'continue' in name:
            parent = ROOT/case/({'h1': 'h1_0.1', 'hessian': 'hessian_0.001', 'axis_h2': 'baseline'}[reg])/'fit.npz'
            initial = load_tt(parent)
        config = {'name': name, 'regularizer': reg, 'strength': strength, 'parent': str(parent) if parent else None,
                  'uses_signal_pilot': False, 'uses_clean_truth': False,
                  'sweeps': 20 if 'long' in name else 6, 'adam_steps': 12000 if 'long' in name else 4000, 'learning_rate': .0003 if 'continue' in name else .001}
        output.mkdir(parents=True, exist_ok=True)
        write_record(output/'config.json', config)
        if name.startswith('sweep'):
            fit_core_sweeps(initial, train, valid, output, strength=strength, regularizer=reg, sweeps=config['sweeps'], newton_steps=5)
        else:
            problem = DirectTrainingProblem(initial, strength=strength, regularizer=reg, objective='logpen' if 'logpen' in name else 'nll')
            if name.startswith('lbfgs'):
                fit_lbfgs(problem, train, valid, output, iterations=300)
            else:
                problem.fit_adam(train, valid, output, steps=config['adam_steps'], batch_size=1024, learning_rate=config['learning_rate'])
        jax.clear_caches()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['swirl4', 'swirl10', 'gl6'], default='swirl4')
    parser.add_argument('--names', nargs='+', default=['sweep_flat', 'sweep_continue', 'sweep_hessian_continue'])
    args = parser.parse_args()
    run(args.case, args.names)
