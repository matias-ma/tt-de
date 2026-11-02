"""Observation-only noise-free study; never imports a synthetic target density."""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from ftde.direct_training import DirectTrainingProblem, write_record
from ftde.simple_initialization import init_tt_constant

ROOT = Path('artifacts/noise_free_v2')
CASES = {'swirl4': Path('artifacts/swirl/swirl_free_dim4'),
         'swirl10': Path('artifacts/swirl/swirl_free_dim10')}


def specifications():
    common = {'modes': 49, 'rank': 32, 'steps': 4000, 'batch': 1024,
              'learning_rate': .001, 'batch_seed': 0, 'seed': 0, 'perturbation': .01,
              'objective': 'nll', 'regularizer': 'axis_h2', 'strength': .0001,
              'gradient_clip': 1., 'conditional_weight': 1.}
    changes = [('baseline', {}), ('unclipped_adam', {'gradient_clip': None}),
               ('long_adam', {'steps': 8000})]
    changes += [(f'{reg}_{strength:g}', {'regularizer': reg, 'strength': strength})
                for reg, values in [('h1', [.001, .01, .1]), ('hessian', [.0001, .001, .01])]
                for strength in values]
    changes += [(f'conditional_{weight:g}', {'objective': 'composite', 'conditional_weight': weight})
                for weight in [.1, 1., 10.]]
    changes += [(f'quadratic_{strength:g}', {'objective': 'l2', 'strength': strength})
                for strength in [.0001, .001]]
    changes += [(f'seed_{seed}', {'seed': seed}) for seed in (1, 2)]
    changes += [(f'perturbation_{value:g}', {'perturbation': value}) for value in (.001, .1)]
    return {name: common | delta for name, delta in changes}


def fit(case, names):
    ROOT.mkdir(exist_ok=True)
    base = CASES[case]
    source = base/'data.npz'
    with np.load(source) as data:
        train, valid = data['train'], data['valid']
    with np.load(base/'init.npz') as saved:
        domain = saved['domain']
    protocol = {'case': case, 'base': str(base), 'data_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'domain': domain.tolist(), 'train_count': len(train), 'validation_count': len(valid),
                'uses_clean_truth': False, 'uses_signal_pilot': False, 'specifications': specifications(),
                'selection': 'paired one-SE held-out joint NLL, then lower common full-Hessian energy',
                'initialization_seeds_only': 'minibatch seed fixed at zero when varying initializer seed',
                'storage': 'only compressed coefficient checkpoints and compact diagnostics; geometry stays in RAM'}
    folder = ROOT/case
    folder.mkdir(exist_ok=True)
    if (folder/'protocol.json').exists():
        assert json.loads((folder/'protocol.json').read_text()) == protocol
    write_record(folder/'protocol.json', protocol)
    for name in names:
        config = specifications()[name]
        if case != 'swirl4' and config['objective'] == 'l2':
            continue
        out = folder/name
        out.mkdir(exist_ok=True)
        if (out/'result.json').exists():
            continue
        used = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
        if used > 512*2**20:
            raise RuntimeError('Study storage budget of 512 MiB exceeded')
        write_record(out/'config.json', config)
        initial = init_tt_constant(domain, rank=config['rank'], modes=config['modes'],
                                   seed=config['seed'], perturbation=config['perturbation'])
        problem = DirectTrainingProblem(initial, strength=config['strength'], regularizer=config['regularizer'],
                    objective=config['objective'], conditional_weight=config['conditional_weight'])
        problem.fit_adam(train, valid, out, steps=config['steps'], batch_size=config['batch'],
                         learning_rate=config['learning_rate'], batch_seed=config['batch_seed'],
                         gradient_clip=config['gradient_clip'])
        del problem, initial
        jax.clear_caches()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=CASES, default='swirl4')
    parser.add_argument('--names', nargs='+', default=['baseline', 'unclipped_adam', 'hessian_0.0001',
        'hessian_0.001', 'h1_0.01', 'conditional_1', 'conditional_0.1'])
    args = parser.parse_args()
    fit(args.case, args.names)
