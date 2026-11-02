"""Observation-only capacity and objective comparison; no synthetic truth evaluation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ftde.direct_training import (
    DirectTrainingProblem,
    NoisyDirectTrainingProblem,
    load_tt,
    write_record,
)
from ftde.kernel_score import draw_frequencies
from ftde.scored_training import fit_scored
from ftde.simple_initialization import init_tt_constant
from ftde.tt_convolution import TTConvolution

ROOT = Path('artifacts/objective_v3')
CASES = {'gl6_free': 'artifacts/figure_refresh/gl_free_dim6',
         'swirl4_free': 'artifacts/swirl/swirl_free_dim4',
         'gl6_paired': 'artifacts/figure_refresh/gl_paired_dim6',
         'gl6_mixture': 'artifacts/figure_refresh/gl_mixture_dim6',
         'swirl4_mixture': 'artifacts/swirl/swirl_mixture_dim4',
         'swirl10_mixture': 'artifacts/swirl/swirl_mixture_dim10'}


def noise_parameters(case, dim):
    if case.endswith('free'):
        return np.ones(1), np.zeros((1, dim, dim))
    if case.startswith('gl'):
        if case.endswith('mixture'):
            return np.array([.7, .3]), np.array([np.eye(dim)*s**2 for s in (.45, 1.35)])
        variance = 1.5**2*(.7*.3**2+.3*.9**2)
        cov = np.eye(dim)*variance
        for k in range(0, dim, 2):
            cov[k, k+1] = cov[k+1, k] = .5*variance
        return np.ones(1), cov[None]
    cov = np.zeros((3, dim, dim))
    cov[0] = np.eye(dim)*.22**2
    cov[1] = np.diag(np.resize([.45**2, .25**2], dim))
    for k in range(dim//2):
        cov[2, 2*k:2*k+2, 2*k:2*k+2] = .5**2*np.array([[1., .6*(-1)**k], [.6*(-1)**k, 1.]])
    return np.array([.5, .3, .2]), cov


def setup(case):
    folder, base = ROOT/case, Path(CASES[case])
    folder.mkdir(parents=True, exist_ok=True)
    with np.load(base/'data.npz') as data:
        train, valid = data['train'], data['valid']
    with np.load(base/'init.npz') as data:
        domain = data['domain']
    weights, covariances = noise_parameters(case, train.shape[1])
    protocol = {'case': case, 'source': str(base),
                'data_sha256': hashlib.sha256((base/'data.npz').read_bytes()).hexdigest(),
                'domain': domain.tolist(), 'noise_weights': weights.tolist(), 'noise_covariances': covariances.tolist(),
                'training_count': len(train), 'validation_count': len(valid),
                'uses_clean_truth': False, 'uses_signal_pilot': False,
                'initialization': 'constant plus random coefficients, perturbation .01, seed 0',
                'kernel': 'Gaussian bandwidth mixture (.2,.5,1) times sqrt(d) in training-standardized coordinates',
                'training_frequencies': 1024, 'validation_frequencies': 512,
                'frequency_seeds': [2701, 2702], 'regularizer': 'full amplitude Hessian, scales .05 box width',
                'selection': 'separate paired-one-SE NLL and kernel choices; prefer fewer coefficients',
                'truth_evaluation': 'only after all candidate and checkpoint hashes are frozen',
                'storage_budget_MiB': 512}
    if (folder/'protocol.json').exists():
        assert json.loads((folder/'protocol.json').read_text()) == protocol
    else:
        write_record(folder/'protocol.json', protocol)
    return folder, base, train, valid, domain, weights, covariances


def run(case, rank, modes, objective, strength, steps, kernel_weight, name=None):
    folder, base, train, valid, domain, weights, covariance = setup(case)
    name = name or f'{objective}_r{rank}_n{modes}_l{strength:g}_w{kernel_weight:g}'
    out = folder/name
    config = {'rank': rank, 'modes': modes, 'objective': objective, 'strength': strength, 'steps': steps,
              'kernel_weight': kernel_weight, 'batch_size': 1024 if case.endswith('free') else 256,
              'seed': 0, 'perturbation': .01, 'learning_rate': .001}
    if (out/'config.json').exists():
        assert json.loads((out/'config.json').read_text()) == config
    if (out/'result.json').exists():
        return
    if (folder/'frozen.json').exists():
        raise RuntimeError('Case frozen: use a new study for more fitting')
    if sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()) > 512*2**20:
        raise RuntimeError('Artifact storage budget exceeded')
    out.mkdir(exist_ok=True)
    write_record(out/'config.json', config)
    initial = init_tt_constant(domain, rank=rank, modes=modes)
    if case.endswith('free'):
        problem = DirectTrainingProblem(initial, regularizer='hessian', strength=strength)
    else:
        kernel = load_tt(base/'kernel.npz')
        obs = np.concatenate([train, valid])
        observation_domain = np.stack([obs.min(axis=0)-.02, obs.max(axis=0)+.02], axis=1)
        forward = TTConvolution.from_tt(initial, kernel).prepare(observation_domain, tolerance=1e-14)
        problem = NoisyDirectTrainingProblem(initial, forward, regularizer='hessian', strength=strength)
    fit_scored(problem, train, valid, out, draw_frequencies(train, 1024, 2701), draw_frequencies(train, 512, 2702),
               noise_weights=weights, noise_covariances=covariance, objective=objective, kernel_weight=kernel_weight,
               steps=steps, batch_size=config['batch_size'], validation_batch=128)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', choices=CASES)
    parser.add_argument('--rank', type=int, default=10)
    parser.add_argument('--modes', type=int, default=25)
    parser.add_argument('--objective', choices=['nll', 'mmd', 'hybrid'], default='nll',
                        help='Training objective (default: nll); mmd and hybrid are experimental opt-ins')
    parser.add_argument('--strength', type=float, default=.001)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--kernel-weight', type=float, default=100.)
    parser.add_argument('--name')
    args = parser.parse_args()
    run(args.case, args.rank, args.modes, args.objective, args.strength, args.steps, args.kernel_weight, args.name)
