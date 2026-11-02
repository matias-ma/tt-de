"""Continue a directly trained squared TT using full-data L-BFGS."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from fit_direct_optimization import prepare_problem, record
from refresh_density_fits import read_model, save_model

from ftde.full_batch_optimization import fit_noisy_tt_lbfgs
from ftde.sobolev import AmplitudeSobolevPenalty

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, default=Path('artifacts/swirl/swirl_mixture_dim4'))
    p.add_argument('--root', type=Path, default=Path('artifacts/direct_optimization/swirl4'))
    p.add_argument('--source', default='squared_n25_r16_l0.0001_b256_s0')
    p.add_argument('--iterations', type=int, default=80)
    p.add_argument('--strength', type=float, default=.0001)
    args = p.parse_args()
    data, domain, covs, weights = prepare_problem(args.base, args.root)
    source = args.root/args.source/'fit.npz'
    out = args.root/(args.source+'_lbfgs')
    out.mkdir(exist_ok=True)
    config = {'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'iterations': args.iterations, 'strength': args.strength,
              'initialization': 'continuation of TT trained directly from constant, no pilot'}
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text()) != config:
        raise ValueError('Changed continuation configuration')
    record(out/'config.json', config)
    if not (out/'result.json').exists():
        initial = read_model(source)
        save_model(out/'init.npz', initial)
        penalty = AmplitudeSobolevPenalty.from_tt(initial, order=2,
                    length_scales=(.05*np.ptp(domain, axis=1) if covs is None else
                                   np.sqrt(np.einsum('c,cii->i', weights, covs))))
        kernel = read_model(args.base/'kernel.npz') if covs is not None else None
        fitted, last, logs, result = fit_noisy_tt_lbfgs(initial, data['train'], data['valid'],
                kernel, penalty, strength=args.strength, iterations=args.iterations,
                callback=lambda row: print('LBFGS', row, flush=True))
        save_model(out/'fit.npz', fitted)
        save_model(out/'last.npz', last)
        np.save(out/'validation.npy', logs)
        record(out/'result.json', result)
