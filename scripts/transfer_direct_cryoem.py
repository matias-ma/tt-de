"""Direct TT learning on cryo-EM from an elementary start; no signal pilots.

Reuse the existing observation splits, boxes, and prescribed measurement
noise. No state labels or oracle densities are read. Compare both swirl
curvature settings by noisy validation, then refine the selected trained TT.
"""

import argparse
import datetime
import fcntl
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from fit_direct_optimization import record
from refresh_density_fits import fit_candidate, read_model, save_model
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from ftde.full_batch_optimization import fit_noisy_tt_lbfgs
from ftde.marginals import squared_pair_grid
from ftde.noisy_model_selection import select_by_paired_noisy_nll
from ftde.simple_initialization import init_tt_constant
from ftde.sobolev import AmplitudeSobolevPenalty
from ftde.tt_convolution import TTConvolution

BASES = {'dim4': 'artifacts/cryoem_continuity/gallery_dim4',
         'path_dim4': 'artifacts/cryoem_continuity/gallery_path_dim4',
         'dim10': 'artifacts/cryoem_stability/gallery_dim10',
         'dim20': 'artifacts/cryoem_stability/gallery_dim20'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_time(minutes):
    """Preserve completed Adam fits instead of losing an unsaved refinement."""
    job = os.environ.get('SLURM_JOB_ID')
    if not job:
        return
    description = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
    match = re.search(r'\bEndTime=(\S+)', description)
    if match and match[1] != 'Unknown':
        end = datetime.datetime.fromisoformat(match[1]).astimezone()
        remaining = (end-datetime.datetime.now(datetime.timezone.utc)).total_seconds()/60
        print('Allocation minutes remaining', round(remaining, 1), flush=True)
        if remaining < minutes:
            raise RuntimeError(f'Resume this saved run with at least {minutes} allocation minutes for full-data refinement')


def run(case):
    base = Path(BASES[case])
    root = Path('artifacts/direct_optimization')/f'cryoem_{case}'
    root.mkdir(exist_ok=True)
    data = np.load(base/'data.npz')
    train, valid = data['train'], data['valid']
    # Only the box is used to construct the new initializer.
    domain = np.load(base/'selected.npz')['domain']
    noise_path = Path('artifacts/swirl_improvement')/f'cryoem_{case}'/'noise.npz'
    noise = np.load(noise_path)
    covs, weights = noise['covariances'], noise['weights']
    scales = np.sqrt(np.einsum('c,cii->i', weights, covs))
    kernel = read_model(base/'kernel.npz')
    config = {'base': str(base), 'data_sha256': digest(base/'data.npz'),
              'kernel_sha256': digest(base/'kernel.npz'), 'noise_sha256': digest(noise_path),
              'historical_sha256': digest(base/'selected.npz'), 'rank': 32, 'modes': 49, 'seed': 0,
              'steps': 4000, 'batch': 256, 'learning_rate': .001, 'strengths': [.0001, .001],
              'lbfgs_iterations': 40, 'training_count': len(train), 'validation_count': len(valid),
              'initialization': 'constant plus small random perturbation; box only',
              'uses_signal_pilot_or_oracle_or_labels': False,
              'selection': 'paired one-SE noisy validation, then common physical amplitude H2'}
    if (root/'config.json').exists() and json.loads((root/'config.json').read_text()) != config:
        raise ValueError('Cryo-EM transfer configuration changed')
    record(root/'config.json', config)

    # Independently check that the saved covariance specification describes
    # the frozen TT kernel actually used in likelihood fitting.
    rng = np.random.default_rng(91473)
    labels = rng.choice(len(weights), 512, p=weights)
    points = np.einsum('nij,nj->ni', np.linalg.cholesky(covs)[labels], rng.normal(size=(512, len(domain))))
    analytic = np.exp(logsumexp(np.stack([np.log(w)+multivariate_normal.logpdf(points, cov=c)
        for w, c in zip(weights, covs, strict=True)]), axis=0))
    actual = np.asarray(kernel(jnp.asarray(points)))
    discrepancy = float(np.max(np.abs(actual/analytic-1)))
    if np.any(actual <= 0) or discrepancy > 1e-6:
        raise FloatingPointError('Prescribed cryo-EM noise and saved kernel disagree')
    record(root/'noise_check.json', {'max_relative_error': discrepancy, 'samples': len(points)})

    for strength in config['strengths']:
        folder = root/f'constant_h2_{strength:g}'
        if (folder/'result.json').exists():
            continue
        folder.mkdir(exist_ok=True)
        initial = init_tt_constant(domain, rank=32, modes=49, seed=0)
        save_model(folder/'init.npz', initial)
        penalty = AmplitudeSobolevPenalty.from_tt(initial, order=2, length_scales=scales)
        fit_candidate(initial, train, valid, folder, kernel=kernel, penalty=penalty,
                      strength=strength, steps=4000, lr=.001, batch_size=256,
                      checkpoint_rule='regularized', preconditioning='sobolev', seed=0)
        jax.clear_caches()

    def energy(model):
        return float(AmplitudeSobolevPenalty.from_tt(model, order=2, length_scales=scales)(model))

    adam_names = [f'constant_h2_{v:g}' for v in config['strengths']]
    adam_models = [read_model(root/n/'fit.npz') for n in adam_names]
    adam_logs = np.stack([np.load(root/n/'validation.npy') for n in adam_names])
    index, detail = select_by_paired_noisy_nll(adam_logs, [energy(m) for m in adam_models])
    source_name = adam_names[index]
    record(root/'adam_selection.json', {'selected': source_name, 'selection': detail})
    folder = root/'lbfgs'
    folder.mkdir(exist_ok=True)
    refinement = {'source': source_name, 'source_sha256': digest(root/source_name/'fit.npz'),
                  'strength': config['strengths'][index], 'iterations': 40}
    if (folder/'config.json').exists() and json.loads((folder/'config.json').read_text()) != refinement:
        raise ValueError('Changed selected continuation source')
    record(folder/'config.json', refinement)
    if not (folder/'result.json').exists():
        if case == 'dim20':
            require_time(50)
        initial = adam_models[index]
        save_model(folder/'init.npz', initial)
        penalty = AmplitudeSobolevPenalty.from_tt(initial, order=2, length_scales=scales)
        fit, last, logs, result = fit_noisy_tt_lbfgs(initial, train, valid, kernel, penalty,
            strength=refinement['strength'], iterations=40, callback=lambda r: print('LBFGS', case, r, flush=True))
        save_model(folder/'fit.npz', fit)
        save_model(folder/'last.npz', last)
        np.save(folder/'validation.npy', logs)
        record(folder/'result.json', result)

    old = read_model(base/'selected.npz')
    if not (root/'historical_validation.npy').exists():
        op = TTConvolution.from_tt(old, kernel).prepare(tolerance=1e-14)
        forward = jax.jit(lambda x: op.log_density(old, x))
        logs = np.concatenate([np.asarray(forward(jnp.asarray(valid[s:s+128]))) for s in range(0, len(valid), 128)])
        np.save(root/'historical_validation.npy', logs)
    names = ['historical', *adam_names, 'lbfgs']
    models = [old, *adam_models, read_model(folder/'fit.npz')]
    logs = np.stack([np.load(root/'historical_validation.npy'), *adam_logs, np.load(folder/'validation.npy')])
    roughness = [energy(model) for model in models]
    chosen, detail = select_by_paired_noisy_nll(logs, roughness)
    new_chosen, new_detail = select_by_paired_noisy_nll(logs[1:], roughness[1:])
    selection = {'selected': names[chosen], 'new_selected': names[1+new_chosen], 'names': names,
                 'validation_nll': (-logs.mean(axis=1)).tolist(), 'roughness': roughness,
                 'selection': detail, 'new_selection': new_detail, 'source': source_name,
                 'uses_clean_truth': False, 'historical_budget_differs': True}
    record(root/'selection.json', selection)
    fit_paths = [base/'selected.npz', *[root/n/'fit.npz' for n in names[1:]]]
    record(root/'frozen.json', {'files': {str(p): digest(p) for p in fit_paths}, 'selection': selection})
    for name, model in zip(names, models, strict=True):
        out = root/name
        out.mkdir(exist_ok=True)
        saved = {}
        for i, j in ((0, 1), (2, 3)):
            xs, ys = np.linspace(*domain[i], 180), np.linspace(*domain[j], 180)
            saved[f'pair{i}{j}'] = squared_pair_grid(model, i, j, xs, ys)
            saved[f'xs{i}{j}'], saved[f'ys{i}{j}'] = xs, ys
        np.savez(out/'marginals.npz', **saved)
    print('SELECTED', case, selection, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=BASES, required=True)
    case = parser.parse_args().case
    # Concurrent drivers may share the same allocation; each case has one writer.
    with (Path('artifacts/direct_optimization')/f'cryoem_{case}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run(case)
