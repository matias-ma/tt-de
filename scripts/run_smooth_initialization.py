"""Isolate smoother random starts at fixed noisy likelihood and regularization."""
import hashlib
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
from objective_study import CASES
from run_marginal_regularization import CASES_RUN, RECIPES, baseline, read
from run_marginal_regularization import ROOT as PREVIOUS

from ftde.direct_training import (
    DirectTrainingProblem,
    NoisyDirectTrainingProblem,
    load_tt,
    write_record,
)
from ftde.noisy_model_selection import select_by_paired_noisy_nll
from ftde.simple_initialization import init_tt_constant
from ftde.tt_convolution import TTConvolution

ROOT = Path('artifacts/smooth_initialization_v6')
VARIANTS = {'power4_seed0': {'spectral_decay': 2., 'seed': 0},
            'lowpass_seed0': {'spectral_scale': 4., 'seed': 0},
            'default_seed1': {'seed': 1}, 'lowpass_seed1': {'spectral_scale': 4., 'seed': 1}}


def run(case, name):
    if (ROOT/'frozen.json').exists():
        raise RuntimeError('Frozen study')
    out = ROOT/case/name
    if (out/'result.json').exists():
        return
    base = Path(CASES[case])
    data = np.load(base/'data.npz')
    train, valid = data['train'], data['valid']
    template = load_tt(baseline(case)/'fit.npz')
    recipe = read(PREVIOUS/case/'fresh_choices.json')['fresh_amplitude']
    strength, marginal = RECIPES[recipe]
    options = VARIANTS[name]
    initial = init_tt_constant(np.asarray(template.domain), rank=max(c.coeffs.shape[1] for c in template.cores),
        modes=template.cores[0].coeffs.shape[-1], **options)
    box = np.stack([np.minimum(train.min(0), valid.min(0))-.02, np.maximum(train.max(0), valid.max(0))+.02], axis=1)
    forward = TTConvolution.from_tt(initial, load_tt(base/'kernel.npz')).prepare(box, tolerance=1e-14)
    problem = NoisyDirectTrainingProblem(initial, forward, regularizer='hessian', strength=strength,
                                         marginal_strength=marginal)
    out.mkdir(parents=True, exist_ok=True)
    e = np.asarray(problem.energies(problem.params))
    write_record(out/'config.json', {'case': case, 'recipe': recipe, 'strength': strength, 'marginal_strength': marginal,
        'initialization': options, 'initial_hessian': float(e[3]/e[0]), 'initial_roughness_h1': float(e[1]/e[0]),
        'steps': 4000, 'learning_rate': .001, 'batch_size': 256, 'batch_seed': 0, 'uses_truth': False,
        'data_sha256': hashlib.sha256((base/'data.npz').read_bytes()).hexdigest()})
    problem.fit_adam(train, valid, out, steps=4000, batch_size=256, learning_rate=.001,
                     checkpoint_every=200, validation_batch_size=128)


def driver():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT/'frozen.json').exists():
        return
    previous = read(PREVIOUS/'frozen.json')
    for path, digest in previous['files'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    write_record(ROOT/'plan.json', {'cases': CASES_RUN, 'variants': VARIANTS,
        'seed1_controls': 'swirl only, default versus lowpass at fixed seed and batch sequence',
        'objective': 'same as preceding fresh_amplitude per case; selected using observations before truth',
        'primary': 'minimum noisy validation NLL', 'secondary': 'lowest marginal curvature within one paired SE',
        'unchanged': ['training and validation data', 'TT capacity', 'batch sequence', 'optimizer', 'regularizer'],
        'truth_evaluation_not_yet_run': True, 'storage_budget_MiB': 128})
    for case in CASES_RUN:
        names = list(VARIANTS) if case.startswith('swirl') else ['power4_seed0', 'lowpass_seed0']
        for name in names:
            if sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()) > 128*2**20:
                raise RuntimeError('Artifact budget exceeded')
            with (ROOT/f'{case}_{name}.log').open('a') as log:
                print('RUN', case, name, flush=True)
                subprocess.run([sys.executable, '-u', __file__, case, name], stdout=log, stderr=subprocess.STDOUT, check=True)
    files, cases, smooth = {}, {}, {}
    for case in CASES_RUN:
        # Original is the matched default-init fresh run, not the earlier strong-penalty fit.
        paths = {'original': PREVIOUS/case/'fresh_amplitude',
                 **{p.name: p for p in (ROOT/case).iterdir() if p.is_dir()}}
        models, scores, roughness = {}, [], []
        for name, folder in paths.items():
            logs = np.load(folder/'validation.npy')
            scores.append(logs)
            model = load_tt(folder/'fit.npz')
            problem = DirectTrainingProblem(model, strength=0., marginal_strength=1.)
            roughness.append(float(jax.jit(problem.marginal_penalty)(problem.params)))
            models[name] = {'path': str(folder/'fit.npz'), 'validation_nll': float(-logs.mean()),
                            'config': read(folder/'config.json'), 'result': read(folder/'result.json')}
            for p in [folder/'fit.npz', folder/'config.json', folder/'result.json', folder/'validation.npy']:
                files[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(CASES[case])/'data.npz', Path(CASES[case])/'kernel.npz', ROOT/'plan.json']:
            files[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        clusters = np.arange(len(scores[0])) % 200 if case.startswith('gl') else None
        chosen, detail = select_by_paired_noisy_nll(np.stack(scores), roughness, cluster_ids=clusters)
        smooth[case] = {'selected': list(models)[chosen], 'names': list(models), 'detail': detail}
        cases[case] = {'models': models, 'selected': min(models, key=lambda n: models[n]['validation_nll'])}
    write_record(ROOT/'frozen.json', {'cases': cases, 'files': files, 'truth_evaluation_not_yet_run': True})
    write_record(ROOT/'smooth_selection.json', {'cases': smooth, 'frozen_sha256': hashlib.sha256((ROOT/'frozen.json').read_bytes()).hexdigest(), 'truth_evaluation_not_yet_run': True})
    print('FROZEN', {c: v['selected'] for c, v in cases.items()}, flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        run(*sys.argv[1:])
    else:
        driver()
