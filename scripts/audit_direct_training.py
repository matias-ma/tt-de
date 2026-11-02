"""Evaluate full training objectives of frozen noise-free fits; no refitting."""

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from fit_direct_optimization import record
from refresh_density_fits import read_model

from ftde.sobolev import AmplitudeSobolevPenalty
from ftde.tt_density import squared_integral_tt

root = Path('artifacts/direct_optimization')
results = {}
for case in ('swirl_free4', 'swirl_free10'):
    folder = root/case
    problem = json.loads((folder/'problem.json').read_text())
    frozen = json.loads((folder/'frozen_evaluation.json').read_text())
    train = np.load(Path(problem['base'])/'data.npz')['train']
    scales = .05*np.ptp(np.asarray(problem['domain']), axis=1)
    for run in sorted(folder.glob('squared_*')):
        if not (run/'result.json').exists():
            continue
        path = run/'fit.npz'
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if frozen['files'][str(path)] != digest:
            raise ValueError('Frozen model changed')
        model = read_model(path)
        result = json.loads((run/'result.json').read_text())
        strength = (json.loads((run/'config.json').read_text())['strength'] if run.name.endswith('_lbfgs')
                    else result['strength'])
        norm = float(squared_integral_tt(model))
        @jax.jit
        def evaluate(x, model=model, norm=norm):
            return jnp.log(model(x)**2+1e-300)-jnp.log(norm)
        total = 0.
        for start in range(0, len(train), 1024):
            total += float(jnp.sum(evaluate(jnp.asarray(train[start:start+1024]))))
        roughness = float(AmplitudeSobolevPenalty.from_tt(model, order=2, length_scales=scales)(model))
        row = {'training_nll': -total/len(train), 'roughness': roughness,
               'training_objective': -total/len(train)+strength*roughness,
               'strength': strength, 'observations': len(train), 'fit_sha256': digest}
        results[f'{case}/{run.name}'] = row
        print(case, run.name, row, flush=True)
        jax.clear_caches()
record(root/'frozen_training_objectives.json', results)
