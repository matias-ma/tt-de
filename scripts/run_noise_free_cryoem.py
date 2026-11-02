"""Transfer the direct full-Hessian regularizer to prescribed-noise cryo-EM fits."""
import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np
from transfer_direct_cryoem import BASES

from ftde.direct_lbfgs import fit_lbfgs
from ftde.direct_training import NoisyDirectTrainingProblem, load_tt, write_record
from ftde.noisy_model_selection import select_by_paired_noisy_nll
from ftde.simple_initialization import init_tt_constant
from ftde.tt_convolution import TTConvolution

ROOT = Path('artifacts/noise_free_v2')


def run(case):
    base, folder = Path(BASES[case]), ROOT/f'cryoem_{case}'
    folder.mkdir(parents=True, exist_ok=True)
    with np.load(base/'data.npz') as data:
        train, valid = data['train'], data['valid']
    with np.load(base/'selected.npz') as saved:
        domain = saved['domain']
    kernel = load_tt(base/'kernel.npz')
    recipes = {}
    for synthetic in ('swirl4', 'swirl10'):
        frozen = json.loads((ROOT/synthetic/'frozen.json').read_text())
        result = json.loads((ROOT/synthetic/frozen['selected']/'result.json').read_text())
        reg, strength = result['regularizer'], result['strength']
        objective = 'logpen' if frozen['selected'].startswith('sweep') or 'logpen' in frozen['selected'] else 'nll'
        recipes[f'{reg}_{strength:g}_{objective}'] = {'regularizer': reg, 'strength': strength, 'objective': objective}
    protocol = {'source': str(base), 'data_sha256': hashlib.sha256((base/'data.npz').read_bytes()).hexdigest(),
                'kernel_sha256': hashlib.sha256((base/'kernel.npz').read_bytes()).hexdigest(),
                'recipes': recipes, 'selection_source': 'frozen observation-only swirl4 and swirl10 selection',
                'length_scales': (.05*np.ptp(domain, axis=1)).tolist(),
                'uses_signal_pilot': False, 'initialization': 'box-only near uniform',
                'steps': 4000, 'batch': 256, 'rank': 32, 'modes': 49,
                'refinement': 'lbfgs', 'refinement_iterations': 40 if case == 'dim20' else 100}
    write_record(folder/'protocol.json', protocol)
    for name, recipe in recipes.items():
        out = folder/name
        if (out/'result.json').exists():
            continue
        initial = init_tt_constant(domain, rank=32, modes=49)
        forward = TTConvolution.from_tt(initial, kernel).prepare(tolerance=1e-14)
        problem = NoisyDirectTrainingProblem(initial, forward, **recipe)
        problem.fit_adam(train, valid, out, steps=4000, batch_size=256, validation_batch_size=128)
        del problem, forward, initial
        jax.clear_caches()
    names = list(recipes)
    logs = np.stack([np.load(folder/n/'validation.npy') for n in names])
    roughness = []
    for name in names:
        r = json.loads((folder/name/'result.json').read_text())
        roughness.append(next(h['roughness_hessian'] for h in r['history'] if h['step'] == r['best_step']))
    chosen, detail = select_by_paired_noisy_nll(logs, roughness)
    write_record(folder/'adam_selection.json', {'selected': names[chosen], 'selection': detail})
    out = folder/'lbfgs'
    if not (out/'result.json').exists():
        initial = load_tt(folder/names[chosen]/'fit.npz')
        forward = TTConvolution.from_tt(initial, kernel).prepare(tolerance=1e-14)
        problem = NoisyDirectTrainingProblem(initial, forward, **recipes[names[chosen]])
        out.mkdir(exist_ok=True)
        write_record(out/'config.json', {'parent': names[chosen], 'parent_sha256': hashlib.sha256((folder/names[chosen]/'fit.npz').read_bytes()).hexdigest()})
        fit_lbfgs(problem, train, valid, out, iterations=protocol['refinement_iterations'], batch_size=128, validation_batch_size=128)
        jax.clear_caches()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cases', choices=BASES, nargs='+')
    for case in parser.parse_args().cases:
        run(case)
