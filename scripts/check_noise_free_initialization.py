"""Fixed-recipe initialization robustness; never reselect the published fit."""
import json
from pathlib import Path

import jax
import numpy as np

from ftde.core_sweeps import fit_core_sweeps
from ftde.direct_training import write_record
from ftde.simple_initialization import init_tt_constant

ROOT = Path('artifacts/noise_free_v2')


if __name__ == '__main__':
    for case in ('swirl4', 'swirl10'):
        frozen = json.loads((ROOT/case/'frozen.json').read_text())
        chosen = json.loads((ROOT/case/frozen['selected']/'result.json').read_text())
        assert frozen['selected'].startswith('sweep')
        base = Path('artifacts/swirl')/f'swirl_free_dim{case[5:]}'
        with np.load(base/'data.npz') as data:
            train, valid = data['train'], data['valid']
        with np.load(base/'init.npz') as data:
            box = data['domain']
        for seed in (1, 2):
            folder = ROOT/'initialization_check'/f'{case}_seed{seed}'
            if (folder/'result.json').exists():
                continue
            folder.mkdir(parents=True, exist_ok=True)
            write_record(folder/'protocol.json', {'frozen_recipe': frozen['selected'], 'seed': seed,
                         'uses_clean_truth': False, 'uses_signal_pilot': False, 'reselects_main_fit': False,
                         'strength': chosen['strength'], 'regularizer': chosen['regularizer']})
            initial = init_tt_constant(box, rank=32, modes=49, seed=seed, perturbation=.01)
            fit_core_sweeps(initial, train, valid, folder, strength=chosen['strength'],
                            regularizer=chosen['regularizer'], sweeps=20, newton_steps=5)
            jax.clear_caches()
