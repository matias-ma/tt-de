"""Run a compact capacity grid and observation-selected objective comparisons."""
import json
import subprocess
import sys

import numpy as np
from objective_study import ROOT

from ftde.direct_training import write_record
from ftde.noisy_model_selection import select_by_paired_noisy_nll


def run(case, rank, modes, objective='nll', strength=.001, kernel_weight=100.):
    name = f'{objective}_r{rank}_n{modes}_l{strength:g}_w{kernel_weight:g}'
    command = [sys.executable, '-u', 'scripts/objective_study.py', case, '--rank', str(rank), '--modes', str(modes),
               '--objective', objective, '--strength', str(strength), '--kernel-weight', str(kernel_weight)]
    print('RUN', case, name, flush=True)
    with (ROOT/f'{case}_{name}.log').open('a') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    capacities = {'gl6_free': [(10, 25), (5, 25), (10, 17), (10, 49), (20, 25), (32, 25), (32, 49)],
                  'swirl4_free': [(10, 25), (10, 49), (20, 49), (32, 49)],
                  'gl6_paired': [(10, 25), (10, 49), (20, 25)],
                  'gl6_mixture': [(10, 25), (10, 49), (20, 25)],
                  'swirl4_mixture': [(10, 25), (10, 49), (20, 25)],
                  'swirl10_mixture': [(10, 25), (10, 49), (20, 25)]}
    ROOT.mkdir(exist_ok=True)
    plan = {'capacity_grid': capacities, 'steps_per_fit': 4000,
            'alternative_objectives_at_observation_selected_capacity': [['mmd', 100.], ['hybrid', 10.], ['hybrid', 100.]],
            'regularizer_grid_at_selected_capacity': [.0001, .01],
            'all_starts': 'near uniform; no fitted signal pilot', 'uses_truth': False}
    plan = json.loads(json.dumps(plan))
    if (ROOT/'plan.json').exists():
        assert json.loads((ROOT/'plan.json').read_text()) == plan
    else:
        write_record(ROOT/'plan.json', plan)
    for case, pairs in capacities.items():
        for rank, modes in pairs:
            run(case, rank, modes)
        folder = ROOT/case
        names = [f'nll_r{r}_n{n}_l0.001_w100' for r, n in pairs]
        logs = np.stack([np.load(folder/n/'validation.npz')['logs'] for n in names])
        dim = json.loads((folder/'protocol.json').read_text())['domain'].__len__()
        counts = [2*r*n+(dim-2)*r*r*n for r, n in pairs]
        chosen, detail = select_by_paired_noisy_nll(logs, counts)
        rank, modes = pairs[chosen]
        write_record(folder/'capacity_selection.json', {'selected': names[chosen], 'rank': rank, 'modes': modes,
                                                       'names': names, 'details': detail})
        print('CAPACITY', case, rank, modes, flush=True)
        for objective, weight in plan['alternative_objectives_at_observation_selected_capacity']:
            run(case, rank, modes, objective=objective, kernel_weight=weight)
        for strength in plan['regularizer_grid_at_selected_capacity']:
            run(case, rank, modes, strength=strength)


if __name__ == '__main__':
    main()
