"""Measure train/validation gaps for frozen noise-free capacity controls."""
import json

import numpy as np
from objective_study import setup

from ftde.direct_training import DirectTrainingProblem, load_tt, write_record


def main():
    for case in ['gl6_free', 'swirl4_free']:
        folder, _, train, valid, _, _, _ = setup(case)
        frozen = json.loads((folder/'frozen.json').read_text())
        capacity = json.loads((folder/'capacity_selection.json').read_text())
        rows = {}
        for run in capacity['names']:
            assert run+'/nll' in frozen['names']
            problem = DirectTrainingProblem(load_tt(folder/run/'fit.npz'))
            losses = []
            for data in [train, valid]:
                total = sum(float(np.asarray(problem.logs(problem.params, problem.features(data[k:k+2048]))).sum())
                            for k in range(0, len(data), 2048))
                losses.append(-total/len(data))
            saved = json.loads((folder/run/'result.json').read_text())
            np.testing.assert_allclose(losses[1], saved['best_nll'], atol=1e-9, rtol=0)
            rows[run] = {'training_nll': losses[0], 'validation_nll': losses[1],
                         'validation_minus_training': losses[1]-losses[0]}
            print(case, run, rows[run], flush=True)
        write_record(folder/'generalization.json', {'uses_truth': False, 'models': rows,
                    'note': 'Observed train/validation gaps, not independent post-selection uncertainty estimates'})


if __name__ == '__main__':
    main()
