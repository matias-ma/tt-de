"""Check amplitude sign sectors on observations; no target density is read."""
import json
from pathlib import Path

import jax
import numpy as np

from ftde.direct_training import DirectTrainingProblem, amplitude, load_tt, write_record

ROOT = Path('artifacts/noise_free_v2')


if __name__ == '__main__':
    result = {}
    for case in ('swirl4', 'swirl10'):
        frozen = json.loads((ROOT/case/'frozen.json').read_text())
        names = [n for n in ['baseline', frozen['selected'], 'sweep_hessian_continue'] if n in frozen['names']]
        with np.load(Path('artifacts/swirl')/f'swirl_free_dim{case[5:]}'/'data.npz') as data:
            points = data['valid']
        result[case] = {}
        for name in names:
            problem = DirectTrainingProblem(load_tt(ROOT/case/name/'fit.npz'))
            values = np.asarray(jax.jit(amplitude)(problem.params, problem.features(points)))
            negative = float(np.mean(values < 0))
            result[case][name] = {'observations': len(values), 'minority_sign_fraction': min(negative, 1-negative),
                                  'exact_zeros': int(np.sum(values == 0)), 'finite_sample_check_only': True}
        print(case, result[case], flush=True)
    write_record(ROOT/'phase_audit.json', result)
